"""QA-LoRA for EfficientNet-B0 — faithful to Xu et al., ICLR 2024.

Real QA-LoRA:
  - Group-wise integer quantization: each output channel's weights split into L groups,
    each group has its own learnable scale and zero-point (increases quantization DOF).
    Weights are stored as actual INT4 integers (range [-8, 7]) and frozen at init
    (true quantized base, not fake-quantized FP32). During forward, weights are
    dequantized with the current (learnable) scale/zero-point:
        W = (weight_q - zero_point) * scale
    This is a fake-quant dequant in the forward pass, applied on top of a true-quant
    base — the same pattern used in QA-LoRA's official implementation, where the
    quantized base is loaded from disk and only the LoRA adapters (and quant params)
    receive gradients.
  - Zero-point merge rule: after fine-tuning, zero-points are updated to absorb LoRA adapters
    into quantized weights without FP16 fallback (paper's Algorithm 1).

Reference Algorithm 1 from paper (adapted for Conv2d):
  QA = nn.AdaptiveAvgPool1d(L)
  lora_A = (L, r)       -- grouped, smaller than standard (D_in, r)
  lora_B = (C_out, r)   -- standard
  quantization: W_q = clamp(round(W / alpha_j + beta_j), -8, 7)  -- INT4 asymmetric
  merge rule: beta_new = beta - scaling * (lora_B @ lora_A.T) / alpha_j
  forward:
    W = dequantize(weight_q, scale, zp)  -- true-quant base, dequant with learnable zp
    result = conv2d(x, W)
    result += (QA(x_unfold) * (D_in//L)) @ lora_A.T @ lora_B.T * scaling
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

import sys
import os

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.backbone.efficientnet_b0 import get_efficientnet_b0
from models.peft.int8_utils import get_mbconv_q_path_names


class QALoRAConv2d(nn.Module):
    """QA-LoRA for a single Conv2d layer — group-wise true integer quant + grouped LoRA."""

    def __init__(
        self,
        conv: nn.Conv2d,
        rank: int = 8,
        num_groups: int = 4,
        alpha: int = 16,
        dropout: float = 0.0,
    ) -> None:
        super().__init__()

        C_out, C_in, k_h, k_w = conv.weight.shape
        self.D_in = C_in * k_h * k_w
        self.num_groups = num_groups
        self.group_size = self.D_in // num_groups
        self.rank = rank
        self.scaling = alpha / rank

        self.stride = conv.stride
        self.padding = conv.padding if isinstance(conv.padding, tuple) else (conv.padding, conv.padding)
        self.dilation = conv.dilation
        self.groups = conv.groups
        self.out_channels = C_out
        self.in_channels = C_in
        self.kernel_size = (k_h, k_w)

        with torch.no_grad():
            W_flat = conv.weight.detach().view(C_out, -1)
            W_grouped = W_flat.view(C_out, num_groups, self.group_size)
            per_group_max = W_grouped.amax(dim=-1, keepdim=True)
            per_group_min = W_grouped.amin(dim=-1, keepdim=True)
            range_per_group = (per_group_max - per_group_min).clamp(min=1e-8)
            init_scale = (range_per_group / 15.0).squeeze(-1)
            init_zp = -8.0 - per_group_min.squeeze(-1) / init_scale

            scale_view = init_scale.view(-1, num_groups, 1)
            zp_view = init_zp.view(-1, num_groups, 1)
            W_quant = torch.round(W_grouped / scale_view + zp_view).clamp(-8, 7).to(torch.int8)

        self.register_buffer("weight_q", W_quant)
        if conv.bias is not None:
            self.register_buffer("bias", conv.bias.detach().clone().contiguous())
        else:
            self.register_buffer("bias", None)

        self.scale = nn.Parameter(init_scale)
        self.zero_point = nn.Parameter(init_zp)

        self.lora_A = nn.Parameter(torch.zeros(num_groups, rank))
        self.lora_B = nn.Parameter(torch.zeros(C_out, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=5.0 ** 0.5)
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape
        k_h, k_w = self.kernel_size

        W_q = self.weight_q.float()
        scale_view = self.scale.view(-1, self.num_groups, 1)
        zp_view = self.zero_point.view(-1, self.num_groups, 1)
        W_dequant = (W_q - zp_view) * scale_view
        W_dequant = W_dequant.view(self.out_channels, self.in_channels, k_h, k_w)

        out = F.conv2d(
            x, W_dequant, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        x_unfold = F.unfold(
            x,
            kernel_size=(k_h, k_w),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        x_grouped = F.adaptive_avg_pool1d(x_unfold.transpose(1, 2), self.num_groups)
        x_grouped = x_grouped.transpose(1, 2) * self.group_size
        x_grouped = self.dropout(x_grouped)

        lora_mid = torch.matmul(x_grouped.transpose(1, 2), self.lora_A)
        lora_out = torch.matmul(lora_mid, self.lora_B.t()) * self.scaling
        lora_out = lora_out.transpose(1, 2)

        l_out_h = (H + 2 * self.padding[0] - k_h) // self.stride[0] + 1
        l_out_w = (W + 2 * self.padding[1] - k_w) // self.stride[1] + 1
        lora_spatial = lora_out.view(B, self.out_channels, l_out_h, l_out_w)

        return out + lora_spatial

    def merge_with_quantization(self) -> None:
        """Merge LoRA adapters into quantized weights via zero-point update.

        Implements the paper's Algorithm 1 merge rule:
            beta_new = beta - scaling * (lora_B @ lora_A.T) / alpha_j

        After calling this, the quantized weights + updated zero-points represent
        the fine-tuned model. lora_A and lora_B are zeroed to avoid double-counting
        in subsequent forward passes.
        """
        with torch.no_grad():
            adapter_update = self.lora_B @ self.lora_A.t()
            zp_update = self.scaling * adapter_update / self.scale
            self.zero_point.data -= zp_update
            self.lora_A.zero_()
            self.lora_B.zero_()


def merge_qalora_weights(model: nn.Module) -> None:
    """Merge QA-LoRA adapters into quantized weights for all QALoRAConv2d layers."""
    qalora_layers = [m for m in model.modules() if isinstance(m, QALoRAConv2d)]
    if not qalora_layers:
        print("QA-LoRA: no QALoRAConv2d layers found to merge.")
        return
    for layer in qalora_layers:
        layer.merge_with_quantization()
    print(f"QA-LoRA: merged {len(qalora_layers)} layer(s).")


def save_merged_qalora_model(model: nn.Module, save_path: str) -> None:
    """Save a merged QA-LoRA model state dict for INT4 inference."""
    torch.save(model.state_dict(), save_path)
    print(f"QA-LoRA: merged state dict saved to {save_path}")


def load_merged_qalora_for_inference(num_classes: int, state_dict_path: str, device: str = "cpu") -> nn.Module:
    """Load a merged QA-LoRA checkpoint for INT4 inference."""
    model = get_qalora_model(num_classes=num_classes)
    state = torch.load(state_dict_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    print(f"QA-LoRA: merged checkpoint loaded from {state_dict_path}")
    return model


def get_qalora_model(
    num_classes: int,
    rank: int = 8,
    num_groups: int = 4,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: list[str] | None = None,
) -> nn.Module:
    model = get_efficientnet_b0(num_classes=num_classes, pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    q_path_names = get_mbconv_q_path_names(model)

    if target_modules is None:
        target_modules = list(q_path_names)

    target_set = set(target_modules)
    replacements: dict[str, nn.Module] = {}

    for name, module in model.named_modules():
        if name not in target_set or not isinstance(module, nn.Conv2d):
            continue
        replacements[name] = QALoRAConv2d(
            module, rank=rank, num_groups=num_groups, alpha=alpha, dropout=dropout
        )

    for name, new_module in replacements.items():
        parent_name = name.rsplit(".", 1)[0] if "." in name else ""
        parent = model.get_submodule(parent_name) if parent_name else model
        attr_name = name.rsplit(".", 1)[-1]
        setattr(parent, attr_name, new_module)

    if "classifier.fc" not in target_set:
        for name, module in model.named_modules():
            if name == "classifier.fc":
                for param in module.parameters():
                    param.requires_grad = True
                break

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError("QA-LoRA: no trainable parameters after setup.")

    print(
        f"QA-LoRA: layers={len(replacements)}, rank={rank}, groups={num_groups}, "
        f"alpha={alpha} -> trainable={trainable:,}"
    )
    model.print_trainable_parameters = lambda: print(
        f"  Trainable parameters: {trainable:,} / "
        f"{sum(p.numel() for p in model.parameters()):,}"
    )
    model.print_trainable_parameters()

    return model
