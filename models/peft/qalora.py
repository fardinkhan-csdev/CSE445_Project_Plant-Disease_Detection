"""QA-LoRA for EfficientNet-B0 — faithful to Xu et al., ICLR 2024.

Real QA-LoRA (not fake):
  - Group-wise fake-quantization: each output channel's weights split into L groups,
    each group has its own learnable scale and zero-point (increases quantization DOF).
  - Grouped LoRA A: LoRA A shape reduced from (D_in, r) to (L, r) by averaging the
    input features into L groups via adaptive avg-pool (decreases adaptation DOF).
  - Balance goal: increase quantization params and decrease adaptation params, so after
    training the weights stay fully quantized without FP16 fallback.

Reference Algorithm 1 from paper (adapted for Conv2d):
  QA = nn.AdaptiveAvgPool1d(L)
  lora_A = (L, r)       -- grouped, smaller than standard (D_in, r)
  lora_B = (C_out, r)   -- standard
  forward:
    W_tilde = group_wise_fake_quant(W, scale, zp)
    result = conv2d(x, W_tilde)
    result += (QA(x_unfold) * group_size) @ lora_A.T @ lora_B.T * scaling
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
    """QA-LoRA for a single Conv2d layer — group-wise quant + grouped LoRA."""

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

        # Frozen original weights
        self.register_buffer("weight_orig", conv.weight.detach().clone().contiguous())
        if conv.bias is not None:
            self.register_buffer("bias", conv.bias.detach().clone().contiguous())
        else:
            self.register_buffer("bias", None)

        # --- Group-wise quantization params (C_out x L) ---
        with torch.no_grad():
            W_flat = self.weight_orig.view(C_out, -1)
            W_grouped = W_flat.view(C_out, num_groups, self.group_size)
            per_group_max = W_grouped.abs().amax(dim=-1, keepdim=True).clamp(min=1e-8)
            init_scale = (per_group_max / 127.0).squeeze(-1)

        self.scale = nn.Parameter(init_scale)                      # (C_out, L)
        self.zero_point = nn.Parameter(torch.zeros(C_out, num_groups))

        # --- Grouped LoRA A: (L, r) instead of standard (D_in, r) ---
        self.lora_A = nn.Parameter(torch.zeros(num_groups, rank))
        self.lora_B = nn.Parameter(torch.zeros(C_out, rank))

        nn.init.kaiming_uniform_(self.lora_A, a=5.0 ** 0.5)
        nn.init.zeros_(self.lora_B)

        self.dropout = nn.Dropout(p=dropout) if dropout > 0 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C_in, H, W = x.shape
        _, _, k_h, k_w = self.weight_orig.shape

        # ================================================================
        # 1. Group-wise fake-quantization of the frozen backbone weights
        # ================================================================
        W_flat = self.weight_orig.view(self.weight_orig.shape[0], self.D_in)
        W_grouped = W_flat.view(self.weight_orig.shape[0], self.num_groups, self.group_size)

        scale_view = self.scale.view(-1, self.num_groups, 1)
        zp_view = self.zero_point.view(-1, self.num_groups, 1)

        W_quant = W_grouped / scale_view + zp_view
        W_quant = W_quant.round().clamp(0, 255)
        W_quant = (W_quant - zp_view) * scale_view
        W_quant = W_quant.view_as(self.weight_orig)

        out = F.conv2d(
            x, W_quant, self.bias, self.stride, self.padding, self.dilation, self.groups
        )

        # ================================================================
        # 2. Grouped LoRA (paper Algorithm 1, adapted for Conv2d)
        # ================================================================
        x_unfold = F.unfold(
            x,
            kernel_size=(k_h, k_w),
            stride=self.stride,
            padding=self.padding,
            dilation=self.dilation,
        )

        # Adaptive avg-pool over feature dim to reduce D_in -> num_groups (the paper's "QA")
        # x_unfold: (B, D_in, N_positions); adaptive_avg_pool1d pools over LAST dim,
        # so transpose -> pool -> transpose back to pool over D_in.
        x_grouped = F.adaptive_avg_pool1d(x_unfold.transpose(1, 2), self.num_groups)
        x_grouped = x_grouped.transpose(1, 2) * self.group_size
        x_grouped = self.dropout(x_grouped)

        # lora_A: (L, r), lora_B: (C_out, r)
        # x_grouped: (B, L, N_positions)
        lora_mid = torch.matmul(x_grouped.transpose(1, 2), self.lora_A)   # (B, N, r)
        lora_out = torch.matmul(lora_mid, self.lora_B.t()) * self.scaling # (B, N, C_out)
        lora_out = lora_out.transpose(1, 2)                                # (B, C_out, N)

        l_out_h = (H + 2 * self.padding[0] - k_h) // self.stride[0] + 1
        l_out_w = (H + 2 * self.padding[1] - k_w) // self.stride[1] + 1
        lora_spatial = lora_out.view(B, self.weight_orig.shape[0], l_out_h, l_out_w)

        return out + lora_spatial


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
