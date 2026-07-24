"""QA-LoRA for EfficientNet-B0.

Reference: Xu et al., 2024 — QA-LoRA: Quantization-Aware LoRA for LLMs.
Core principle: learnable fake-quantization on backbone Conv2d base layers
combined with standard LoRA adapters.  At inference, merged backbone + adapter
weights can be folded into fully quantized INT8 inference.

Difference from QLoRA:
  QLoRA  — fixed post-training INT8 backbone + full-precision LoRA adapters
  QA-LoRA — learned per-channel scale / zero-point + full-precision LoRA adapters
             (quantization is part of the training graph).
"""

import os
import sys
import torch
import torch.nn.functional as F
import torch.nn as nn
from peft import LoraConfig, get_peft_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.backbone.efficientnet_b0 import get_efficientnet_b0
from models.peft.int8_utils import get_mbconv_q_path_names
from models.peft.fake_quant import fake_quantize_per_channel


def _apply_learnable_fake_quant(conv: nn.Conv2d) -> None:
    if hasattr(conv, "_qa_scale"):
        return

    with torch.no_grad():
        weight = conv.weight.detach().float()
        max_val = weight.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8) / 127.0
        init_scale = max_val.squeeze(-1).squeeze(-1).squeeze(-1)

    scale_param = nn.Parameter(init_scale.clone())
    zp_param = nn.Parameter(torch.zeros(conv.weight.shape[0]))

    conv.register_parameter("_qa_scale", scale_param)
    conv.register_parameter("_qa_zp", zp_param)

    def qa_forward(x: torch.Tensor) -> torch.Tensor:
        scale_view = conv._qa_scale.view(-1, 1, 1, 1)
        zp_view = conv._qa_zp.view(-1, 1, 1, 1)
        dequant_weight = fake_quantize_per_channel(conv.weight, scale_view, zp_view)
        return F.conv2d(
            x,
            dequant_weight,
            conv.bias,
            conv.stride,
            conv.padding,
            conv.dilation,
            conv.groups,
        )

    conv.forward = qa_forward


def get_qalora_model(
    num_classes: int,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: list | None = None,
) -> nn.Module:
    model = get_efficientnet_b0(num_classes=num_classes, pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    q_path_modules = get_mbconv_q_path_names(model)

    if target_modules is None:
        target_modules = list(q_path_modules)
        target_modules.append("classifier.fc")

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=None,
    )

    peft_model = get_peft_model(model, lora_config)

    target_names = set(q_path_modules)
    num_qa = 0

    for name, module in peft_model.named_modules():
        if not hasattr(module, "base_layer"):
            continue

        normalized = name
        for prefix in ("base_model.model.", "model."):
            if normalized.startswith(prefix):
                normalized = normalized[len(prefix):]

        base_layer = (
            module.get_base_layer() if hasattr(module, "get_base_layer") else module.base_layer
        )

        if not isinstance(base_layer, nn.Conv2d):
            continue
        if normalized not in target_names:
            continue

        _apply_learnable_fake_quant(base_layer)
        num_qa += 1

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError("QA-LoRA: no trainable parameters after setup.")

    print(f"QA-LoRA: learnable fake-quant layers={num_qa}, LoRA targets={len(target_modules)}")
    peft_model.print_trainable_parameters()

    return peft_model
