"""CNN-QKLoRA (§9): selective INT8 Q-path + FP32 K-path with tiered LoRA ranks."""

import os
import sys

import torch.nn as nn
from peft import LoraConfig, get_peft_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.backbone.efficientnet_b0 import get_efficientnet_b0
from models.peft.int8_utils import (
    count_int8_conv_layers,
    get_mbconv_k_path_se_names,
    get_mbconv_q_path_names,
    quantize_lora_base_layers,
)


def get_qklora_model(
    num_classes: int,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    q_rank: int = 16,
    k_rank: int = 4,
    q_target_modules: list | None = None,
    k_target_modules: list | None = None,
) -> nn.Module:
    """
    INT8 CNN-QKLoRA for EfficientNet-B0.

    Q path (Quantized):
      - MBConv expand/project 1x1 convolutions + head conv (features.8.0)
      - Weight-only INT8 frozen backbone weights
      - Higher-rank LoRA adapters (default r=16)

    K path (Kept high precision):
      - Squeeze-and-Excitation 1x1 convs (fc1/fc2) + classifier head
      - FP32 frozen backbone weights
      - Lower-rank LoRA adapters (default r=4)

    Depthwise convolutions remain FP32 and frozen with no adapters (PEFT limitation).
    """
    model = get_efficientnet_b0(num_classes=num_classes, pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    if q_target_modules is None:
        q_target_modules = get_mbconv_q_path_names(model)
    if k_target_modules is None:
        k_target_modules = get_mbconv_k_path_se_names(model)
        k_target_modules.append("classifier.fc")

    all_target_modules = list(dict.fromkeys(q_target_modules + k_target_modules))

    rank_pattern = {}
    for name in q_target_modules:
        rank_pattern[name] = q_rank
    for name in k_target_modules:
        rank_pattern[name] = k_rank

    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=all_target_modules,
        rank_pattern=rank_pattern,
        lora_dropout=dropout,
        bias="none",
        task_type=None,
    )

    peft_model = get_peft_model(model, lora_config)

    quantize_targets = set(q_target_modules)
    num_quantized = quantize_lora_base_layers(peft_model, quantize_targets)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError("CNN-QKLoRA: no trainable LoRA parameters after setup.")

    print(
        f"CNN-QKLoRA: Q-path INT8 layers={num_quantized} (rank={q_rank}), "
        f"K-path FP32 LoRA layers={len(k_target_modules)} (rank={k_rank})"
    )
    print(f"CNN-QKLoRA: verified INT8 conv layers={count_int8_conv_layers(peft_model)}")
    peft_model.print_trainable_parameters()

    return peft_model
