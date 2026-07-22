"""CNN-QLoRA (§8): weight-only INT8 Q-path backbone + trainable LoRA adapters."""

import os
import sys

import torch.nn as nn
from peft import LoraConfig, get_peft_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.backbone.efficientnet_b0 import get_efficientnet_b0
from models.peft.int8_utils import (
    count_int8_conv_layers,
    get_mbconv_q_path_names,
    quantize_lora_base_layers,
)


def get_qlora_model(
    num_classes: int,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: list | None = None,
) -> nn.Module:
    """
    INT8 CNN-QLoRA for EfficientNet-B0.

    1. Freeze full-precision backbone
    2. Attach LoRA to MBConv expand/project 1x1 convs + classifier head
    3. Quantize frozen base weights of Q-path conv layers to INT8 (weight-only)
    4. Train only LoRA parameters (activations remain FP32)

    SE layers and depthwise convs stay FP32/frozen (no LoRA on depthwise).
    """
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

    quantize_targets = set(q_path_modules)
    num_quantized = quantize_lora_base_layers(peft_model, quantize_targets)

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError("CNN-QLoRA: no trainable LoRA parameters after INT8 quantization.")

    print(f"CNN-QLoRA: INT8 Q-path layers={num_quantized}, LoRA targets={len(target_modules)}")
    print(f"CNN-QLoRA: verified INT8 conv layers={count_int8_conv_layers(peft_model)}")
    peft_model.print_trainable_parameters()

    return peft_model
