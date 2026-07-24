"""LoRA V3 — CNN-selective insertion matching LoRAE/LoRA-C/CoLoRA.

Difference from V1 LoRA:
  V1: targets ALL non-depthwise Conv2d layers (stem + pointwise + head)
  V3: targets ONLY MBConv expand/project 1×1 convs + classifier.fc
       Excludes stem and head convs to better match LoRAE's "only pointwise" principle.
"""

import os
import sys

import torch.nn as nn
from peft import LoraConfig, get_peft_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.backbone.efficientnet_b0 import get_efficientnet_b0
from models.peft.int8_utils import get_mbconv_q_path_names


def get_lora_v3_model(
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

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError("LoRA V3: no trainable parameters after setup.")

    print(f"LoRA V3: pointwise-only targets={len(target_modules)}, trainable={trainable:,}")
    peft_model.print_trainable_parameters()

    return peft_model
