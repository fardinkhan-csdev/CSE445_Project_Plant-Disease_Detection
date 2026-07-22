import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.backbone.efficientnet_b0 import get_efficientnet_b0


def get_lora_model(num_classes: int, rank: int = 8, alpha: int = 16,
                   dropout: float = 0.1, target_modules: list = None) -> nn.Module:
    # Get base model
    model = get_efficientnet_b0(num_classes=num_classes, pretrained=True)

    # Freeze base model
    for param in model.parameters():
        param.requires_grad = False

    # LoRA baseline: target all non-depthwise Conv2d layers (groups == 1) plus classifier.fc.
    # QLoRA/QKLoRA use narrower target_modules from their own config files.
    if target_modules is None:
        target_modules = []
        for name, module in model.named_modules():
            if isinstance(module, nn.Conv2d) and module.groups == 1:
                target_modules.append(name)
        # Also target the classifier's linear layer
        target_modules.append("classifier.fc")

    # Configure LoRA
    # task_type=None is required for non-transformer models so that peft does not
    # inject transformer-specific keyword arguments (e.g. input_ids) into forward().
    lora_config = LoraConfig(
        r=rank,
        lora_alpha=alpha,
        target_modules=target_modules,
        lora_dropout=dropout,
        bias="none",
        task_type=None
    )

    # Apply LoRA and return the peft model directly.
    # peft_model(x) correctly routes through the LoRA adapter layers.
    peft_model = get_peft_model(model, lora_config)
    peft_model.print_trainable_parameters()

    return peft_model
