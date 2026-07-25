"""LoRA V3 — CNN-selective insertion matching LoRAE/LoRA-C/CoLoRA.

Difference from V1 LoRA:
  V1: targets ALL non-depthwise Conv2d layers (stem + pointwise + head)
  V3: targets MBConv expand/project 1×1 convs + stem (features.0.0) + head
       (features.8.0) + classifier.fc
"""

import os
import sys

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model, PeftModel

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
        target_modules.append("features.8.0")
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

    print(f"LoRA V3: targets={len(target_modules)}, trainable={trainable:,}")
    peft_model.print_trainable_parameters()

    return peft_model


def merge_lora_weights(model: nn.Module) -> None:
    """Merge LoRA adapters into base weights in-place.
    
    Computes W_merged = W0 + BA for every LoRA layer, then replaces
    the base weight with the merged result. After calling this, the model
    runs with zero adapter overhead — identical to the original paper's
    inference claim.
    """
    if not isinstance(model, PeftModel):
        raise TypeError("merge_lora_weights requires a PeftModel. Use get_lora_v3_model() output.")

    model.merge_and_unload()
    print("LoRA V3: adapters merged into base weights.")


def save_merged_model(model: nn.Module, save_path: str) -> None:
    """Save a merged model state dict for zero-overhead inference.
    
    Args:
        model: PeftModel after calling merge_lora_weights(), or any nn.Module.
        save_path: Path to save the state dict (e.g. 'lora_v3_merged.pth').
    """
    torch.save(model.state_dict(), save_path)
    print(f"LoRA V3: merged state dict saved to {save_path}")


def load_merged_for_inference(num_classes: int, state_dict_path: str, device: str = "cpu") -> nn.Module:
    """Load a merged checkpoint for pure base-model inference.
    
    Args:
        num_classes: Number of output classes (for backbone init).
        state_dict_path: Path to state dict saved by save_merged_model().
        device: 'cpu', 'cuda', etc.
    
    Returns:
        EfficientNet-B0 with merged LoRA weights, no PEFT wrapper.
    """
    model = get_efficientnet_b0(num_classes=num_classes, pretrained=True)
    state = torch.load(state_dict_path, map_location=device, weights_only=True)
    model.load_state_dict(state)
    model.eval()
    model.to(device)
    print(f"LoRA V3: merged checkpoint loaded from {state_dict_path}")
    return model
