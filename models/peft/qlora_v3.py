"""CNN-QLoRA V3 — real 4-bit NF4 via bitsandbytes on 1×1 conv layers.

Difference from V1 QLoRA:
  V1: custom INT8 per-channel quantization (weight-only, CNN workaround)
  V3: uses bitsandbytes 4-bit NF4 quantization on Q-path pointwise convs,
      matching Dettmers et al. (2023) more closely.
  V3 compute dtype: bfloat16, matching the paper's BF16 computation requirement.

Strategy:
  1. Build backbone, locate Q-path pointwise convs
  2. Quantize their weights to NF4 and store as buffers
  3. Freeze all backbone weights
  4. Apply PEFT LoRA on top of quantized base layers
"""

import os
import sys

import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
from models.backbone.efficientnet_b0 import get_efficientnet_b0
from models.peft.int8_utils import get_mbconv_q_path_names

try:
    import bitsandbytes.functional as bnb_f
    BNB_AVAILABLE = True
except ImportError:
    BNB_AVAILABLE = False

COMPUTE_DTYPE = torch.bfloat16


def _quantize_conv_to_nf4(conv: nn.Conv2d) -> None:
    if conv.kernel_size[0] != 1 or conv.kernel_size[1] != 1:
        return
    if not BNB_AVAILABLE:
        return

    with torch.no_grad():
        w = conv.weight.detach().reshape(conv.out_channels, -1)
        q_weight, q_state = bnb_f.quantize_4bit(
            w,
            quant_type='nf4',
            compress_statistics=False,
        )

    q_weight = q_weight.contiguous()
    conv.register_buffer('q_weight', q_weight)
    conv.q_state = q_state
    if conv.bias is not None:
        conv.register_buffer('_orig_bias', conv.bias.clone())
    else:
        conv.register_buffer('_orig_bias', None)

    if "weight" in conv._parameters:
        del conv._parameters["weight"]

    _patch_conv_forward(conv)


def _dequantize_conv_weight(conv: nn.Conv2d) -> torch.Tensor:
    q_weight = conv.q_weight
    q_state = conv.q_state
    target_device = q_weight.device
    if getattr(q_state, 'absmax', None) is not None and q_state.absmax.device != target_device:
        q_state.absmax = q_state.absmax.to(target_device)
    w_deq = bnb_f.dequantize_4bit(q_weight, q_state)
    w_deq = w_deq.reshape(conv.out_channels, conv.in_channels, 1, 1)
    return w_deq.to(COMPUTE_DTYPE)


def _patch_conv_forward(conv: nn.Conv2d) -> None:
    if getattr(conv, '_nf4_patched', False):
        return
    import torch.nn.functional as F

    def nf4_forward(x: torch.Tensor) -> torch.Tensor:
        w_deq = _dequantize_conv_weight(conv)
        if w_deq.device != x.device:
            w_deq = w_deq.to(x.device)
        return F.conv2d(x, w_deq, conv.bias, conv.stride, conv.padding, conv.dilation, conv.groups)

    conv.forward = nf4_forward
    conv._nf4_patched = True


def get_qlora_v3_model(
    num_classes: int,
    rank: int = 8,
    alpha: int = 16,
    dropout: float = 0.1,
    target_modules: list | None = None,
) -> nn.Module:
    if not BNB_AVAILABLE:
        raise RuntimeError("bitsandbytes is required for QLoRA V3 (4-bit NF4).")

    model = get_efficientnet_b0(num_classes=num_classes, pretrained=True)

    for param in model.parameters():
        param.requires_grad = False

    q_path_modules = set(get_mbconv_q_path_names(model))  # Must capture BEFORE PEFT wrapping

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

    num_quantized = 0

    for name, module in peft_model.named_modules():
        if not hasattr(module, "base_layer"):
            continue
        base_layer = module.get_base_layer() if hasattr(module, "get_base_layer") else module.base_layer
        if not isinstance(base_layer, nn.Conv2d):
            continue
        if name not in q_path_modules and _normalize_name(name) not in q_path_modules:
            continue
        if base_layer.kernel_size[0] != 1 or base_layer.kernel_size[1] != 1:
            continue
        _quantize_conv_to_nf4(base_layer)
        _patch_conv_forward(base_layer)
        num_quantized += 1

    trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
    if trainable == 0:
        raise RuntimeError("QLoRA V3: no trainable LoRA parameters after NF4 quantization.")

    nf4_layers = sum(1 for n, m in peft_model.named_modules() if getattr(m, '_nf4_patched', False))
    print(f"QLoRA V3: NF4 Q-path layers={num_quantized}, LoRA targets={len(target_modules)}")
    peft_model.print_trainable_parameters()

    return peft_model


def _normalize_name(name: str) -> str:
    for prefix in ("base_model.model.", "model."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name
