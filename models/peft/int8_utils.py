"""Weight-only INT8 helpers for CNN-QLoRA and CNN-QKLoRA."""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def is_pointwise_conv(module: nn.Module) -> bool:
    return (
        isinstance(module, nn.Conv2d)
        and module.kernel_size == (1, 1)
        and module.groups == 1
    )


def is_depthwise_conv(module: nn.Module) -> bool:
    return isinstance(module, nn.Conv2d) and module.groups > 1


def is_se_conv_name(name: str) -> bool:
    """SE squeeze/excite layers in torchvision EfficientNet (Conv2d 1x1 named fc1/fc2)."""
    return ".fc1" in name or ".fc2" in name


def is_mbconv_expand_or_project(name: str, module: nn.Module) -> bool:
    """MBConv channel expand/project 1x1 convs (Q-path), excluding SE and stem."""
    if not is_pointwise_conv(module):
        return False
    if is_se_conv_name(name):
        return False
    if name == "features.0.0":
        return False
    return ".block.0.0" in name or ".block.2.0" in name or ".block.3.0" in name or name == "features.8.0"


def get_mbconv_q_path_names(model: nn.Module) -> list[str]:
    """
    Q-path: MBConv expand + project 1x1 convolutions and final head conv (features.8.0).
    Excludes SE (fc1/fc2) and stem (features.0.0).
    """
    return [
        name
        for name, module in model.named_modules()
        if is_mbconv_expand_or_project(name, module)
    ]


def get_mbconv_k_path_se_names(model: nn.Module) -> list[str]:
    """
    K-path SE layers: Squeeze-and-Excitation 1x1 convs (torchvision uses Conv2d, not Linear).
    Depthwise convolutions are FP32/frozen with no LoRA (PEFT limitation).
    """
    return [
        name
        for name, module in model.named_modules()
        if is_pointwise_conv(module) and is_se_conv_name(name)
    ]


def get_pointwise_conv_names(model: nn.Module) -> list[str]:
    """All 1x1 pointwise convs including SE (legacy helper). Prefer get_mbconv_q_path_names."""
    return [name for name, module in model.named_modules() if is_pointwise_conv(module)]


def get_se_linear_names(model: nn.Module) -> list[str]:
    """Linear SE layers (unused in torchvision EfficientNet; kept for compatibility)."""
    names = []
    for name, module in model.named_modules():
        if not isinstance(module, nn.Linear):
            continue
        if not name.startswith("features."):
            continue
        if ".block." not in name:
            continue
        if name.endswith(".fc1") or name.endswith(".fc2"):
            names.append(name)
    return names


def _normalize_module_name(name: str) -> str:
    for prefix in ("base_model.model.", "model."):
        if name.startswith(prefix):
            return name[len(prefix):]
    return name


def _matches_target(name: str, target_names: set[str]) -> bool:
    normalized = _normalize_module_name(name)
    if normalized in target_names or name in target_names:
        return True
    # PEFT LoRA wrappers use the parent module name without ".base_layer"
    if name.endswith(".base_layer"):
        parent = _normalize_module_name(name[: -len(".base_layer")])
        return parent in target_names
    return False


def apply_int8_weight_only_conv2d(conv: nn.Conv2d) -> None:
    """Store Conv2d weights as INT8 and dequantize at forward time (activations stay FP32)."""
    if hasattr(conv, "weight_int8"):
        return

    with torch.no_grad():
        weight = conv.weight.detach().float()
        per_channel_scale = weight.abs().amax(dim=(1, 2, 3), keepdim=True).clamp(min=1e-8) / 127.0
        weight_int8 = torch.round(weight / per_channel_scale).clamp(-128, 127).to(torch.int8)

    conv.register_buffer("weight_int8", weight_int8)
    conv.register_buffer("weight_scale", per_channel_scale.squeeze(-1).squeeze(-1).squeeze(-1))

    # Drop the FP32 weight parameter so we don't store weights twice in memory/checkpoints.
    if "weight" in conv._parameters:
        del conv._parameters["weight"]

    bias = conv.bias
    stride = conv.stride
    padding = conv.padding
    dilation = conv.dilation
    groups = conv.groups

    def int8_forward(x: torch.Tensor) -> torch.Tensor:
        scale = conv.weight_scale.view(-1, 1, 1, 1)
        dequant_weight = conv.weight_int8.float() * scale
        return F.conv2d(x, dequant_weight, bias, stride, padding, dilation, groups)

    conv.forward = int8_forward  # type: ignore[method-assign]


def quantize_lora_base_layers(model: nn.Module, target_names: set[str]) -> int:
    """Apply INT8 weight-only quantization to frozen PEFT LoRA base Conv2d layers."""
    quantized_ids: set[int] = set()
    target_set = set(target_names)

    for name, module in model.named_modules():
        if not hasattr(module, "base_layer"):
            continue
        if not _matches_target(name, target_set):
            continue

        base_layer = module.get_base_layer() if hasattr(module, "get_base_layer") else module.base_layer
        if not isinstance(base_layer, nn.Conv2d):
            continue
        if id(base_layer) in quantized_ids:
            continue

        apply_int8_weight_only_conv2d(base_layer)
        quantized_ids.add(id(base_layer))

    return len(quantized_ids)


def count_int8_conv_layers(model: nn.Module) -> int:
    return sum(
        1
        for _, module in model.named_modules()
        if isinstance(module, nn.Conv2d) and hasattr(module, "weight_int8")
    )
