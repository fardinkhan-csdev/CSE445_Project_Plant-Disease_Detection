"""Fake-quantization utilities with learnable scale / zero-point for QA-LoRA."""

import torch


def fake_quantize_per_channel(x: torch.Tensor, scale: torch.Tensor, zero_point: torch.Tensor) -> torch.Tensor:
    scale_view = scale.view(-1, *([1] * (x.ndim - 1)))
    zp_view = zero_point.view(-1, *([1] * (x.ndim - 1)))
    x_q = torch.round(x / scale_view + zp_view).clamp(-128, 127)
    return (x_q - zp_view) * scale_view
