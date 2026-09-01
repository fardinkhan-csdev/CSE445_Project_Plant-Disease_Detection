"""Smoke test for CNN-QLoRA and CNN-QKLoRA model construction and training step."""

import sys

import torch

sys.path.insert(0, ".")

from models.peft.qlora import get_qlora_model
from models.peft.qklora import get_qklora_model
from models.peft.int8_utils import count_int8_conv_layers, get_mbconv_k_path_se_names, get_mbconv_q_path_names
from models.backbone.efficientnet_b0 import get_efficientnet_b0


def _test_model(name: str, model) -> None:
    dummy = torch.randn(2, 3, 224, 224)
    labels = torch.randint(0, 38, (2,))
    model.train()
    out = model(dummy)
    loss = torch.nn.functional.cross_entropy(out, labels)
    loss.backward()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    grad_ok = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)
    int8_layers = count_int8_conv_layers(model)

    print(f"\n{name}")
    print(f"  output shape: {tuple(out.shape)}")
    print(f"  loss: {loss.item():.4f}")
    print(f"  trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    print(f"  params with grad: {grad_ok}")
    print(f"  int8 conv layers: {int8_layers}")

    assert out.shape == (2, 38)
    assert trainable > 0
    assert grad_ok > 0


def main() -> None:
    base = get_efficientnet_b0(38, pretrained=False)
    print(f"Q-path modules: {len(get_mbconv_q_path_names(base))}")
    print(f"K-path SE modules: {len(get_mbconv_k_path_se_names(base))}")

    _test_model("CNN-QLoRA", get_qlora_model(num_classes=38))
    _test_model("CNN-QKLoRA", get_qklora_model(num_classes=38))
    print("\nSmoke test passed.")


if __name__ == "__main__":
    main()
