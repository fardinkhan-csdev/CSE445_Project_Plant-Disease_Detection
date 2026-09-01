"""Smoke test for V3 model fixes:
  - QA-LoRA: H vs W output size + INT4 storage
  - QLoRA: BF16 compute dtype in dequant
  - LoRA: unchanged baseline
"""

import sys

import torch

sys.path.insert(0, ".")

from models.peft.qlora_v3 import get_qlora_v3_model, _dequantize_conv_weight
from models.peft.qalora import get_qalora_model, QALoRAConv2d
from models.peft.lora_v3 import get_lora_v3_model
from models.peft.int8_utils import get_mbconv_q_path_names
from models.backbone.efficientnet_b0 import get_efficientnet_b0


def _test_model(name: str, model, input_shape=(2, 3, 224, 224), num_classes=38) -> torch.Tensor:
    dummy = torch.randn(*input_shape)
    labels = torch.randint(0, num_classes, (input_shape[0],))
    model.train()
    out = model(dummy)
    loss = torch.nn.functional.cross_entropy(out, labels)
    loss.backward()

    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    grad_ok = sum(1 for p in model.parameters() if p.requires_grad and p.grad is not None)

    print(f"\n{name}")
    print(f"  input shape: {input_shape}")
    print(f"  output shape: {tuple(out.shape)}")
    print(f"  loss: {loss.item():.4f}")
    print(f"  trainable: {trainable:,} / {total:,} ({100 * trainable / total:.2f}%)")
    print(f"  params with grad: {grad_ok}")

    assert out.shape == (input_shape[0], num_classes), f"{name}: bad output shape {out.shape}"
    assert trainable > 0, f"{name}: no trainable params"
    assert grad_ok > 0, f"{name}: no params received grad"
    assert torch.isfinite(loss).item(), f"{name}: loss is non-finite"

    model.eval()
    with torch.no_grad():
        out_eval = model(dummy)
    assert torch.isfinite(out_eval).all().item(), f"{name}: eval output has non-finite values"
    return out


def test_qalora_int4_storage() -> None:
    """Verify QA-LoRA uses INT4 [-8,7] storage in `weight_q`."""
    print("\n--- QA-LoRA INT4 storage check ---")
    model = get_qalora_model(num_classes=38, rank=8, num_groups=4, alpha=16, dropout=0.0)
    qalora_layers = [m for m in model.modules() if isinstance(m, QALoRAConv2d)]
    assert len(qalora_layers) > 0, "No QALoRAConv2d layers found"
    sample = qalora_layers[0]
    wq = sample.weight_q
    print(f"  weight_q dtype: {wq.dtype}")
    print(f"  weight_q shape: {tuple(wq.shape)}")
    print(f"  weight_q range: [{wq.min().item()}, {wq.max().item()}]")
    print(f"  scale shape: {tuple(sample.scale.shape)}")
    print(f"  zero_point shape: {tuple(sample.zero_point.shape)}")
    print(f"  lora_A shape: {tuple(sample.lora_A.shape)}")
    print(f"  lora_B shape: {tuple(sample.lora_B.shape)}")
    assert wq.dtype == torch.int8, f"weight_q should be int8, got {wq.dtype}"
    assert wq.min().item() >= -8 and wq.max().item() <= 7, f"INT4 range violated: [{wq.min()}, {wq.max()}]"
    assert sample.scale.shape == (sample.out_channels, sample.num_groups)
    assert sample.zero_point.shape == (sample.out_channels, sample.num_groups)
    assert sample.lora_A.shape == (sample.num_groups, sample.rank)
    assert sample.lora_B.shape == (sample.out_channels, sample.rank)
    print("  INT4 storage: OK")


def test_qalora_nonsquare_input() -> None:
    """Verify the H/W fix by passing a non-square input."""
    print("\n--- QA-LoRA non-square input check ---")
    model = get_qalora_model(num_classes=38, rank=8, num_groups=4, alpha=16, dropout=0.0)
    model.eval()
    for h, w in [(224, 224), (160, 192), (96, 224)]:
        with torch.no_grad():
            x = torch.randn(1, 3, h, w)
            try:
                out = model(x)
                print(f"  input {h}x{w} -> output {tuple(out.shape)}: OK")
            except Exception as e:
                print(f"  input {h}x{w} -> FAILED: {e}")
                raise
    model.train()
    print("  Non-square inputs: OK")


def test_qlora_bf16_dequant() -> None:
    """Verify QLoRA dequant returns BF16 weights."""
    print("\n--- QLoRA BF16 dequant check ---")
    model = get_qlora_v3_model(num_classes=38, rank=8, alpha=16, dropout=0.0)
    nf4_layers = [m for m in model.modules() if hasattr(m, "q_weight") and hasattr(m, "_nf4_quant_type")]
    assert len(nf4_layers) > 0, "No NF4-quantized conv layers found"
    sample = nf4_layers[0]
    w_deq = _dequantize_conv_weight(sample)
    print(f"  dequantized weight dtype: {w_deq.dtype}")
    print(f"  dequantized weight shape: {tuple(w_deq.shape)}")
    print(f"  NF4 layers count: {len(nf4_layers)}")
    assert w_deq.dtype == torch.bfloat16, f"dequant should return BF16, got {w_deq.dtype}"
    print("  BF16 dequant: OK")


def test_forward_backward(name: str, builder) -> None:
    _test_model(name, builder())


def main() -> None:
    print("=" * 60)
    print("SMOKE TESTS: V3 PEFT quantization fixes")
    print("=" * 60)

    test_qalora_int4_storage()
    test_qalora_nonsquare_input()
    test_qlora_bf16_dequant()

    print("\n" + "=" * 60)
    print("FORWARD + BACKWARD on dummy batch")
    print("=" * 60)
    test_forward_backward("LoRA V3", lambda: get_lora_v3_model(num_classes=38, rank=8, alpha=16, dropout=0.0))
    test_forward_backward("QLoRA V3", lambda: get_qlora_v3_model(num_classes=38, rank=8, alpha=16, dropout=0.0))
    test_forward_backward("QA-LoRA V3", lambda: get_qalora_model(num_classes=38, rank=8, num_groups=4, alpha=16, dropout=0.0))

    print("\n" + "=" * 60)
    print("ALL SMOKE TESTS PASSED")
    print("=" * 60)


if __name__ == "__main__":
    main()
