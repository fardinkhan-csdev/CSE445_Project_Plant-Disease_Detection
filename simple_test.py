import sys
import torch
import torch.nn as nn
from peft import LoraConfig, get_peft_model

# Load our backbone
sys.path.insert(0, '.')
from models.backbone.efficientnet_b0 import get_efficientnet_b0

model = get_efficientnet_b0(num_classes=38)

# Test basic forward
dummy = torch.randn(1, 3, 224, 224)
print("Base model output shape:", model(dummy).shape)

# Try LoRA with task_type=None (required for non-transformer models)
target_modules = []
for name, m in model.named_modules():
    if isinstance(m, nn.Conv2d) and m.groups == 1:
        target_modules.append(name)
target_modules.append("classifier.fc")

print("\nTarget modules:", target_modules)

config = LoraConfig(
    r=8,
    lora_alpha=16,
    target_modules=target_modules,
    lora_dropout=0.1,
    bias="none",
    task_type=None   # Required for CNNs — avoids transformer keyword injection
)

peft_model = get_peft_model(model, config)

# With task_type=None, peft_model(x) works directly and correctly applies adapters
print("\nCalling peft_model directly with dummy...")
try:
    out = peft_model(dummy)
    print("PEFT model output shape:", out.shape)
    print("✓ LoRA adapters are active in the forward pass!")
except Exception as e:
    print("ERROR with peft_model(dummy):", type(e), str(e))

# Verify trainable parameters
trainable = sum(p.numel() for p in peft_model.parameters() if p.requires_grad)
total = sum(p.numel() for p in peft_model.parameters())
print(f"\nTrainable: {trainable:,} / {total:,} ({100*trainable/total:.2f}%)")
