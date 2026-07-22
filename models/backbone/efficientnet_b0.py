import torch
import torch.nn as nn
from pathlib import Path
from torchvision import models
from models.classifier import PlantDiseaseClassifier


def find_cached_efficientnet_b0_weights() -> str:
    checkpoint_dir = Path(torch.hub.get_dir()) / "checkpoints"
    candidates = sorted(checkpoint_dir.glob("efficientnet_b0*.pth"))
    if not candidates:
        raise FileNotFoundError(
            "Cached EfficientNet-B0 weights were not found. "
            "Run 'py -3.11 download_assets.py' once before training."
        )
    return str(candidates[0])


def get_efficientnet_b0(num_classes: int, pretrained: bool = True) -> nn.Module:
    model = models.efficientnet_b0(weights=None)
    if pretrained:
        state_dict_path = find_cached_efficientnet_b0_weights()
        state_dict = torch.load(state_dict_path, map_location="cpu")
        model.load_state_dict(state_dict)
    
    # Replace the final classifier layer with the dedicated head module
    in_features = model.classifier[1].in_features
    model.classifier = PlantDiseaseClassifier(in_features=in_features, num_classes=num_classes, dropout=0.2)
    
    return model


def freeze_backbone(model: nn.Module):
    # Freeze all layers except the classifier
    for param in model.features.parameters():
        param.requires_grad = False
    for param in model.classifier.parameters():
        param.requires_grad = True


def unfreeze_backbone(model: nn.Module, num_layers: int = 0):
    """Unfreeze the last num_layers blocks of model.features.

    Iterates over the child *modules* (blocks) of model.features rather than the
    flat list of tensors so that each "layer" corresponds to a complete MBConv
    block rather than an arbitrary single-tensor slice.
    """
    if num_layers == 0:
        return

    # model.features is an nn.Sequential of blocks; take the last num_layers blocks.
    feature_blocks = list(model.features.children())
    for block in feature_blocks[-num_layers:]:
        for param in block.parameters():
            param.requires_grad = True


if __name__ == '__main__':
    # Test model
    model = get_efficientnet_b0(num_classes=38)
    print(f"Model created with {sum(p.numel() for p in model.parameters())} total parameters")
    print(f"Trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    freeze_backbone(model)
    print(f"After freezing backbone, trainable parameters: {sum(p.numel() for p in model.parameters() if p.requires_grad)}")
    
    # Test forward pass
    dummy_input = torch.randn(2, 3, 224, 224)
    output = model(dummy_input)
    print(f"Output shape: {output.shape}")
