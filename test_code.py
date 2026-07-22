import os
import sys
import torch

# Add project root to path
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

def test_data_loader():
    print("Testing data_loader module...")
    try:
        from data.data_loader import (
            get_data_transforms,
            PlantVillageDataset,
            get_cached_color_image_root,
            get_cached_hf_metadata_paths,
            build_image_samples,
            get_data_loaders,
        )

        print("Verifying PlantVillage dataset availability...")
        image_root = get_cached_color_image_root("data/raw")
        get_cached_hf_metadata_paths("mohanty/PlantVillage")
        samples = build_image_samples(image_root)
        transforms_dict = get_data_transforms()
        dataset = PlantVillageDataset(samples, transform=transforms_dict['val'])

        image, label, crop, disease, image_path = dataset[0]
        assert tuple(image.shape) == (3, 224, 224), f"Unexpected image shape: {tuple(image.shape)}"
        assert isinstance(label, int), f"Label should be an int, got {type(label)}"
        assert isinstance(crop, str) and isinstance(disease, str), "Crop and disease metadata should be strings"
        print("Verifying Hugging Face color split loader...")
        train_loader, val_loader, test_loader, class_info = get_data_loaders()
        assert class_info['num_classes'] == 38, f"Expected 38 classes, got {class_info['num_classes']}"
        assert len(train_loader.dataset) > 0, "Train split should not be empty"
        assert len(val_loader.dataset) > 0, "Validation split should not be empty"
        assert len(test_loader.dataset) > 0, "Test split should not be empty"
        print("[PASS] data_loader.py imported successfully and image dataset is available!")
        return True
    except Exception as e:
        print(f"[FAIL] data_loader.py failed: {e}")
        return False

def test_backbone():
    print("\nTesting EfficientNet-B0 backbone...")
    try:
        from models.backbone.efficientnet_b0 import get_efficientnet_b0
        model = get_efficientnet_b0(num_classes=38)
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (2, 38), f"Expected output shape (2, 38), got {output.shape}"
        print(f"[PASS] EfficientNet-B0 works! Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"[FAIL] EfficientNet-B0 failed: {e}")
        return False

def test_lora():
    print("\nTesting LoRA model...")
    try:
        from models.peft.lora import get_lora_model
        model = get_lora_model(num_classes=38, rank=8, alpha=16)
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (2, 38), f"Expected output shape (2, 38), got {output.shape}"
        print(f"[PASS] LoRA model works! Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"[FAIL] LoRA model failed: {e}")
        return False

def test_qlora():
    print("\nTesting QLoRA model...")
    try:
        from models.peft.qlora import get_qlora_model
        model = get_qlora_model(num_classes=38, rank=8, alpha=16)
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (2, 38), f"Expected output shape (2, 38), got {output.shape}"
        print(f"[PASS] QLoRA model works! Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"[FAIL] QLoRA model failed: {e}")
        return False

def test_qklora():
    print("\nTesting Q/K LoRA model...")
    try:
        from models.peft.qklora import get_qklora_model
        model = get_qklora_model(num_classes=38, rank=8, alpha=16)
        dummy_input = torch.randn(2, 3, 224, 224)
        output = model(dummy_input)
        assert output.shape == (2, 38), f"Expected output shape (2, 38), got {output.shape}"
        print(f"[PASS] Q/K LoRA model works! Output shape: {output.shape}")
        return True
    except Exception as e:
        print(f"[FAIL] Q/K LoRA model failed: {e}")
        return False

def test_metrics():
    print("\nTesting metrics module...")
    try:
        import numpy as np
        from evaluation.metrics import calculate_metrics
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 2, 0, 1, 1])
        idx_to_class = {0: "A", 1: "B", 2: "C"}
        metrics = calculate_metrics(y_true, y_pred, idx_to_class)
        print(f"[PASS] Metrics module works! Accuracy: {metrics['accuracy']:.4f}")
        return True
    except Exception as e:
        print(f"[FAIL] Metrics module failed: {e}")
        return False

def test_utils():
    print("\nTesting utils modules...")
    try:
        from utils.logger import setup_logger
        from utils.memory_tracker import get_gpu_memory_stats
        from utils.visualization import plot_training_curves
        print("[PASS] Utils modules imported successfully!")
        return True
    except Exception as e:
        print(f"[FAIL] Utils modules failed: {e}")
        return False

if __name__ == "__main__":
    print("="*50)
    print("Testing Leaf Disease Classification Codebase")
    print("="*50)

    results = {
        "data_loader": test_data_loader(),
        "backbone": test_backbone(),
        "lora": test_lora(),
        "qlora": test_qlora(),
        "qklora": test_qklora(),
        "metrics": test_metrics(),
        "utils": test_utils()
    }

    print("\n" + "="*50)
    print("Test Summary:")
    print("="*50)
    for name, passed in results.items():
        status = "[PASS]" if passed else "[FAIL]"
        print(f"{name:15s}: {status}")
    print(f"\nTotal: {sum(results.values())}/{len(results)} tests passed!")
