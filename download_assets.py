import os
import sys

from torchvision import models
from huggingface_hub import snapshot_download


PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def main():
    print("=" * 70)
    print("Leaf Disease Classification - One-Click Downloader")
    print("=" * 70)

    try:
        from data.data_loader import download_plant_village_dataset, build_image_samples

        raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
        print("\n[1/3] Preparing PlantVillage color images...")
        image_root = download_plant_village_dataset(raw_dir)
        samples = build_image_samples(image_root)
        print(f"Dataset ready: {len(samples):,} images found in {image_root}")

        print("\n[2/3] Caching official Hugging Face split metadata...")
        snapshot_download(
            repo_id="mohanty/PlantVillage",
            repo_type="dataset",
            allow_patterns=[
                "splits/color_train.txt",
                "splits/color_test.txt",
                "leaf_grouping/leaf-map.json",
                "README.md",
            ]
        )
        print("Official Hugging Face split metadata is ready in the local cache.")

        print("\n[3/3] Downloading EfficientNet-B0 pretrained weights...")
        weights = models.EfficientNet_B0_Weights.DEFAULT
        models.efficientnet_b0(weights=weights)
        print("EfficientNet-B0 weights are ready in the local torch cache.")

        print("\nAll required assets are ready.")
        print("You can now run: py -3.11 launcher.py")
        return 0
    except KeyboardInterrupt:
        print("\nDownload interrupted by user.")
        return 1
    except Exception as exc:
        print(f"\nDownload failed: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
