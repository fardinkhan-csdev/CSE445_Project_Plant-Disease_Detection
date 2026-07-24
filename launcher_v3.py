import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')


def run_command(cmd_list, label=""):
    if label:
        print(f"\n{label}")
        print("=" * 70)
    try:
        subprocess.run(cmd_list, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\nCommand failed with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\nAction interrupted by user!")


def check_training_readiness():
    """Fail fast if training assets are missing instead of downloading them."""
    try:
        from data.data_loader import get_cached_color_image_root, get_cached_hf_metadata_paths
        from models.backbone.efficientnet_b0 import find_cached_efficientnet_b0_weights

        raw_dir = os.path.join(PROJECT_ROOT, "data", "raw")
        get_cached_color_image_root(raw_dir)
        get_cached_hf_metadata_paths("mohanty/PlantVillage")
        find_cached_efficientnet_b0_weights()
        return True
    except Exception as exc:
        print("\nTraining assets are not fully prepared.")
        print(f"   {exc}")
        print("   Run: py -3.11 download_assets.py")
        print("   Then start the launcher again.")
        return False


def main():
    while True:
        print("\n" + "=" * 70)
        print("V3 PEFT Launcher — LoRA | QLoRA | QA-LoRA")
        print("=" * 70)
        print("\nTraining Options:")
        print("  1. Train LoRA (V3)")
        print("  2. Train QLoRA (V3)")
        print("  3. Train QA-LoRA (V3)")
        print("  4. Train ALL V3 methods sequentially")
        print("\nEvaluation:")
        print("  5. Evaluate V3 checkpoints")
        print("\n0. Exit")

        choice = input("\nEnter choice (1-5, 0): ").strip()
        base = [sys.executable, "main_v3.py"]

        if choice == '1':
            if not check_training_readiness():
                continue
            run_command(base + ["lora"], label="Training LoRA V3")
        elif choice == '2':
            if not check_training_readiness():
                continue
            run_command(base + ["qlora"], label="Training QLoRA V3")
        elif choice == '3':
            if not check_training_readiness():
                continue
            run_command(base + ["qalora"], label="Training QA-LoRA V3")
        elif choice == '4':
            if not check_training_readiness():
                continue
            run_command(base + ["all"], label="Training ALL V3 Methods")
        elif choice == '5':
            run_command([sys.executable, "launcher_test_v3.py"], label="Evaluate V3 Checkpoints")
        elif choice == '0':
            print("\nGoodbye!")
            sys.exit(0)
        else:
            print("\nInvalid choice. Please enter 1-5 or 0.")

        input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    print(f"Using Python {sys.version_info.major}.{sys.version_info.minor}")
    if sys.version_info.major != 3 or sys.version_info.minor != 11:
        print("WARNING: You are NOT using Python 3.11!")
        print("   Please run this script with: py -3.11 launcher_v3.py")
    main()
