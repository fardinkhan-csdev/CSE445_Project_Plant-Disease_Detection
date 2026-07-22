import os
import sys
import subprocess

# Project root directory
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

def run_command(cmd_list):
    """Run a command and wait for it to finish"""
    try:
        subprocess.run(
            cmd_list,
            cwd=PROJECT_ROOT,
            check=True
        )
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Action interrupted by user!")


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
        print("\n❌ Training assets are not fully prepared.")
        print(f"   {exc}")
        print("   Run: py -3.11 download_assets.py")
        print("   Then start the launcher again.")
        return False

def run_resume():
    """Run continue.py one-click resume flow (subprocess avoids 'continue' keyword conflict)."""
    run_command([sys.executable, os.path.join(PROJECT_ROOT, "continue.py")])


def main():
    while True:
        print("\n" + "="*70)
        print("🌿 Leaf Disease Classification - Trainer Launcher")
        print("="*70)
        print("\nPlease choose an option:")
        print("  1. Train LoRA")
        print("  2. Train QLoRA")
        print("  3. Train Q/K LoRA")
        print("  4. Train ALL THREE (one after another)")
        print("  5. Resume interrupted training")
        print("  6. Exit Launcher")
        
        choice = input("\nEnter your choice (1-6): ").strip()
        
        if choice == '1':
            if not check_training_readiness():
                continue
            print("\n🚀 Starting LoRA Training...")
            run_command([sys.executable, "main.py", "lora"])
        elif choice == '2':
            if not check_training_readiness():
                continue
            print("\n🚀 Starting QLoRA Training...")
            run_command([sys.executable, "main.py", "qlora"])
        elif choice == '3':
            if not check_training_readiness():
                continue
            print("\n🚀 Starting Q/K LoRA Training...")
            run_command([sys.executable, "main.py", "qklora"])
        elif choice == '4':
            if not check_training_readiness():
                continue
            print("\n🚀 Starting ALL THREE Experiments...")
            run_command([sys.executable, "main.py", "all"])
        elif choice == '5':
            run_resume()
        elif choice == '6':
            print("\n👋 Goodbye!")
            sys.exit(0)
        else:
            print("\n❌ Invalid choice! Please enter a number between 1 and 6!")
        
        # Pause after each action
        input("\nPress Enter to return to menu...")

if __name__ == '__main__':
    # Make sure we're using Python 3.11
    print(f"🔍 Using Python {sys.version_info.major}.{sys.version_info.minor}")
    if sys.version_info.major != 3 or sys.version_info.minor != 11:
        print("\n⚠️  WARNING: You are NOT using Python 3.11!")
        print("   Please run this script with: py -3.11 launcher.py")
    main()
