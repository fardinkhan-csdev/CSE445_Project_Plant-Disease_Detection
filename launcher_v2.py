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


def main():
    while True:
        print("\n" + "=" * 70)
        print("V2 Enhanced Training Launcher - Leaf Disease Classification")
        print("=" * 70)
        print("\nTraining Options:")
        print("  1. Train LoRA (baseline v2)")
        print("  2. Train QLoRA (v2)")
        print("  3. Train Q/K LoRA (v2)")
        print("  4. Train ALL v2 methods sequentially")
        print("\nAdaptation Options:")
        print("  5. MixStyle vs Baseline comparison (LoRA)")
        print("  6. Launch PlantDoc evaluation on v2 checkpoints")
        print("\n0. Exit")

        choice = input("\nEnter choice (1-6, 0): ").strip()

        base = [sys.executable, "main_v2.py"]

        if choice == '1':
            run_command(base + ["lora"], label="Training LoRA V2")
        elif choice == '2':
            run_command(base + ["qlora"], label="Training QLoRA V2")
        elif choice == '3':
            run_command(base + ["qklora"], label="Training Q/K LoRA V2")
        elif choice == '4':
            run_command(base, label="Training ALL V2 Methods")
        elif choice == '5':
            run_command([sys.executable, "compare_mixstyle_vs_baseline.py"], label="MixStyle vs Baseline Comparison")
        elif choice == '6':
            run_command([sys.executable, "launcher_plantdoc.py"], label="PlantDoc Evaluator (V1/V2 checkpoints)")
        elif choice == '0':
            print("\nGoodbye!")
            sys.exit(0)
        else:
            print("\nInvalid choice. Please enter 1-6 or 0.")

        input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    main()
