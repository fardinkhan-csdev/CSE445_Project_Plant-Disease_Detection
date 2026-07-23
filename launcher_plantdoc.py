import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

if sys.platform == 'win32':
    sys.stdout.reconfigure(encoding='utf-8', errors='ignore')
    sys.stderr.reconfigure(encoding='utf-8', errors='ignore')


def run_command(cmd_list, label=""):
    """Run a command and wait for it to finish."""
    if label:
        print(f"\n🚀 {label}")
        print("=" * 70)
    try:
        subprocess.run(cmd_list, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Action interrupted by user!")


def main():
    while True:
        print("\n" + "=" * 70)
        print("🌍 PlantDoc Real-World Evaluator & Launcher")
        print("=" * 70)
        print("\nEvaluation Options:")
        print("  1. 1st Test — Raw PlantDoc images (no segmentation)")
        print("  2. 2nd Test — Segmented foreground + white background")
        print("  3. 3rd Test — Cross-method ensemble (LoRA+QLoRA+QKLoRA, segmented)")
        print("  4. 4th Test — k-NN on frozen backbone (raw PlantDoc, no classifier)")
        print("  5. 5th Test — Multi-Resolution Inference Pyramid (no training)")
        print("  6. 6th Test — Image Quality Gating (exclude blurry/underexposed/occluded images)")
        print("  7. 7th Test — Style + CORAL (runs StyleNorm, CORAL, and Both sequentially)")
        print("  A. Run ALL SEVEN tests (1+2+3+4+5+6+7)")
        print("\nDataset:")
        print("  B. Download PlantDoc dataset")
        print("  0. Exit")
        print("\nTip: Options 1–7 show live tqdm progress bars during evaluation.")

        choice = input("\nEnter choice (1-9, A, B, 0): ").strip().upper()

        base = [sys.executable, "run_plantdoc_evaluation.py"]

        if choice == '1':
            run_command(base, label="Running 1st PlantDoc Test (Raw Images, all splits)")
        elif choice == '2':
            run_command(base + ["--segmented", "--splits", "test"], label="Running 2nd PlantDoc Test (Segmented, test only)")
        elif choice == '3':
            run_command(base + ["--ensemble", "--splits", "test"], label="Running 3rd PlantDoc Test (Ensemble from saved CSV, test only)")
        elif choice == '4':
            run_command(base + ["--knn", "--splits", "test"], label="Running 4th PlantDoc Test (k-NN on frozen backbone, test only)")
        elif choice == '5':
            run_command(base + ["--multires", "--splits", "test"], label="Running 5th PlantDoc Test (Multi-Resolution, test only)")
        elif choice == '6':
            run_command(base + ["--quality-gate", "--splits", "test"], label="Running 6th PlantDoc Test (Image Quality Gating, test only)")
        elif choice == '7':
            run_command(base + ["--style-norm", "--splits", "test"], label="7a: Style Normalization Only")
            run_command(base + ["--feature-align", "--splits", "test"], label="7b: CORAL Feature Whitening Only")
            run_command(base + ["--style-norm", "--feature-align", "--splits", "test"], label="7c: Style Normalization + CORAL (Both Together)")
        elif choice == 'A':
            run_command(base, label="1st Test: Raw Images")
            run_command(base + ["--segmented", "--splits", "test"], label="2nd Test: Segmented")
            run_command(base + ["--ensemble", "--splits", "test"], label="3rd Test: Ensemble from saved CSV")
            run_command(base + ["--knn", "--splits", "test"], label="4th Test: k-NN on frozen backbone")
            run_command(base + ["--multires", "--splits", "test"], label="5th Test: Multi-Resolution Inference Pyramid")
            run_command(base + ["--quality-gate", "--splits", "test"], label="6th Test: Image Quality Gating")
            run_command(base + ["--style-norm", "--splits", "test"], label="7a: Style Normalization Only")
            run_command(base + ["--feature-align", "--splits", "test"], label="7b: CORAL Feature Whitening Only")
            run_command(base + ["--style-norm", "--feature-align", "--splits", "test"], label="7c: Style Normalization + CORAL (Both Together)")
        elif choice == 'B':
            run_command([sys.executable, "download_plantdoc.py"], label="Downloading PlantDoc Dataset")
        elif choice == '0':
            print("\n👋 Goodbye!")
            sys.exit(0)
        else:
            print("\n❌ Invalid choice. Please enter 1-7, A, B, or 0.")

        input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    main()
