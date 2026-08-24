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
        print("PlantDoc V3 Real-World Evaluator")
        print("=" * 70)
        print("\nEvaluation Options:")
        print("  1. Raw PlantDoc images (all splits)")
        print("  2. Segmented foreground + white background (test)")
        print("  3. Style + CORAL domain adaptation (test)")
        print("  4. k-NN on frozen backbone (test)")
        print("  5. Multi-Resolution Inference Pyramid (test)")
        print("  6. Image Quality Gating (test)")
        print("  A. Run ALL tests (1+2)")
        print("\nDataset:")
        print("  B. Download PlantDoc dataset")
        print("  0. Exit")

        choice = input("\nEnter choice (1-6, A, B, 0): ").strip().upper()

        base = [sys.executable, "run_plantdoc_evaluation.py", "--v3"]

        if choice == '1':
            run_command(base, label="1st Test: Raw PlantDoc Images (V3)")
        elif choice == '2':
            run_command(base + ["--segmented", "--splits", "test"], label="2nd Test: Segmented (V3)")
        elif choice == '3':
            run_command(base + ["--style-norm", "--splits", "test"], label="3a: Style Normalization (V3)")
            run_command(base + ["--feature-align", "--splits", "test"], label="3b: CORAL (V3)")
            run_command(base + ["--style-norm", "--feature-align", "--splits", "test"], label="3c: StyleNorm+CORAL (V3)")
        elif choice == '4':
            run_command(base + ["--knn", "--splits", "test"], label="4th Test: k-NN on frozen backbone (V3)")
        elif choice == '5':
            run_command(base + ["--multires", "--splits", "test"], label="5th Test: Multi-Resolution (V3)")
        elif choice == '6':
            run_command(base + ["--quality-gate", "--splits", "test"], label="6th Test: Quality Gating (V3)")
        elif choice == 'A':
            run_command(base, label="1st Test: Raw Images (V3)")
            run_command(base + ["--segmented", "--splits", "test"], label="2nd Test: Segmented (V3)")
        elif choice == 'B':
            run_command([sys.executable, "download_plantdoc.py"], label="Downloading PlantDoc Dataset")
        elif choice == '0':
            print("\nGoodbye!")
            sys.exit(0)
        else:
            print("\nInvalid choice. Please enter 1-6, A, B, or 0.")

        input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    main()
