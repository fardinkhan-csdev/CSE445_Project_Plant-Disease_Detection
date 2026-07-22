import os
import sys
import subprocess

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)


def run_command(cmd_list):
    """Run a command and wait for it to finish."""
    try:
        subprocess.run(cmd_list, cwd=PROJECT_ROOT, check=True)
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Command failed with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\n\n⚠️ Action interrupted by user!")


def main():
    while True:
        print("\n" + "=" * 70)
        print("PlantDoc Real-World Evaluator & Launcher")
        print("=" * 70)
        print("\nPlease choose an option:")
        print("  1. Download PlantDoc Dataset")
        print("  2. Run PlantDoc evaluation on train + valid + test")
        print("  3. Download PlantDoc and run full evaluation")
        print("  4. Exit Launcher")
        print("\nTip: option 2 evaluates all three trained checkpoints against the mapped PlantDoc splits and shows progress while running.")

        choice = input("\nEnter choice (1-4): ").strip()
        if choice == '1':
            run_command([sys.executable, "download_plantdoc.py"])
        elif choice == '2':
            run_command([sys.executable, "run_plantdoc_evaluation.py"])
        elif choice == '3':
            run_command([sys.executable, "download_plantdoc.py"])
            run_command([sys.executable, "run_plantdoc_evaluation.py"])
        elif choice == '4':
            print("\n👋 Goodbye!")
            sys.exit(0)
        else:
            print("\n❌ Invalid choice. Please enter a number between 1 and 4.")

        input("\nPress Enter to return to menu...")


if __name__ == "__main__":
    main()
