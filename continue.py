import os
import sys
import json
import subprocess

# Project root
PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
        sys.stderr.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass

STATUS_PATH = os.path.join(PROJECT_ROOT, 'experiments', 'results', 'training_status.json')


def load_status():
    """Load training_status.json. Returns None if it doesn't exist or is unreadable."""
    if not os.path.exists(STATUS_PATH):
        return None
    try:
        with open(STATUS_PATH, 'r') as f:
            return json.load(f)
    except Exception:
        return None


def print_header():
    print("\n" + "="*70)
    print("🌿 Leaf Disease Classification - Continue / Resume Launcher")
    print("="*70)


def main():
    print_header()

    status = load_status()

    # ── No status file at all ──────────────────────────────────────────────
    if status is None:
        print("\n❌ No running, left, or crashed session found.")
        print("   Start a new training session with the main launcher.")
        input("\nPress Enter to exit...")
        return

    experiment_name   = status.get('experiment_name', '?')
    current_status    = status.get('status', 'unknown')
    current_epoch     = status.get('current_epoch', 0)
    total_epochs      = status.get('total_epochs', 20)
    patience          = status.get('early_stopping_patience', 7)
    patience_counter  = status.get('patience_counter', 0)

    print(f"\n  📋 Last session info:")
    print(f"     Experiment   : {experiment_name.upper()}")
    print(f"     Status       : {current_status}")
    print(f"     Epoch reached: {current_epoch} / {total_epochs}")
    print(f"     Patience     : {patience_counter} / {patience}")

    # ── Completed normally ─────────────────────────────────────────────────
    if current_status == 'completed':
        print("\n✅ Training already completed successfully — nothing to resume.")
        input("\nPress Enter to exit...")
        return

    # ── Epoch limit reached ────────────────────────────────────────────────
    if current_epoch >= total_epochs:
        print("\n✅ Max epochs already reached — nothing to resume.")
        input("\nPress Enter to exit...")
        return

    # ── Early-stopping limit reached ───────────────────────────────────────
    if patience_counter >= patience:
        print("\n✅ Patience limit already reached — nothing to resume.")
        input("\nPress Enter to exit...")
        return

    # ── Crashed / interrupted: offer to resume ─────────────────────────────
    remaining = total_epochs - current_epoch
    print(f"\n⚡ Crashed or interrupted session detected!")
    print(f"   Will resume from epoch {current_epoch + 1} "
          f"({remaining} epoch(s) remaining).")

    answer = input("\n  Continue training? [Y/n]: ").strip().lower()
    if answer not in ('', 'y', 'yes'):
        print("\n👋 Aborted. Goodbye!")
        return

    print(f"\n🚀 Resuming {experiment_name.upper()} training from epoch {current_epoch + 1}...\n")
    try:
        subprocess.run(
            [sys.executable, 'main.py', experiment_name, '--resume'],
            cwd=PROJECT_ROOT,
            check=True
        )
        print("\n🎉 Training resumed and completed!")
    except subprocess.CalledProcessError as e:
        print(f"\n❌ Training exited with error code {e.returncode}")
    except KeyboardInterrupt:
        print("\n\n⚠️  Interrupted again. Run this launcher to resume later.")

    input("\nPress Enter to exit...")


if __name__ == '__main__':
    # Make sure we're using Python 3.11
    print(f"🔍 Using Python {sys.version_info.major}.{sys.version_info.minor}")
    if sys.version_info.major != 3 or sys.version_info.minor != 11:
        print("\n⚠️  WARNING: You are NOT using Python 3.11!")
        print("   Please run this script with: py -3.11 continue.py")
    main()
