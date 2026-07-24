import os
import sys


def main():
    print("="*60)
    print("Leaf Disease Classification - V3 PEFT Track")
    print("LoRA vs QLoRA vs QA-LoRA")
    print("="*60)

    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)

    trainer_name = 'all'
    resume = False

    args = sys.argv[1:]
    if '--resume' in args:
        resume = True
        args.remove('--resume')

    if len(args) > 0:
        trainer_name = args[0].lower()

    from experiments.experiment_runner_v3 import main as run_experiments
    run_experiments(trainer_name, resume=resume)


if __name__ == '__main__':
    main()
