import os
import sys


def main():
    print("="*60)
    print("Leaf Disease Classification - LoRA vs QLoRA vs Q/K LoRA")
    print("="*60)
    
    # Add the project root to the path
    project_root = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, project_root)
    
    # Determine which trainer to run
    trainer_name = 'all'
    resume = False
    
    args = sys.argv[1:]
    if '--resume' in args:
        resume = True
        args.remove('--resume')
        
    if len(args) > 0:
        trainer_name = args[0].lower()
    
    # Run experiments
    from experiments.experiment_runner import main as run_experiments
    run_experiments(trainer_name, resume=resume)


if __name__ == '__main__':
    main()
