import os
import sys
import json
import torch
import pandas as pd
from datetime import datetime
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from data.data_loader import get_data_loaders
from evaluation.evaluator import Evaluator
from utils.visualization import plot_confusion_matrix, plot_class_metrics
from training.lora_trainer import LoRATrainer
from training.qlora_trainer import QLoRATrainer
from training.qklora_trainer import QKLoRATrainer


def run_experiment(trainer_class, trainer_name, train_loader, val_loader, test_loader, class_info, resume=False):
    print(f"\n{'='*50}")
    print(f"Running {trainer_name} Experiment")
    print(f"{'='*50}")
    
    # Create trainer
    trainer = trainer_class(
        train_loader,
        val_loader,
        class_info['num_classes'],
        class_weights=class_info.get('class_weights'),
    )
    
    # Count trainable parameters
    trainable_params = trainer.count_trainable_parameters()
    print(f"Trainable Parameters: {trainable_params:,}")
    
    # Train
    trainer.train(resume=resume)

    # Load best checkpoint (so test reflects best val model)
    trainer.load_checkpoint('best')

    # Evaluate on test set
    evaluator = Evaluator(trainer.model, trainer.device, class_info['idx_to_class'])
    test_metrics, y_true, y_pred, y_probs, image_paths = evaluator.evaluate(test_loader)
    
    # Print test metrics
    print(f"\nTest Metrics for {trainer_name}:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision (Macro): {test_metrics['precision_macro']:.4f}")
    print(f"Recall (Macro): {test_metrics['recall_macro']:.4f}")
    print(f"F1 (Macro): {test_metrics['f1_macro']:.4f}")
    
    # Plot confusion matrix and class metrics
    class_names = [class_info['idx_to_class'][i] for i in range(class_info['num_classes'])]
    plot_confusion_matrix(y_true, y_pred, class_names, trainer.plot_dir, trainer_name, run_tag=trainer.run_id)
    plot_class_metrics(test_metrics, class_names, trainer.plot_dir, trainer_name, run_tag=trainer.run_id)
    
    # Save results
    results = {
        'experiment': trainer_name,
        'trainable_parameters': trainable_params,
        'training_time': trainer.training_time,
        'peak_gpu_memory': trainer.peak_gpu_memory,
        'best_val_acc': trainer.best_val_acc,
        'test_accuracy': test_metrics['accuracy'],
        'test_precision_macro': test_metrics['precision_macro'],
        'test_recall_macro': test_metrics['recall_macro'],
        'test_f1_macro': test_metrics['f1_macro']
    }
    
    return results


def main(trainer_name='all', resume=False):
    # Get data loaders
    print("Loading data...")
    train_loader, val_loader, test_loader, class_info = get_data_loaders()
    print(f"Number of classes: {class_info['num_classes']}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    # Map trainer names to classes
    trainer_map = {
        'lora': LoRATrainer,
        'qlora': QLoRATrainer,
        'qklora': QKLoRATrainer
    }
    
    # Run experiments
    all_results = []
    
    if trainer_name == 'all':
        # Run all three in sequence
        for name, trainer_class in trainer_map.items():
            results = run_experiment(trainer_class, name, train_loader, val_loader, test_loader, class_info, resume=resume)
            all_results.append(results)
    else:
        # Run single experiment
        if trainer_name not in trainer_map:
            print(f"ERROR: Unknown trainer '{trainer_name}'. Valid options: lora, qlora, qklora, all")
            return
        results = run_experiment(trainer_map[trainer_name], trainer_name, train_loader, val_loader, test_loader, class_info, resume=resume)
        all_results.append(results)
    
    # Save all results
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)

        # --- 1. Save timestamped backup (for safety) ---
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S_%f')
        results_path = os.path.join('experiments', 'results', f'experiment_results_{timestamp}.csv')
        results_df.to_csv(results_path, index=False)
        print(f"\nTimestamped backup saved to {results_path}")

        # --- 2. Upsert into canonical experiment_results.csv ---
        # Load existing rows, replace any row for the same method, then append new ones.
        canonical_path = os.path.join('experiments', 'results', 'experiment_results.csv')
        try:
            if os.path.exists(canonical_path):
                existing_df = pd.read_csv(canonical_path)
                # Drop old rows for methods we are now updating
                new_methods = results_df['experiment'].str.lower().tolist()
                existing_df = existing_df[~existing_df['experiment'].str.lower().isin(new_methods)]
                combined_df = pd.concat([existing_df, results_df], ignore_index=True)
            else:
                combined_df = results_df
            combined_df.to_csv(canonical_path, index=False)
            print(f"Canonical experiment_results.csv updated: {canonical_path}")
        except Exception as e:
            print(f"Warning: could not update canonical CSV: {e}")

        print("\nExperiment Results Summary:")
        print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
