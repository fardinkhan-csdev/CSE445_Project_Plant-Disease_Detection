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
from training.lora_trainer_v2 import LoRATrainerV2
from training.qlora_trainer_v2 import QLoRATrainerV2
from training.qklora_trainer_v2 import QKLoRATrainerV2


def run_experiment(trainer_class, trainer_name, train_loader, val_loader, test_loader, class_info, resume=False):
    print(f"\n{'='*50}")
    print(f"Running {trainer_name} Experiment (V2)")
    print(f"{'='*50}")
    
    trainer = trainer_class(
        train_loader,
        val_loader,
        class_info['num_classes'],
        class_weights=class_info.get('class_weights'),
    )
    
    trainable_params = trainer.count_trainable_parameters()
    print(f"Trainable Parameters: {trainable_params:,}")
    
    trainer.train(resume=resume)

    trainer.load_checkpoint('best')

    evaluator = Evaluator(trainer.model, trainer.device, class_info['idx_to_class'])
    test_metrics, y_true, y_pred, y_probs, image_paths = evaluator.evaluate(test_loader)
    
    print(f"\nTest Metrics for {trainer_name}:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"Precision (Macro): {test_metrics['precision_macro']:.4f}")
    print(f"Recall (Macro): {test_metrics['recall_macro']:.4f}")
    print(f"F1 (Macro): {test_metrics['f1_macro']:.4f}")
    
    class_names = [class_info['idx_to_class'][i] for i in range(class_info['num_classes'])]
    plot_confusion_matrix(y_true, y_pred, class_names, trainer.plot_dir, trainer_name, run_tag=trainer.run_id)
    plot_class_metrics(test_metrics, class_names, trainer.plot_dir, trainer_name, run_tag=trainer.run_id)
    
    results = {
        'experiment': trainer_name,
        'version': 'v2',
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
    print("Loading data...")
    train_loader, val_loader, test_loader, class_info = get_data_loaders(config_path='config/base_config_v2.yaml')
    print(f"Number of classes: {class_info['num_classes']}")
    print(f"Train batches: {len(train_loader)}, Val batches: {len(val_loader)}, Test batches: {len(test_loader)}")
    
    trainer_map = {
        'lora': LoRATrainerV2,
        'qlora': QLoRATrainerV2,
        'qklora': QKLoRATrainerV2
    }
    
    all_results = []
    
    if trainer_name == 'all':
        for name, trainer_class in trainer_map.items():
            results = run_experiment(trainer_class, name, train_loader, val_loader, test_loader, class_info, resume=resume)
            all_results.append(results)
    else:
        if trainer_name not in trainer_map:
            print(f"ERROR: Unknown trainer '{trainer_name}'. Valid options: lora, qlora, qklora, all")
            return
        results = run_experiment(trainer_map[trainer_name], trainer_name, train_loader, val_loader, test_loader, class_info, resume=resume)
        all_results.append(results)
    
    if len(all_results) > 0:
        results_df = pd.DataFrame(all_results)
        results_path = os.path.join('experiments', 'results', f'experiment_results_v2_{datetime.now().strftime("%Y%m%d_%H%M%S")}.csv')
        os.makedirs(os.path.dirname(results_path), exist_ok=True)
        results_df.to_csv(results_path, index=False)
        print(f"\nV2 results saved to {results_path}")
        print("\nExperiment Results Summary:")
        print(results_df.to_string(index=False))


if __name__ == '__main__':
    main()
