"""Evaluate all V3 checkpoints non-interactively."""
import os
import sys
import yaml
import torch
import csv
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from data.data_loader import get_data_loaders
from training.lora_trainer_v3 import LoRATrainerV3
from training.qlora_trainer_v3 import QLoRATrainerV3
from training.qalora_trainer import QALoRATrainer
from evaluation.evaluator import Evaluator
from training.trainer import prepare_model_state_dict_for_load

TRAINER_MAP = {
    'lora': LoRATrainerV3,
    'qlora': QLoRATrainerV3,
    'qalora': QALoRATrainer,
}

EXPERIMENT_NAME_MAP = {
    'lora': 'lora_v3',
    'qlora': 'qlora_v3',
    'qalora': 'qalora',
}


def list_checkpoints_for_experiment(experiment_name: str, checkpoint_dir: str):
    files = []
    p = Path(checkpoint_dir)
    if not p.exists():
        return files
    for f in sorted(p.iterdir()):
        if f.is_file() and f.name.startswith(f"{experiment_name}_") and f.suffix == '.pth':
            files.append(str(f))
    return files


def checkpoint_size_mb(checkpoint_path: str) -> float:
    return os.path.getsize(checkpoint_path) / (1024 * 1024)


def evaluate_checkpoint(trainer, test_loader, class_info, checkpoint_path):
    ckpt = torch.load(checkpoint_path, map_location=trainer.device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    state_dict = prepare_model_state_dict_for_load(state_dict)
    trainer.model.load_state_dict(state_dict, strict=False)

    evaluator = Evaluator(trainer.model, trainer.device, class_info['idx_to_class'])
    metrics, y_true, y_pred, y_probs, image_paths = evaluator.evaluate(test_loader)

    eval_dir = os.path.join('experiments', 'results', 'eval_v3')
    os.makedirs(eval_dir, exist_ok=True)
    base_name = Path(checkpoint_path).stem
    csv_path = os.path.join(eval_dir, f"{base_name}_confidences.csv")

    with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.writer(csvfile)
        header = ['image_path', 'true_label', 'pred_label'] + [f'prob_{i}' for i in range(len(y_probs[0]))]
        writer.writerow(header)
        for img_p, t, p, probs in zip(image_paths, y_true, y_pred, y_probs):
            writer.writerow([img_p, int(t), int(p)] + [float(x) for x in probs.tolist()])

    return metrics, csv_path


def run_test_for(experiment_key: str):
    print(f"\n{'='*70}")
    print(f"Evaluating {experiment_key.upper()} V3")
    print(f"{'='*70}")

    train_loader, val_loader, test_loader, class_info = get_data_loaders(config_path='config/base_config_v3.yaml')
    trainer_class = TRAINER_MAP[experiment_key]

    trainer = trainer_class(
        train_loader,
        val_loader,
        class_info['num_classes'],
        class_weights=class_info.get('class_weights'),
    )

    experiment_name = EXPERIMENT_NAME_MAP[experiment_key]
    checkpoint_dir = trainer.checkpoint_dir
    best_path = os.path.join(checkpoint_dir, f"{experiment_name}_best.pth")

    if not os.path.exists(best_path):
        print(f"Best checkpoint not found at {best_path}. Available files:")
        for f in sorted(Path(checkpoint_dir).iterdir()):
            if f.is_file() and f.name.startswith(f"{experiment_name}_"):
                print(f"  {f.name}")
        return None

    print(f"Loading best checkpoint: {best_path}")
    metrics, csv_path = evaluate_checkpoint(trainer, test_loader, class_info, best_path)

    print(f"Test metrics for {experiment_key} (best):")
    print(f"  Accuracy            : {metrics['accuracy']:.4f}")
    print(f"  F1 Macro            : {metrics['f1_macro']:.4f}")
    print(f"  Precision Macro     : {metrics['precision_macro']:.4f}")
    print(f"  Recall Macro        : {metrics['recall_macro']:.4f}")
    print(f"Binary metrics (best):")
    print(f"  Accuracy            : {metrics['binary_accuracy']:.4f}")
    print(f"  F1                  : {metrics['binary_f1']:.4f}")
    print(f"  ROC AUC             : {metrics['binary_roc_auc']:.4f}")
    print(f"Correctness (best):")
    print(f"  Both Correct        : {metrics['both_correct_pct']:.2f}%")
    print(f"  Name Only Correct   : {metrics['name_only_correct_pct']:.2f}%")
    print(f"  Disease Only Correct: {metrics['disease_only_correct_pct']:.2f}%")
    print(f"  None Correct        : {metrics['none_correct_pct']:.2f}%")
    print(f"Per-sample confidences saved to: {csv_path}")

    return {
        'method': experiment_key,
        'checkpoint': best_path,
        'accuracy': metrics['accuracy'],
        'f1_macro': metrics['f1_macro'],
        'precision_macro': metrics['precision_macro'],
        'recall_macro': metrics['recall_macro'],
        'binary_accuracy': metrics['binary_accuracy'],
        'binary_f1': metrics['binary_f1'],
        'binary_roc_auc': metrics['binary_roc_auc'],
        'both_correct_pct': metrics['both_correct_pct'],
        'name_only_correct_pct': metrics['name_only_correct_pct'],
        'disease_only_correct_pct': metrics['disease_only_correct_pct'],
        'none_correct_pct': metrics['none_correct_pct'],
        'confidences_csv': csv_path,
    }


def main():
    print('\n' + '='*70)
    print('V3 Test Launcher — Evaluating ALL methods')
    print('='*70)

    results = []
    for key in ['lora', 'qlora', 'qalora']:
        try:
            result = run_test_for(key)
            if result:
                results.append(result)
        except Exception as e:
            print(f"ERROR evaluating {key}: {e}")
            import traceback
            traceback.print_exc()

    if results:
        ranked = sorted(results, key=lambda r: r['accuracy'], reverse=True)
        print('\n' + '='*70)
        print('FINAL RANKING (by accuracy)')
        print('='*70)
        for rank, row in enumerate(ranked, start=1):
            print(f"#{rank} {row['method'].upper():<10} Acc={row['accuracy']:.4f}  F1={row['f1_macro']:.4f}  BinAcc={row['binary_accuracy']:.4f}  BinAUC={row['binary_roc_auc']:.4f}")

        rank_path = os.path.join('experiments', 'results', 'eval_v3', 'all_v3_ranking.csv')
        with open(rank_path, 'w', newline='', encoding='utf-8') as rf:
            writer = csv.writer(rf)
            writer.writerow([
                'rank', 'method', 'checkpoint', 'accuracy', 'f1_macro',
                'precision_macro', 'recall_macro', 'binary_accuracy', 'binary_f1', 'binary_roc_auc',
                'both_correct_pct', 'name_only_correct_pct', 'disease_only_correct_pct', 'none_correct_pct',
                'confidences_csv'
            ])
            for rank, row in enumerate(ranked, start=1):
                writer.writerow([
                    rank, row['method'], row['checkpoint'], row['accuracy'], row['f1_macro'],
                    row['precision_macro'], row['recall_macro'], row['binary_accuracy'],
                    row['binary_f1'], row['binary_roc_auc'], row['both_correct_pct'],
                    row['name_only_correct_pct'], row['disease_only_correct_pct'],
                    row['none_correct_pct'], row['confidences_csv'],
                ])
        print(f"\nRanking saved to {rank_path}")


if __name__ == '__main__':
    main()
