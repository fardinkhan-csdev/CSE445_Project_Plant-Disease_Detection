import os
import sys
import yaml
import torch
import csv
from pathlib import Path

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, PROJECT_ROOT)

from data.data_loader import get_data_loaders
from training.lora_trainer import LoRATrainer
from training.qlora_trainer import QLoRATrainer
from training.qklora_trainer import QKLoRATrainer
from evaluation.evaluator import Evaluator
from training.trainer import prepare_model_state_dict_for_load

TRAINER_MAP = {
    'lora': LoRATrainer,
    'qlora': QLoRATrainer,
    'qklora': QKLoRATrainer
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


def evaluate_checkpoint(trainer, test_loader, class_info, checkpoint_path, save_confidences=True):
    # Load checkpoint's model_state_dict
    ckpt = torch.load(checkpoint_path, map_location=trainer.device)
    state_dict = ckpt['model_state_dict'] if 'model_state_dict' in ckpt else ckpt
    state_dict = prepare_model_state_dict_for_load(state_dict)
    trainer.model.load_state_dict(state_dict, strict=False)

    evaluator = Evaluator(trainer.model, trainer.device, class_info['idx_to_class'])
    metrics, y_true, y_pred, y_probs, image_paths = evaluator.evaluate(test_loader)

    # Save per-sample confidences
    eval_dir = os.path.join('experiments', 'results', 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    base_name = Path(checkpoint_path).stem
    csv_path = os.path.join(eval_dir, f"{base_name}_confidences.csv")

    if save_confidences:
        with open(csv_path, 'w', newline='', encoding='utf-8') as csvfile:
            writer = csv.writer(csvfile)
            header = ['image_path', 'true_label', 'pred_label'] + [f'prob_{i}' for i in range(len(y_probs[0]))]
            writer.writerow(header)
            for img_p, t, p, probs in zip(image_paths, y_true, y_pred, y_probs):
                writer.writerow([img_p, int(t), int(p)] + [float(x) for x in probs.tolist()])

    return metrics, csv_path


def run_test_for(experiment_key: str):
    # Load data
    train_loader, val_loader, test_loader, class_info = get_data_loaders()
    trainer_class = TRAINER_MAP[experiment_key]

    # Create trainer instance (we only need its model and device)
    trainer = trainer_class(
        train_loader,
        val_loader,
        class_info['num_classes'],
        class_weights=class_info.get('class_weights'),
    )

    # Default: load best checkpoint
    checkpoint_dir = trainer.checkpoint_dir
    best_path = os.path.join(checkpoint_dir, f"{experiment_key}_best.pth")
    if os.path.exists(best_path):
        print(f"Loading best checkpoint: {best_path}")
        metrics, csv_path = evaluate_checkpoint(trainer, test_loader, class_info, best_path)
        print(f"Test metrics for {experiment_key} (best): Accuracy={metrics['accuracy']:.4f}, F1_macro={metrics['f1_macro']:.4f}")
        print(f"Binary metrics (best): Accuracy={metrics['binary_accuracy']:.4f}, F1={metrics['binary_f1']:.4f}, ROC AUC={metrics['binary_roc_auc']:.4f}")
        print(f"Correctness (best): Both Correct={metrics['both_correct_pct']:.2f}%, Crop Only={metrics['name_only_correct_pct']:.2f}%, None Correct={metrics['none_correct_pct']:.2f}%")
        print(f"Per-sample confidences saved to: {csv_path}")
    else:
        print(f"Best checkpoint not found at {best_path}.")

    # Offer evaluate-all
    all_ckpts = list_checkpoints_for_experiment(experiment_key, checkpoint_dir)
    if all_ckpts:
        ans = input("Do you want to evaluate ALL checkpoints for this experiment and rank them? (y/N): ").strip().lower()
        if ans == 'y':
            results = []
            for ck in all_ckpts:
                print(f"Evaluating {ck} ...")
                m, csvp = evaluate_checkpoint(trainer, test_loader, class_info, ck)
                results.append({
                    'checkpoint': ck,
                    'size_mb': round(checkpoint_size_mb(ck), 2),
                    'accuracy': m['accuracy'],
                    'f1_macro': m['f1_macro'],
                    'binary_accuracy': m['binary_accuracy'],
                    'binary_f1': m['binary_f1'],
                    'binary_roc_auc': m['binary_roc_auc'],
                    'both_correct_pct': m['both_correct_pct'],
                    'name_only_correct_pct': m['name_only_correct_pct'],
                    'disease_only_correct_pct': m['disease_only_correct_pct'],
                    'none_correct_pct': m['none_correct_pct'],
                    'confidences_csv': csvp,
                })

            ranked_results = sorted(results, key=lambda row: row['accuracy'], reverse=True)
            for rank, row in enumerate(ranked_results, start=1):
                row['rank'] = rank

            # Save ranking
            rank_path = os.path.join('experiments', 'results', 'eval', f'{experiment_key}_checkpoint_ranking.csv')
            with open(rank_path, 'w', newline='', encoding='utf-8') as rf:
                writer = csv.writer(rf)
                writer.writerow([
                    'rank', 'checkpoint', 'size_mb', 'accuracy', 'f1_macro',
                    'binary_accuracy', 'binary_f1', 'binary_roc_auc',
                    'both_correct_pct', 'name_only_correct_pct', 'disease_only_correct_pct', 'none_correct_pct',
                    'confidences_csv'
                ])
                for row in ranked_results:
                    writer.writerow([
                        row['rank'],
                        row['checkpoint'],
                        row['size_mb'],
                        row['accuracy'],
                        row['f1_macro'],
                        row['binary_accuracy'],
                        row['binary_f1'],
                        row['binary_roc_auc'],
                        row['both_correct_pct'],
                        row['name_only_correct_pct'],
                        row['disease_only_correct_pct'],
                        row['none_correct_pct'],
                        row['confidences_csv'],
                    ])
            print(f"Ranking saved to {rank_path}")


def main():
    print('\n' + '='*70)
    print('🌿 Leaf Disease Classification - Test Launcher')
    print('='*70)
    print('\nPlease choose an option to TEST:')
    print('  1. LORA')
    print('  2. QLoRA')
    print('  3. Q/K LoRA')
    print('  4. ALL (run each)')
    print('  5. Exit')

    choice = input('\nEnter your choice (1-5): ').strip()
    if choice == '1':
        run_test_for('lora')
    elif choice == '2':
        run_test_for('qlora')
    elif choice == '3':
        run_test_for('qklora')
    elif choice == '4':
        for key in TRAINER_MAP.keys():
            print(f"\n--- Testing {key} ---")
            run_test_for(key)
    else:
        print('Exiting test launcher')


if __name__ == '__main__':
    main()
