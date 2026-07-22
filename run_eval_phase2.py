import os
import sys
import csv

PROJECT_ROOT = os.path.dirname(os.path.abspath(__file__))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from launcher_test import TRAINER_MAP, checkpoint_size_mb, evaluate_checkpoint
from data.data_loader import get_data_loaders

def eval_experiment_all(exp_key):
    print(f"==================================================")
    print(f"Evaluating Canonical Checkpoints for: {exp_key.upper()}")
    print(f"==================================================")
    
    train_loader, val_loader, test_loader, class_info = get_data_loaders()
    trainer_class = TRAINER_MAP[exp_key]
    
    trainer = trainer_class(
        train_loader,
        val_loader,
        class_info['num_classes'],
        class_weights=class_info.get('class_weights'),
    )
    
    checkpoint_dir = trainer.checkpoint_dir
    canonical_files = [f"{exp_key}_best.pth", f"{exp_key}_last.pth", f"{exp_key}_latest.pth"]
    all_ckpts = [os.path.join(checkpoint_dir, f) for f in canonical_files if os.path.exists(os.path.join(checkpoint_dir, f))]
    
    if not all_ckpts:
        print(f"No canonical checkpoints found for {exp_key} in {checkpoint_dir}")
        return

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

    eval_dir = os.path.join('experiments', 'results', 'eval')
    os.makedirs(eval_dir, exist_ok=True)
    rank_path = os.path.join(eval_dir, f'{exp_key}_checkpoint_ranking.csv')
    
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
                row['rank'], row['checkpoint'], row['size_mb'],
                row['accuracy'], row['f1_macro'], row['binary_accuracy'],
                row['binary_f1'], row['binary_roc_auc'], row['both_correct_pct'],
                row['name_only_correct_pct'], row['disease_only_correct_pct'],
                row['none_correct_pct'], row['confidences_csv'],
            ])
    print(f"Ranking successfully saved to {rank_path}")

if __name__ == "__main__":
    eval_experiment_all('qklora')
