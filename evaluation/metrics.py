import numpy as np
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score
from typing import Dict, List, Optional, Tuple


def split_class_name(class_name: str) -> Tuple[str, str]:
    """Splits class name like Tomato___Late_blight into (Crop, Disease)."""
    if "___" in class_name:
        parts = class_name.split("___")
        return parts[0], parts[1]
    else:
        return class_name, "healthy" if "healthy" in class_name.lower() else "unknown"


def calculate_metrics(y_true: np.ndarray, y_pred: np.ndarray, idx_to_class: Dict[int, str], y_probs: Optional[np.ndarray] = None) -> Dict:
    # Overall metrics
    accuracy = accuracy_score(y_true, y_pred)
    precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
    recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
    f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
    
    # Class-wise metrics
    precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
    recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
    f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
    
    # Create class-wise dictionaries
    precision = {}
    recall = {}
    f1 = {}
    for idx, cls in idx_to_class.items():
        precision[cls] = precision_per_class[idx]
        recall[cls] = recall_per_class[idx]
        f1[cls] = f1_per_class[idx]
        
    metrics = {
        'accuracy': accuracy,
        'precision_macro': precision_macro,
        'recall_macro': recall_macro,
        'f1_macro': f1_macro,
        'precision': precision,
        'recall': recall,
        'f1': f1
    }

    # Binary Evaluation (Diseased vs Healthy)
    healthy_indices = [idx for idx, name in idx_to_class.items() if "healthy" in name.lower()]
    y_true_bin = np.array([0 if label in healthy_indices else 1 for label in y_true])
    y_pred_bin = np.array([0 if pred in healthy_indices else 1 for pred in y_pred])

    metrics['binary_accuracy'] = accuracy_score(y_true_bin, y_pred_bin)
    metrics['binary_precision'] = precision_score(y_true_bin, y_pred_bin, zero_division=0)
    metrics['binary_recall'] = recall_score(y_true_bin, y_pred_bin, zero_division=0)
    metrics['binary_f1'] = f1_score(y_true_bin, y_pred_bin, zero_division=0)

    if y_probs is not None:
        y_probs_bin = 1.0 - np.sum(y_probs[:, healthy_indices], axis=1)
        metrics['binary_roc_auc'] = roc_auc_score(y_true_bin, y_probs_bin) if len(np.unique(y_true_bin)) > 1 else 0.0
    else:
        metrics['binary_roc_auc'] = 0.0

    # Correctness Aspects (Crop vs Disease)
    both_correct = 0
    name_only_correct = 0
    disease_only_correct = 0
    none_correct = 0

    for true_idx, pred_idx in zip(y_true, y_pred):
        true_crop, true_disease = split_class_name(idx_to_class[true_idx])
        pred_crop, pred_disease = split_class_name(idx_to_class[pred_idx])
        crop_matches = (true_crop.lower() == pred_crop.lower())
        disease_matches = (true_disease.lower() == pred_disease.lower())
        if crop_matches and disease_matches:
            both_correct += 1
        elif crop_matches:
            name_only_correct += 1
        elif disease_matches:
            disease_only_correct += 1
        else:
            none_correct += 1

    total_samples = len(y_true)
    metrics['both_correct_pct'] = (both_correct / total_samples) * 100.0
    metrics['name_only_correct_pct'] = (name_only_correct / total_samples) * 100.0
    metrics['disease_only_correct_pct'] = (disease_only_correct / total_samples) * 100.0
    metrics['none_correct_pct'] = (none_correct / total_samples) * 100.0

    return metrics
