import os
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix


def plot_training_curves(train_losses: list, val_losses: list, train_accs: list, val_accs: list, 
                         plot_dir: str, experiment_name: str, run_tag: str = None):
    os.makedirs(plot_dir, exist_ok=True)
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Loss curves
    ax1.plot(train_losses, label='Train Loss')
    ax1.plot(val_losses, label='Val Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.set_title('Training and Validation Loss')
    ax1.legend()
    ax1.grid(True)
    
    # Accuracy curves
    ax2.plot(train_accs, label='Train Accuracy')
    ax2.plot(val_accs, label='Val Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.set_title('Training and Validation Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    suffix = f'_{run_tag}' if run_tag else ''
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{experiment_name}{suffix}_training_curves.png'))
    plt.close()


def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, classes: list, 
                          plot_dir: str, experiment_name: str, run_tag: str = None):
    os.makedirs(plot_dir, exist_ok=True)
    
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(15, 15))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.xticks(rotation=90)
    plt.yticks(rotation=0)
    suffix = f'_{run_tag}' if run_tag else ''
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{experiment_name}{suffix}_confusion_matrix.png'))
    plt.close()


def plot_class_metrics(metrics: dict, classes: list, plot_dir: str, experiment_name: str, run_tag: str = None):
    os.makedirs(plot_dir, exist_ok=True)
    
    precisions = [metrics['precision'][cls] for cls in classes]
    recalls = [metrics['recall'][cls] for cls in classes]
    f1s = [metrics['f1'][cls] for cls in classes]
    
    x = np.arange(len(classes))
    width = 0.25
    
    fig, ax = plt.subplots(figsize=(15, 7))
    rects1 = ax.bar(x - width, precisions, width, label='Precision')
    rects2 = ax.bar(x, recalls, width, label='Recall')
    rects3 = ax.bar(x + width, f1s, width, label='F1-Score')
    
    ax.set_xlabel('Class')
    ax.set_ylabel('Score')
    ax.set_title('Class-wise Metrics')
    ax.set_xticks(x)
    ax.set_xticklabels(classes, rotation=90)
    ax.legend()
    ax.grid(True, axis='y')
    
    suffix = f'_{run_tag}' if run_tag else ''
    plt.tight_layout()
    plt.savefig(os.path.join(plot_dir, f'{experiment_name}{suffix}_class_metrics.png'))
    plt.close()
