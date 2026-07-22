from utils.visualization import plot_confusion_matrix


def save_confusion_matrix(y_true, y_pred, class_names, out_dir, experiment_name):
    """Simple wrapper to save confusion matrix to the project's plots folder."""
    plot_confusion_matrix(y_true, y_pred, class_names, out_dir, experiment_name)
