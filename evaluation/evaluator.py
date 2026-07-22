import torch
import numpy as np
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Tuple
from .metrics import calculate_metrics


class Evaluator:
    def __init__(self, model: nn.Module, device: torch.device, idx_to_class: Dict[int, str]):
        self.model = model
        self.device = device
        self.idx_to_class = idx_to_class
    
    def evaluate(self, data_loader: DataLoader) -> Tuple[Dict, np.ndarray, np.ndarray, np.ndarray, list]:
        self.model.eval()
        all_preds = []
        all_labels = []
        all_probs = []
        all_paths = []

        with torch.no_grad():
            for batch in data_loader:
                # Dataset now returns (image, label, crop, disease, image_path)
                images = batch[0]
                labels = batch[1]
                image_paths = batch[4]

                images = images.to(self.device)
                labels = labels.to(self.device)

                outputs = self.model(images)
                probs = torch.softmax(outputs, dim=1)
                _, preds = torch.max(outputs, 1)

                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
                all_probs.extend(probs.cpu().numpy())
                all_paths.extend(image_paths)

        y_true = np.array(all_labels)
        y_pred = np.array(all_preds)
        y_probs = np.array(all_probs)

        metrics = calculate_metrics(y_true, y_pred, self.idx_to_class, y_probs)

        return metrics, y_true, y_pred, y_probs, all_paths
