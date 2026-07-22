import torch.nn as nn


class PlantDiseaseClassifier(nn.Module):
    """Classifier head for EfficientNet-B0 plant disease classification.

    Replaces the default EfficientNet head with a task-specific head.
    Note: No Softmax is applied here — PyTorch's CrossEntropyLoss internally
    applies log_softmax, so adding Softmax here would corrupt the loss.
    For inference probabilities, apply torch.softmax() externally (see evaluator.py).

    Architecture:
        Dropout(p=0.2) → Linear(in_features → num_classes)

    The upstream EfficientNet-B0 already includes an AdaptiveAvgPool2d that
    reduces spatial dimensions to (1, 1) before this head, so no explicit
    Global Average Pooling is needed here.
    """

    def __init__(self, in_features: int, num_classes: int, dropout: float = 0.2):
        super().__init__()
        self.dropout = nn.Dropout(p=dropout, inplace=True)
        self.fc = nn.Linear(in_features, num_classes)

    def forward(self, x):
        x = self.dropout(x)
        x = self.fc(x)
        return x
