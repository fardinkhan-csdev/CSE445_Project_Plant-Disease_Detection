import os
import sys
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.trainer import BaseTrainer
from models.peft.qalora import get_qalora_model


class QALoRATrainer(BaseTrainer):
    def __init__(self, train_loader, val_loader, num_classes,
                 base_config_path: str = 'config/base_config_v3.yaml',
                 qalora_config_path: str = 'config/qalora_config.yaml',
                 class_weights=None):
        with open(qalora_config_path, 'r') as f:
            qalora_config = yaml.safe_load(f)['peft']

        model = get_qalora_model(
            num_classes=num_classes,
            rank=qalora_config['rank'],
            alpha=qalora_config['alpha'],
            dropout=qalora_config['dropout'],
            num_groups=qalora_config.get('num_groups', 4),
            target_modules=qalora_config.get('target_modules'),
        )

        super().__init__(model, train_loader, val_loader,
                        config_path=base_config_path,
                        experiment_name='qalora',
                        class_weights=class_weights)

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
