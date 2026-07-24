import os
import sys
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.trainer import BaseTrainer
from models.peft.qlora_v3 import get_qlora_v3_model


class QLoRATrainerV3(BaseTrainer):
    def __init__(self, train_loader, val_loader, num_classes,
                 base_config_path: str = 'config/base_config_v3.yaml',
                 qlora_config_path: str = 'config/qlora_config.yaml',
                 class_weights=None):
        with open(qlora_config_path, 'r') as f:
            qlora_config = yaml.safe_load(f)['peft']

        model = get_qlora_v3_model(
            num_classes=num_classes,
            rank=qlora_config['rank'],
            alpha=qlora_config['alpha'],
            dropout=qlora_config['dropout'],
            target_modules=qlora_config.get('target_modules'),
        )

        super().__init__(model, train_loader, val_loader,
                         config_path=base_config_path,
                         experiment_name='qlora_v3',
                         class_weights=class_weights)

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
