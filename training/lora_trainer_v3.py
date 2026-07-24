import os
import sys
import yaml

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.trainer import BaseTrainer
from models.peft.lora_v3 import get_lora_v3_model


class LoRATrainerV3(BaseTrainer):
    def __init__(self, train_loader, val_loader, num_classes,
                 base_config_path: str = 'config/base_config_v3.yaml',
                 lora_config_path: str = 'config/lora_config.yaml',
                 class_weights=None):
        with open(lora_config_path, 'r') as f:
            lora_config = yaml.safe_load(f)['peft']

        model = get_lora_v3_model(
            num_classes=num_classes,
            rank=lora_config['rank'],
            alpha=lora_config['alpha'],
            dropout=lora_config['dropout'],
            target_modules=lora_config.get('target_modules'),
        )

        super().__init__(model, train_loader, val_loader,
                         config_path=base_config_path,
                         experiment_name='lora_v3',
                         class_weights=class_weights)

    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
