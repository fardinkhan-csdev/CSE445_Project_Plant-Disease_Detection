import os
import sys
import yaml
import torch
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from training.trainer_v2 import BaseTrainerV2
from models.peft.qklora import get_qklora_model


class QKLoRATrainerV2(BaseTrainerV2):
    def __init__(self, train_loader, val_loader, num_classes, 
                 base_config_path: str = 'config/base_config_v2.yaml',
                 qklora_config_path: str = 'config/qklora_config.yaml',
                 class_weights=None):
        with open(qklora_config_path, 'r') as f:
            qklora_config = yaml.safe_load(f)['peft']
        
        quant_config = qklora_config.get('quantization', {})
        model = get_qklora_model(
            num_classes=num_classes,
            rank=qklora_config['rank'],
            alpha=qklora_config['alpha'],
            dropout=qklora_config['dropout'],
            q_rank=quant_config.get('q_rank', 16),
            k_rank=quant_config.get('k_rank', 4),
            q_target_modules=qklora_config.get('q_target_modules'),
            k_target_modules=qklora_config.get('k_target_modules'),
        )
        
        super().__init__(model, train_loader, val_loader, 
                        config_path=base_config_path, 
                        experiment_name='qklora_v2',
                        class_weights=class_weights)
    
    def count_trainable_parameters(self) -> int:
        return sum(p.numel() for p in self.model.parameters() if p.requires_grad)
