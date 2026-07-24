"""
AdaBN: Adaptive Batch Normalization

Implementation of Li et al., "Revisiting Batch Normalization for Practical
Domain Adaptation", AAAI 2016.
https://arxiv.org/abs/1412.6470

AdaBN recomputes BatchNorm layer statistics (running_mean, running_var) on
unlabeled target-domain data without updating weights. This parameter-free
test-time adaptation closes a significant portion of the domain gap.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader


class AdaBNAdapter:
    """
    AdaBN Test-Time Adapter
    
    Re-estimates BatchNorm statistics on target-domain (unlabeled) data.
    No weights are updated; only running_mean and running_var are refreshed.
    
    Args:
        model: Neural network with BatchNorm2d layers
        device: Compute device
        num_passes: How many times to iterate through target data for stats
    """
    
    def __init__(self, model: nn.Module, device: str = 'cuda', num_passes: int = 1):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        self.num_passes = num_passes
        
    def adapt(self, dataloader: DataLoader) -> nn.Module:
        self.model.train()
        
        # Freeze all parameters - we only update BN running statistics
        for param in self.model.parameters():
            param.requires_grad = False
            
        # Identify all BatchNorm layers
        bn_layers = []
        for module in self.model.modules():
            if isinstance(module, (nn.BatchNorm2d, nn.SyncBatchNorm)):
                bn_layers.append(module)
                
        print(f"AdaBN: Adapting {len(bn_layers)} BatchNorm layers over {self.num_passes} pass(es)...")
        
        # Forward passes through target data to accumulate BN stats
        for pass_idx in range(self.num_passes):
            for batch_idx, batch in enumerate(dataloader):
                x = batch[0].to(self.device, non_blocking=True)
                with torch.no_grad():
                    self.model(x)
                    
        self.model.eval()
        print(f"AdaBN: Adaptation complete.")
        return self.model
