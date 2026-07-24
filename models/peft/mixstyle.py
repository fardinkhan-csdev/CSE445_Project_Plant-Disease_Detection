"""
MixStyle: Feature Style Mixing for Domain Generalization

Implementation of Zhou et al., "Domain Generalization with MixStyle", ICLR 2021.
https://arxiv.org/abs/2104.00608

This module mixes instance-level feature statistics (mean and standard deviation)
within a minibatch during training to synthesize novel domain styles and improve
out-of-distribution robustness without requiring any target domain data.
"""

import torch
import torch.nn as nn
from typing import List, Optional


class MixStyle(nn.Module):
    """
    MixStyle Module
    
    Algorithm (from paper Algorithm 1 - Replace mode):
    1. Given feature map X in R^{NxCHW}
    2. For each sample i: compute per-instance statistics
       mu_i = mean(X_i) over spatial dimensions
       sigma_i = std(X_i) over spatial dimensions
    3. With probability p, generate random permutation pi of batch indices
    4. Mix statistics: for each sample i, replace stats with those from pi[i]:
       X'_i = (X_i - mu_i) * sigma_{pi[i]} / sigma_i + mu_{pi[i]}
    5. Return mixed features X'
    """
    
    def __init__(self, p: float = 0.5, eps: float = 1e-6):
        super().__init__()
        self.p = p      # Probability of applying MixStyle
        self.eps = eps  # Numerical stability term
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if not self.training or torch.rand(1).item() > self.p:
            return x
            
        if x.size(0) <= 1:
            return x
            
        batch_size = x.size(0)
        
        # Random permutation of batch indices
        perm = torch.randperm(batch_size, device=x.device)
        
        # Per-instance statistics over spatial dimensions (H, W)
        mu = x.mean(dim=[2, 3], keepdim=True)
        var = x.var(dim=[2, 3], keepdim=True, unbiased=False)
        std = var.sqrt()
        
        # Shuffle statistics
        mu_mix = mu[perm]
        std_mix = std[perm]
        
        # Normalize with original stats, denormalize with shuffled stats
        x_norm = (x - mu) / (std + self.eps)
        x_mixed = x_norm * std_mix + mu_mix
        
        return x_mixed


def register_mixstyle_hooks(model: nn.Module, mixstyle: MixStyle, layer_indices: List[int]) -> list:
    """
    Register MixStyle forward hooks on selected layers of a model.
    
    For EfficientNet-B0, layer_indices refers to indices into model.features.
    Example: [0, 2, 4] injects after stem and first two MBConv stacks.
    
    Returns list of hook handles (call remove() to clean up).
    """
    handles = []
    
    if not hasattr(model, 'features'):
        raise ValueError("Model does not have a 'features' attribute. Ensure it's EfficientNet-B0.")
        
    for idx, layer in enumerate(model.features):
        if idx in layer_indices:
            def _hook(module, inp, out, ms=mixstyle):
                return ms(out)
            handle = layer.register_forward_hook(_hook)
            handles.append(handle)
            
    return handles


def remove_mixstyle_hooks(handles: list) -> None:
    for handle in handles:
        handle.remove()
