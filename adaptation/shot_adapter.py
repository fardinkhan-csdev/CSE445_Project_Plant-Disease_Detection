"""
SHOT-Style Unsupervised Adaptation

Implementation of Liang et al., "Do We Really Need to Access Source Data?
Source Hypothesis Transfer for Unsupervised Domain Adaptation", ICML 2020.
https://arxiv.org/abs/2002.10757

SHOT freezes the source classifier and adapts the feature extractor on unlabeled
target data via entropy minimization and self-supervised pseudo-labeling.

For this PEFT setup, the frozen classifier acts as the fixed "source hypothesis"
and only LoRA adapters remain trainable. This yields a lightweight, label-free
adaptation that respects the parameter-efficient fine-tuning constraint.

IMPORTANT: This adapter should be run on LoRA-trained checkpoints first.
QLoRA/QKLoRA add INT8 quantization complexity; SHOT with those methods
requires explicit gradient-flow verification.
"""

import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader


class SHOTAdapter:
    """
    SHOT Unsupervised Domain Adaptation Adapter
    
    Args:
        model: Pre-trained model (LoRA-adapted EfficientNet-B0)
        device: Compute device
    """
    
    def __init__(self, model: torch.nn.Module, device: str = 'cuda'):
        self.model = model
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
    def adapt(self, target_loader: DataLoader,
              epochs: int = 10,
              entropy_weight: float = 1.0,
              consistency_weight: float = 0.5,
              lr: float = 1e-5) -> torch.nn.Module:
        """
        Run SHOT-style adaptation on unlabeled target data.
        
        Args:
            target_loader: DataLoader for unlabeled target-domain images
            epochs: Number of adaptation epochs
            entropy_weight: Weight for entropy minimization loss
            consistency_weight: Weight for pseudo-label consistency loss
            lr: Learning rate for LoRA adapters only
            
        Returns:
            Adapted model (frozen backbone + classifier; LoRA adapters trained)
        """
        self.model.to(self.device)
        self.model.train()
        
        # Freeze backbone and classifier - only LoRA adapters train
        frozen_count = 0
        trainable_count = 0
        trainable_params = []
        
        for name, param in self.model.named_parameters():
            is_lora = 'lora' in name.lower() or 'adapter' in name.lower()
            if is_lora:
                param.requires_grad = True
                trainable_count += param.numel()
                trainable_params.append(param)
            else:
                param.requires_grad = False
                frozen_count += param.numel()
                
        if not trainable_params:
            raise RuntimeError(
                "SHOT requires trainable LoRA parameters. "
                "Ensure the checkpoint contains LoRA adapters."
            )
            
        print(f"SHOT: {trainable_count:,} trainable LoRA params, {frozen_count:,} frozen params")
        
        optimizer = torch.optim.AdamW(trainable_params, lr=lr)
        
        for epoch in range(epochs):
            total_loss = 0.0
            batch_count = 0
            
            for batch in target_loader:
                x = batch[0].to(self.device, non_blocking=True)
                
                # Forward pass
                outputs = self.model(x)
                probs = torch.softmax(outputs, dim=1)
                log_probs = torch.log(probs + 1e-10)
                
                # Entropy minimization: sharpen predictions
                entropy = -(probs * log_probs).sum(dim=1)
                entropy_loss = entropy.mean()
                
                # Self-supervised pseudo-labeling: consistency with argmax
                pseudo_labels = outputs.argmax(dim=1).detach()
                consistency_loss = F.cross_entropy(outputs, pseudo_labels)
                
                loss = entropy_weight * entropy_loss + consistency_weight * consistency_loss
                
                optimizer.zero_grad()
                loss.backward()
                torch.nn.utils.clip_grad_norm_(trainable_params, max_norm=1.0)
                optimizer.step()
                
                total_loss += loss.item()
                batch_count += 1
                
            avg_loss = total_loss / batch_count if batch_count > 0 else 0.0
            print(f"SHOT Epoch {epoch+1}/{epochs} | Loss: {avg_loss:.4f}")
            
        # Freeze all parameters after adaptation
        for param in self.model.parameters():
            param.requires_grad = False
        self.model.eval()
        print("SHOT: Adaptation complete. Model is in eval mode with frozen parameters.")
        return self.model
