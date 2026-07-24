"""
GLCM Texture Feature Branch for Hybrid CNN-Texture Classification

Based on Quilondrino et al. (2025): hybrid CNN + GLCM features achieves
99.57% on PlantVillage and 91.47% on PlantDoc, versus <30% for vanilla CNN.

This module extracts Haralick texture features from input images and feeds
them alongside CNN features into a small MLP before the final classifier.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
import numpy as np


class GLCMFeatureExtractor(nn.Module):
    """
    Extracts Haralick texture features from grayscale images using GLCM.
    
    Features extracted:
    - Contrast
    - Homogeneity (Inverse Difference Moment)
    - Energy (Angular Second Moment)
    - Correlation
    - Entropy
    
    Computed at multiple distances and angles for robustness.
    """
    
    def __init__(self, distances: list = None, angles: list = None):
        super().__init__()
        if distances is None:
            distances = [1, 2, 3]
        if angles is None:
            angles = [0, 45, 90, 135]
            
        self.distances = distances
        self.angles = angles
        self.num_features = len(distances) * len(angles) * 4  # 4 features per distance/angle
        
    def _compute_glcm(self, img: torch.Tensor, distance: int, angle_deg: float) -> torch.Tensor:
        """Compute GLCM for a single image."""
        h, w = img.shape
        glcm = torch.zeros(256, 256, device=img.device, dtype=torch.float32)
        
        angle_rad = torch.deg2rad(torch.tensor(angle_deg, device=img.device))
        dx = int(round(torch.cos(angle_rad).item()))
        dy = int(round(torch.sin(angle_rad).item()))
        
        y_indices = torch.arange(distance * abs(dy), h - distance * abs(dy), device=img.device)
        x_indices = torch.arange(distance * abs(dx), w - distance * abs(dx), device=img.device)
        
        if len(y_indices) == 0 or len(x_indices) == 0:
            return glcm
            
        grid_y, grid_x = torch.meshgrid(y_indices, x_indices, indexing='ij')
        
        i_vals = img[grid_y, grid_x]
        j_vals = img[grid_y + dy * distance, grid_x + dx * distance]
        
        mask = (j_vals >= 0) & (j_vals < 256)
        i_vals = i_vals[mask].long()
        j_vals = j_vals[mask].long()
        
        glcm.index_put_((i_vals, j_vals), torch.ones_like(i_vals, dtype=torch.float32), accumulate=True)
        
        glcm = glcm + glcm.T
        
        total = glcm.sum()
        if total > 0:
            glcm = glcm / total
            
        return glcm
    
    def _extract_features(self, glcm: torch.Tensor, eps: float = 1e-10) -> torch.Tensor:
        """Extract Haralick features from a normalized GLCM."""
        i_indices = torch.arange(256, device=glcm.device).view(-1, 1)
        j_indices = torch.arange(256, device=glcm.device).view(1, -1)
        
        contrast = torch.sum(glcm * ((i_indices - j_indices) ** 2))
        
        homogeneity = torch.sum(glcm / (1.0 + torch.abs(i_indices - j_indices) + eps))
        
        energy = torch.sum(glcm ** 2)
        
        mu_i = torch.sum(glcm * i_indices, dim=1, keepdim=True).sum()
        mu_j = torch.sum(glcm * j_indices, dim=0, keepdim=True).sum()
        mu_i = mu_i.expand_as(glcm)
        mu_j = mu_j.expand_as(glcm)
        
        sigma_i = torch.sqrt(torch.sum(glcm * ((i_indices - mu_i) ** 2)))
        sigma_j = torch.sqrt(torch.sum(glcm * ((j_indices - mu_j) ** 2)))
        
        correlation = torch.sum(glcm * ((i_indices - mu_i) * (j_indices - mu_j)))
        if sigma_i * sigma_j > eps:
            correlation = correlation / (sigma_i * sigma_j)
        
        features = torch.stack([
            contrast,
            homogeneity,
            energy,
            correlation
        ])
        
        return features
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W) - expects RGB images
        Returns:
            GLCM features of shape (B, num_features)
        """
        batch_size = x.size(0)
        
        # Convert to grayscale and uint8
        x_gray = 0.299 * x[:, 0, :, :] + 0.587 * x[:, 1, :, :] + 0.114 * x[:, 2, :, :]
        x_gray = (x_gray * 255).clamp(0, 255).long()
        
        all_features = []
        
        for b in range(batch_size):
            img = x_gray[b]
            sample_features = []
            
            for dist in self.distances:
                for angle in self.angles:
                    glcm = self._compute_glcm(img, dist, angle)
                    features = self._extract_features(glcm)
                    sample_features.append(features)
            
            all_features.append(torch.cat(sample_features))
        
        return torch.stack(all_features)


class HybridClassifier(nn.Module):
    """
    Hybrid CNN + GLTM Classifier
    
    Concatenates CNN backbone features with extracted GLCM texture features
    and feeds them through a small MLP before final classification.
    
    Architecture:
    - CNN backbone (EfficientNet-B0): 1280-dim global average pooled features
    - GLCM branch: variable-dim texture features
    - Fusion MLP: concat -> MLP -> num_classes
    """
    
    def __init__(self, cnn_feature_dim: int, glcm_feature_dim: int, num_classes: int, dropout: float = 0.3):
        super().__init__()
        
        self.fusion_mlp = nn.Sequential(
            nn.Linear(cnn_feature_dim + glcm_feature_dim, 512),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(256, num_classes)
        )
        
        self.glcm_extractor = GLCMFeatureExtractor()
        
    def forward(self, cnn_features: torch.Tensor, images: torch.Tensor) -> torch.Tensor:
        glcm_features = self.glcm_extractor(images)
        combined = torch.cat([cnn_features, glcm_features], dim=1)
        return self.fusion_mlp(combined)
