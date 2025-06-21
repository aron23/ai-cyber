"""
Patch Embedding Module for Vision Transformers

This module provides various patch embedding strategies optimized for
network packet image data.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, List, Tuple


class PatchEmbedding(nn.Module):
    """
    Standard patch embedding layer for Vision Transformer.
    Converts packet images into patch embeddings using convolutional projection.
    """
    
    def __init__(self, 
                 image_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 1,
                 embed_dim: int = 768,
                 norm_layer: Optional[nn.Module] = None):
        """
        Args:
            image_size: Size of input square image
            patch_size: Size of each patch
            in_channels: Number of input channels (1 for grayscale packet images)
            embed_dim: Dimension of patch embeddings
            norm_layer: Optional normalization layer
        """
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Convolutional projection
        self.proj = nn.Conv2d(in_channels, embed_dim, 
                            kernel_size=patch_size, stride=patch_size)
        
        self.norm = norm_layer(embed_dim) if norm_layer else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, C, H, W)
            
        Returns:
            Patch embeddings of shape (B, num_patches, embed_dim)
        """
        B, C, H, W = x.shape
        assert H == W == self.image_size, \
            f"Input size ({H}x{W}) doesn't match expected size ({self.image_size}x{self.image_size})"
        
        # Extract and project patches
        x = self.proj(x)  # (B, embed_dim, n_h, n_w)
        x = x.flatten(2)  # (B, embed_dim, num_patches)
        x = x.transpose(1, 2)  # (B, num_patches, embed_dim)
        x = self.norm(x)
        
        return x


class HybridPatchEmbedding(nn.Module):
    """
    Hybrid CNN-Transformer patch embedding.
    Uses lightweight CNN features before patch extraction for richer representations.
    Particularly effective for packet data where local byte patterns are important.
    """
    
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 1,
                 embed_dim: int = 768,
                 feature_dim: int = 64):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Lightweight CNN backbone
        self.backbone = nn.Sequential(
            # First conv block
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(32),
            nn.GELU(),
            
            # Second conv block  
            nn.Conv2d(32, feature_dim, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(feature_dim),
            nn.GELU(),
        )
        
        # Patch projection
        self.proj = nn.Conv2d(feature_dim, embed_dim,
                            kernel_size=patch_size, stride=patch_size)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Extract CNN features then create patch embeddings."""
        # Extract local features
        x = self.backbone(x)
        
        # Project to patch embeddings
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        
        return x


class AdaptivePatchEmbedding(nn.Module):
    """
    Adaptive patch embedding with multi-scale patches.
    Captures both fine-grained and coarse patterns in packet data.
    """
    
    def __init__(self,
                 image_size: int = 224,
                 patch_sizes: List[int] = [8, 16, 32],
                 in_channels: int = 1,
                 embed_dim: int = 768):
        super().__init__()
        self.image_size = image_size
        self.patch_sizes = sorted(patch_sizes)
        
        # Ensure embed_dim is divisible by number of scales
        assert embed_dim % len(patch_sizes) == 0, \
            f"embed_dim {embed_dim} must be divisible by number of patch sizes {len(patch_sizes)}"
        
        dim_per_scale = embed_dim // len(patch_sizes)
        
        # Multi-scale patch extractors
        self.projections = nn.ModuleList([
            nn.Conv2d(in_channels, dim_per_scale,
                     kernel_size=ps, stride=ps)
            for ps in patch_sizes
        ])
        
        # Learnable scale embeddings
        self.scale_embeddings = nn.ParameterList([
            nn.Parameter(torch.zeros(1, 1, dim_per_scale))
            for _ in patch_sizes
        ])
        
        # Initialize scale embeddings
        for scale_emb in self.scale_embeddings:
            nn.init.trunc_normal_(scale_emb, std=0.02)
            
        # Calculate total number of patches
        self.num_patches = sum((image_size // ps) ** 2 for ps in patch_sizes)
        
    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, List[int]]:
        """
        Extract multi-scale patches.
        
        Returns:
            patches: Combined patches from all scales (B, total_patches, embed_dim)
            patch_counts: Number of patches from each scale
        """
        B = x.shape[0]
        all_patches = []
        patch_counts = []
        
        for proj, scale_emb, patch_size in zip(self.projections, 
                                              self.scale_embeddings, 
                                              self.patch_sizes):
            # Extract patches at this scale
            patches = proj(x)  # (B, dim_per_scale, H', W')
            patches = patches.flatten(2).transpose(1, 2)  # (B, num_patches, dim_per_scale)
            
            # Add scale embedding
            patches = patches + scale_emb
            
            all_patches.append(patches)
            patch_counts.append(patches.shape[1])
        
        # Concatenate all scales
        x = torch.cat(all_patches, dim=1)  # (B, total_patches, embed_dim)
        
        return x, patch_counts


class ConvStemPatchEmbedding(nn.Module):
    """
    Patch embedding with convolutional stem.
    Gradually reduces spatial dimensions while increasing channels.
    More parameter efficient than single large convolution.
    """
    
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 1,
                 embed_dim: int = 768):
        super().__init__()
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_patches = (image_size // patch_size) ** 2
        
        # Convolutional stem
        self.conv1 = nn.Conv2d(in_channels, embed_dim // 8, 
                              kernel_size=3, stride=2, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(embed_dim // 8)
        
        self.conv2 = nn.Conv2d(embed_dim // 8, embed_dim // 4,
                              kernel_size=3, stride=2, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(embed_dim // 4)
        
        self.conv3 = nn.Conv2d(embed_dim // 4, embed_dim // 2,
                              kernel_size=3, stride=2, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(embed_dim // 2)
        
        # Final projection to embed_dim
        kernel_size = image_size // 8  # After 3 stride-2 convs
        self.proj = nn.Conv2d(embed_dim // 2, embed_dim,
                            kernel_size=kernel_size // (image_size // patch_size))
        
        self.act = nn.GELU()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply convolutional stem then create patches."""
        # Convolutional stem
        x = self.act(self.bn1(self.conv1(x)))
        x = self.act(self.bn2(self.conv2(x)))
        x = self.act(self.bn3(self.conv3(x)))
        
        # Final projection
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        
        return x


def create_patch_embedding(
    image_size: int = 224,
    patch_size: int = 16,
    in_channels: int = 1,
    embed_dim: int = 768,
    embedding_type: str = 'standard',
    **kwargs
) -> nn.Module:
    """
    Factory function to create patch embedding modules.
    
    Args:
        image_size: Input image size
        patch_size: Patch size (or list of sizes for adaptive)
        in_channels: Number of input channels
        embed_dim: Embedding dimension
        embedding_type: Type of embedding ('standard', 'hybrid', 'adaptive', 'convstem')
        **kwargs: Additional arguments for specific embedding types
        
    Returns:
        Patch embedding module
    """
    if embedding_type == 'standard':
        return PatchEmbedding(image_size, patch_size, in_channels, embed_dim)
    
    elif embedding_type == 'hybrid':
        feature_dim = kwargs.get('feature_dim', 64)
        return HybridPatchEmbedding(image_size, patch_size, in_channels, embed_dim, feature_dim)
    
    elif embedding_type == 'adaptive':
        patch_sizes = kwargs.get('patch_sizes', [8, 16, 32])
        return AdaptivePatchEmbedding(image_size, patch_sizes, in_channels, embed_dim)
    
    elif embedding_type == 'convstem':
        return ConvStemPatchEmbedding(image_size, patch_size, in_channels, embed_dim)
    
    else:
        raise ValueError(f"Unknown embedding type: {embedding_type}")


# Utility functions for patch analysis
def analyze_patch_statistics(patches: torch.Tensor) -> dict:
    """
    Analyze statistical properties of patches.
    
    Args:
        patches: Tensor of shape (B, num_patches, patch_dim)
        
    Returns:
        Dictionary with patch statistics
    """
    B, N, D = patches.shape
    
    # Compute statistics
    patch_mean = patches.mean(dim=2)  # (B, N)
    patch_std = patches.std(dim=2)    # (B, N)
    
    # Compute diversity metrics
    mean_diversity = patch_mean.std(dim=1).mean().item()
    spatial_correlation = F.cosine_similarity(
        patches[:, :-1, :], patches[:, 1:, :], dim=2
    ).mean().item()
    
    return {
        'mean_patch_activation': patch_mean.mean().item(),
        'std_patch_activation': patch_std.mean().item(),
        'patch_diversity': mean_diversity,
        'spatial_correlation': spatial_correlation,
        'dead_patches': (patch_mean.abs() < 0.01).float().mean().item()
    }


def visualize_patch_importance(
    patches: torch.Tensor,
    attention_weights: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """
    Create importance map from patches for visualization.
    
    Args:
        patches: Patch embeddings (B, num_patches, embed_dim)
        attention_weights: Optional attention weights (B, num_patches)
        
    Returns:
        Importance maps (B, H, W) suitable for visualization
    """
    B, N, D = patches.shape
    
    # Compute importance scores
    if attention_weights is not None:
        importance = attention_weights
    else:
        # Use L2 norm of patch embeddings as importance
        importance = patches.norm(dim=2)  # (B, N)
    
    # Reshape to 2D grid (assuming square grid)
    grid_size = int(np.sqrt(N))
    importance_map = importance.reshape(B, grid_size, grid_size)
    
    # Upsample for visualization
    importance_map = F.interpolate(
        importance_map.unsqueeze(1),
        size=(224, 224),
        mode='bilinear',
        align_corners=False
    ).squeeze(1)
    
    return importance_map