"""
Vision Transformer for Packet-based Malware Detection

This module implements Vision Transformer architectures specifically optimized
for network packet image data and malware classification.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Optional, Tuple, List
import math

from .patch_embed import create_patch_embedding
from .position_embed import create_position_embedding


class MultiHeadSelfAttention(nn.Module):
    """Multi-head self-attention mechanism optimized for packet data."""
    
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.1):
        super().__init__()
        assert embed_dim % num_heads == 0
        
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5
        
        # Linear projections for Q, K, V
        self.qkv = nn.Linear(embed_dim, embed_dim * 3, bias=False)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Args:
            x: Input tensor (B, N, C)
            mask: Optional attention mask (B, N, N)
        """
        B, N, C = x.shape
        
        # Generate Q, K, V
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, self.head_dim).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # (B, num_heads, N, head_dim)
        
        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale  # (B, num_heads, N, N)
        
        # Apply mask if provided
        if mask is not None:
            attn = attn.masked_fill(mask == 0, -1e9)
        
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)
        
        # Apply attention to values
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)  # (B, N, C)
        x = self.proj(x)
        x = self.dropout(x)
        
        return x


class TransformerBlock(nn.Module):
    """Transformer encoder block with packet-specific optimizations."""
    
    def __init__(self, 
                 embed_dim: int,
                 num_heads: int,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 drop_path: float = 0.0):
        super().__init__()
        
        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadSelfAttention(embed_dim, num_heads, dropout)
        
        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, embed_dim),
            nn.Dropout(dropout)
        )
        
        # Stochastic depth
        self.drop_path = DropPath(drop_path) if drop_path > 0.0 else nn.Identity()
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Self-attention with residual connection
        x = x + self.drop_path(self.attn(self.norm1(x)))
        
        # MLP with residual connection
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        
        return x


class DropPath(nn.Module):
    """Stochastic Depth (Drop Path) regularization."""
    
    def __init__(self, drop_prob: float = 0.0):
        super().__init__()
        self.drop_prob = drop_prob
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.drop_prob == 0.0 or not self.training:
            return x
        
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        output = x.div(keep_prob) * random_tensor
        
        return output


class PacketViT(nn.Module):
    """
    Vision Transformer for Packet-based Malware Detection.
    
    Optimized for network packet image data with various enhancements:
    - Flexible patch embedding strategies
    - Byte-order aware position embeddings
    - Packet-specific attention mechanisms
    """
    
    def __init__(self,
                 image_size: int = 224,
                 patch_size: int = 16,
                 in_channels: int = 1,
                 num_classes: int = 6,
                 embed_dim: int = 768,
                 depth: int = 12,
                 num_heads: int = 12,
                 mlp_ratio: float = 4.0,
                 dropout: float = 0.1,
                 drop_path_rate: float = 0.1,
                 patch_embedding_type: str = 'standard',
                 position_embedding_type: str = 'learnable'):
        """
        Args:
            image_size: Input image size
            patch_size: Patch size
            in_channels: Number of input channels (1 for grayscale packet images)
            num_classes: Number of output classes
            embed_dim: Embedding dimension
            depth: Number of transformer blocks
            num_heads: Number of attention heads
            mlp_ratio: MLP expansion ratio
            dropout: Dropout rate
            drop_path_rate: Stochastic depth rate
            patch_embedding_type: Type of patch embedding
            position_embedding_type: Type of position embedding
        """
        super().__init__()
        
        self.image_size = image_size
        self.patch_size = patch_size
        self.num_classes = num_classes
        self.embed_dim = embed_dim
        
        # Calculate number of patches
        self.num_patches = (image_size // patch_size) ** 2
        
        # Patch embedding
        self.patch_embed = create_patch_embedding(
            image_size=image_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim,
            embedding_type=patch_embedding_type
        )
        
        # CLS token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position embedding
        self.pos_embed = create_position_embedding(
            num_patches=self.num_patches,
            embed_dim=embed_dim,
            embedding_type=position_embedding_type,
            cls_token=True,
            patch_size=patch_size,
            image_size=image_size
        )
        
        self.pos_dropout = nn.Dropout(dropout)
        
        # Transformer blocks with stochastic depth
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.blocks = nn.ModuleList([
            TransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout,
                drop_path=dpr[i]
            )
            for i in range(depth)
        ])
        
        # Final layer norm
        self.norm = nn.LayerNorm(embed_dim)
        
        # Classification head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()
        
        # Initialize weights
        self._init_weights()
        
    def _init_weights(self):
        """Initialize model weights."""
        # Initialize CLS token
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        
        # Initialize classification head
        if isinstance(self.head, nn.Linear):
            nn.init.trunc_normal_(self.head.weight, std=0.02)
            nn.init.constant_(self.head.bias, 0)
        
        # Initialize other parameters
        self.apply(self._init_module_weights)
        
    def _init_module_weights(self, m):
        """Initialize weights for individual modules."""
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)
            
    def forward(self, x: torch.Tensor, packet_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass.
        
        Args:
            x: Input tensor (B, C, H, W)
            packet_lengths: Optional packet lengths for position embedding
            
        Returns:
            Logits (B, num_classes)
        """
        B = x.shape[0]
        
        # Patch embedding
        x = self.patch_embed(x)  # (B, num_patches, embed_dim)
        
        # Add CLS token
        cls_tokens = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat([cls_tokens, x], dim=1)  # (B, num_patches + 1, embed_dim)
        
        # Position embedding
        if hasattr(self.pos_embed, 'forward') and packet_lengths is not None:
            x = self.pos_embed(x, packet_lengths)
        else:
            x = self.pos_embed(x)
        
        x = self.pos_dropout(x)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x)
        
        x = self.norm(x)
        
        # Classification using CLS token
        cls_output = x[:, 0]  # (B, embed_dim)
        logits = self.head(cls_output)  # (B, num_classes)
        
        return logits
    
    def get_attention_maps(self, x: torch.Tensor, layer_idx: int = -1) -> torch.Tensor:
        """
        Extract attention maps for visualization.
        
        Args:
            x: Input tensor (B, C, H, W)
            layer_idx: Layer index to extract attention from (-1 for last layer)
            
        Returns:
            Attention maps (B, num_heads, num_patches + 1, num_patches + 1)
        """
        B = x.shape[0]
        
        # Forward pass up to the specified layer
        x = self.patch_embed(x)
        cls_tokens = self.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        x = self.pos_embed(x)
        x = self.pos_dropout(x)
        
        # Forward through blocks
        target_layer = len(self.blocks) + layer_idx if layer_idx < 0 else layer_idx
        
        for i, block in enumerate(self.blocks):
            if i == target_layer:
                # Extract attention from this layer
                x_norm = block.norm1(x)
                B, N, C = x_norm.shape
                
                qkv = block.attn.qkv(x_norm).reshape(B, N, 3, block.attn.num_heads, block.attn.head_dim)
                qkv = qkv.permute(2, 0, 3, 1, 4)
                q, k, v = qkv[0], qkv[1], qkv[2]
                
                attn = (q @ k.transpose(-2, -1)) * block.attn.scale
                attn = F.softmax(attn, dim=-1)
                
                return attn
            
            x = block(x)
            
        return None


def create_packet_vit(model_size: str = 'base', num_classes: int = 6, **kwargs) -> PacketViT:
    """
    Factory function to create PacketViT models of different sizes.
    
    Args:
        model_size: Model size ('tiny', 'small', 'base', 'large')
        num_classes: Number of output classes
        **kwargs: Additional arguments to override defaults
        
    Returns:
        PacketViT model
    """
    configs = {
        'tiny': {
            'image_size': 64,
            'patch_size': 8,
            'embed_dim': 192,
            'depth': 6,
            'num_heads': 3,
            'mlp_ratio': 4.0
        },
        'small': {
            'image_size': 128,
            'patch_size': 16,
            'embed_dim': 384,
            'depth': 8,
            'num_heads': 6,
            'mlp_ratio': 4.0
        },
        'base': {
            'image_size': 224,
            'patch_size': 16,
            'embed_dim': 768,
            'depth': 12,
            'num_heads': 12,
            'mlp_ratio': 4.0
        },
        'large': {
            'image_size': 224,
            'patch_size': 16,
            'embed_dim': 1024,
            'depth': 24,
            'num_heads': 16,
            'mlp_ratio': 4.0
        }
    }
    
    if model_size not in configs:
        raise ValueError(f"Unknown model size: {model_size}")
    
    config = configs[model_size]
    config.update(kwargs)  # Override with user-provided values
    config['num_classes'] = num_classes
    
    return PacketViT(**config)


class PacketViTWithFeatures(nn.Module):
    """
    PacketViT enhanced with additional feature inputs.
    Combines vision transformer with handcrafted features.
    """
    
    def __init__(self,
                 vit_config: dict,
                 feature_dim: int = 256,
                 fusion_method: str = 'concat'):
        """
        Args:
            vit_config: Configuration for base ViT model
            feature_dim: Dimension of additional features
            fusion_method: How to fuse ViT and features ('concat', 'add', 'attention')
        """
        super().__init__()
        
        # Base ViT model (without classification head)
        vit_config_copy = vit_config.copy()
        vit_config_copy['num_classes'] = 0  # No classification head
        self.vit = PacketViT(**vit_config_copy)
        
        self.embed_dim = vit_config['embed_dim']
        self.feature_dim = feature_dim
        self.fusion_method = fusion_method
        self.num_classes = vit_config['num_classes']
        
        # Feature processing
        self.feature_proj = nn.Sequential(
            nn.Linear(feature_dim, self.embed_dim),
            nn.LayerNorm(self.embed_dim),
            nn.GELU(),
            nn.Dropout(0.1)
        )
        
        # Fusion layer
        if fusion_method == 'concat':
            self.classifier = nn.Linear(self.embed_dim * 2, self.num_classes)
        elif fusion_method == 'add':
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        elif fusion_method == 'attention':
            self.fusion_attn = nn.MultiheadAttention(self.embed_dim, num_heads=8, batch_first=True)
            self.classifier = nn.Linear(self.embed_dim, self.num_classes)
        else:
            raise ValueError(f"Unknown fusion method: {fusion_method}")
            
    def forward(self, 
                images: torch.Tensor, 
                features: torch.Tensor,
                packet_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with both images and features.
        
        Args:
            images: Input images (B, C, H, W)
            features: Additional features (B, feature_dim)
            packet_lengths: Optional packet lengths
            
        Returns:
            Logits (B, num_classes)
        """
        # Get ViT embeddings (without classification)
        B = images.shape[0]
        x = self.vit.patch_embed(images)
        cls_tokens = self.vit.cls_token.expand(B, -1, -1)
        x = torch.cat([cls_tokens, x], dim=1)
        
        if hasattr(self.vit.pos_embed, 'forward') and packet_lengths is not None:
            x = self.vit.pos_embed(x, packet_lengths)
        else:
            x = self.vit.pos_embed(x)
            
        x = self.vit.pos_dropout(x)
        
        for block in self.vit.blocks:
            x = block(x)
            
        x = self.vit.norm(x)
        vit_features = x[:, 0]  # CLS token
        
        # Process additional features
        processed_features = self.feature_proj(features)
        
        # Fusion
        if self.fusion_method == 'concat':
            fused = torch.cat([vit_features, processed_features], dim=1)
        elif self.fusion_method == 'add':
            fused = vit_features + processed_features
        elif self.fusion_method == 'attention':
            # Use attention to fuse features
            query = vit_features.unsqueeze(1)  # (B, 1, embed_dim)
            key_value = processed_features.unsqueeze(1)  # (B, 1, embed_dim)
            
            fused, _ = self.fusion_attn(query, key_value, key_value)
            fused = fused.squeeze(1)  # (B, embed_dim)
        
        # Classification
        logits = self.classifier(fused)
        
        return logits