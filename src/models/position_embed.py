"""
Position Embedding Module for Vision Transformers

This module provides various position embedding strategies optimized for
network packet image data, including byte-order aware embeddings.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Tuple


class PositionEmbedding(nn.Module):
    """
    Standard position embeddings for Vision Transformers.
    Supports learnable, sinusoidal, and 2D position encodings.
    """
    
    def __init__(self, 
                 num_patches: int,
                 embed_dim: int,
                 embedding_type: str = 'learnable',
                 cls_token: bool = True):
        """
        Args:
            num_patches: Total number of patches
            embed_dim: Embedding dimension
            embedding_type: Type of position embedding ('learnable', 'sinusoidal', '2d')
            cls_token: Whether to include position for CLS token
        """
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.embedding_type = embedding_type
        self.cls_token = cls_token
        
        # Add 1 for CLS token if needed
        num_positions = num_patches + 1 if cls_token else num_patches
        
        if embedding_type == 'learnable':
            self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
            nn.init.trunc_normal_(self.pos_embed, std=0.02)
            
        elif embedding_type == 'sinusoidal':
            self.register_buffer('pos_embed', 
                               self._create_sinusoidal_embedding(num_positions))
            
        elif embedding_type == '2d':
            # Assume square grid of patches
            grid_size = int(np.sqrt(num_patches))
            assert grid_size * grid_size == num_patches, \
                "2D position embedding requires square number of patches"
            self.register_buffer('pos_embed', 
                               self._create_2d_embedding(grid_size, cls_token))
            
        else:
            raise ValueError(f"Unknown embedding type: {embedding_type}")
            
    def _create_sinusoidal_embedding(self, num_positions: int) -> torch.Tensor:
        """Create sinusoidal position embeddings."""
        position = torch.arange(num_positions).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, self.embed_dim, 2) * 
                           -(np.log(10000.0) / self.embed_dim))
        
        pos_embed = torch.zeros(1, num_positions, self.embed_dim)
        pos_embed[0, :, 0::2] = torch.sin(position * div_term)
        pos_embed[0, :, 1::2] = torch.cos(position * div_term)
        
        return pos_embed
    
    def _create_2d_embedding(self, grid_size: int, cls_token: bool) -> torch.Tensor:
        """Create 2D position embeddings considering spatial structure."""
        num_positions = grid_size * grid_size + (1 if cls_token else 0)
        pos_embed = torch.zeros(1, num_positions, self.embed_dim)
        
        # CLS token position
        if cls_token:
            # CLS token gets special encoding
            pos_embed[0, 0, :] = 0  # Will be learned separately
            start_idx = 1
        else:
            start_idx = 0
        
        # Create 2D positional grid
        y_embed = torch.arange(grid_size).unsqueeze(1).repeat(1, grid_size).flatten()
        x_embed = torch.arange(grid_size).repeat(grid_size)
        
        # Encode positions
        half_dim = self.embed_dim // 2
        div_term = torch.exp(torch.arange(0, half_dim, 2) * 
                           -(np.log(10000.0) / half_dim))
        
        # Y-axis encoding
        pos_embed[0, start_idx:, 0:half_dim:2] = torch.sin(y_embed.unsqueeze(1) * div_term)
        pos_embed[0, start_idx:, 1:half_dim:2] = torch.cos(y_embed.unsqueeze(1) * div_term)
        
        # X-axis encoding  
        pos_embed[0, start_idx:, half_dim::2] = torch.sin(x_embed.unsqueeze(1) * div_term)
        pos_embed[0, start_idx:, half_dim+1::2] = torch.cos(x_embed.unsqueeze(1) * div_term)
        
        return pos_embed
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add position embeddings to input."""
        return x + self.pos_embed[:, :x.size(1)]


class ByteOrderAwarePositionEmbedding(nn.Module):
    """
    Position embedding that considers the original byte order in packets.
    Incorporates knowledge about packet structure (headers vs payload).
    """
    
    def __init__(self,
                 num_patches: int,
                 embed_dim: int,
                 patch_size: int = 16,
                 image_size: int = 224,
                 cls_token: bool = True):
        super().__init__()
        self.num_patches = num_patches
        self.embed_dim = embed_dim
        self.patch_size = patch_size
        self.image_size = image_size
        self.cls_token = cls_token
        
        # Learnable embeddings for different packet regions
        self.header_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.payload_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.padding_embed = nn.Parameter(torch.zeros(1, 1, embed_dim))
        
        # Position-dependent scaling
        self.position_scale = nn.Parameter(torch.ones(1, 1, embed_dim))
        
        # Base position embedding
        num_positions = num_patches + (1 if cls_token else 0)
        self.base_pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        
        # Initialize parameters
        nn.init.trunc_normal_(self.header_embed, std=0.02)
        nn.init.trunc_normal_(self.payload_embed, std=0.02)
        nn.init.trunc_normal_(self.padding_embed, std=0.02)
        nn.init.trunc_normal_(self.base_pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor, 
                packet_lengths: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Add position embeddings with byte-order awareness.
        
        Args:
            x: Patch embeddings (B, num_patches, embed_dim)
            packet_lengths: Original packet lengths for each sample (B,)
        """
        B, N, D = x.shape
        
        # Start with base position embedding
        pos_embed = self.base_pos_embed[:, :N].expand(B, -1, -1).clone()
        
        # Skip CLS token if present
        start_idx = 1 if self.cls_token and N > self.num_patches else 0
        
        # Add byte-order aware components
        patches_per_row = self.image_size // self.patch_size
        
        for i in range(start_idx, N):
            patch_idx = i - start_idx
            
            # Calculate which bytes this patch represents
            row = patch_idx // patches_per_row
            col = patch_idx % patches_per_row
            byte_start = row * self.image_size + col * self.patch_size
            byte_end = byte_start + self.patch_size * self.patch_size
            
            # Determine patch type based on byte position
            # Typical packet structure: Ethernet (14) + IP (20) + TCP/UDP (20/8) = ~42-54 bytes
            if byte_start < 54:
                # Header region
                pos_embed[:, i] += self.header_embed
            elif packet_lengths is not None:
                # Check if this patch contains actual packet data or padding
                for b in range(B):
                    if byte_end > packet_lengths[b]:
                        # This patch contains padding
                        pos_embed[b, i] += self.padding_embed
                    else:
                        # Payload region
                        pos_embed[b, i] += self.payload_embed
            else:
                # Default to payload if no length info
                pos_embed[:, i] += self.payload_embed
                
        return x + pos_embed * self.position_scale


class RelativePositionEmbedding(nn.Module):
    """
    Relative position embedding for self-attention.
    Encodes relative distances between patches rather than absolute positions.
    """
    
    def __init__(self,
                 num_patches: int,
                 num_heads: int,
                 max_relative_position: Optional[int] = None):
        super().__init__()
        self.num_patches = num_patches
        self.num_heads = num_heads
        
        # Assume square grid
        grid_size = int(np.sqrt(num_patches))
        if max_relative_position is None:
            max_relative_position = grid_size - 1
        
        self.max_relative_position = max_relative_position
        
        # Relative position bias table
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * max_relative_position + 1) * (2 * max_relative_position + 1), 
                       num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)
        
        # Compute relative position index
        coords_h = torch.arange(grid_size)
        coords_w = torch.arange(grid_size)
        coords = torch.stack(torch.meshgrid([coords_h, coords_w]))  # 2, grid_size, grid_size
        coords_flatten = torch.flatten(coords, 1)  # 2, num_patches
        
        relative_coords = coords_flatten[:, :, None] - coords_flatten[:, None, :]  # 2, N, N
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # N, N, 2
        
        # Shift to start from 0
        relative_coords[:, :, 0] += max_relative_position
        relative_coords[:, :, 1] += max_relative_position
        relative_coords[:, :, 0] *= 2 * max_relative_position + 1
        
        relative_position_index = relative_coords.sum(-1)  # N, N
        self.register_buffer("relative_position_index", relative_position_index)
        
    def forward(self) -> torch.Tensor:
        """
        Returns:
            Relative position bias of shape (num_patches, num_patches, num_heads)
        """
        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.view(-1)
        ].view(self.num_patches, self.num_patches, -1)  # N, N, num_heads
        
        return relative_position_bias.permute(2, 0, 1)  # num_heads, N, N


class InterpolatedPositionEmbedding(nn.Module):
    """
    Position embedding that can be interpolated to different resolutions.
    Useful for handling variable image sizes or fine-tuning at different resolutions.
    """
    
    def __init__(self,
                 embed_dim: int,
                 default_grid_size: int = 14,  # 224/16
                 cls_token: bool = True):
        super().__init__()
        self.embed_dim = embed_dim
        self.default_grid_size = default_grid_size
        self.cls_token = cls_token
        
        # Learnable position embedding at default resolution
        default_num_patches = default_grid_size ** 2
        num_positions = default_num_patches + (1 if cls_token else 0)
        
        self.pos_embed = nn.Parameter(torch.zeros(1, num_positions, embed_dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        
    def forward(self, x: torch.Tensor, grid_size: Optional[int] = None) -> torch.Tensor:
        """
        Add interpolated position embeddings.
        
        Args:
            x: Input tensor (B, N, D)
            grid_size: Target grid size (if different from default)
        """
        B, N, D = x.shape
        
        if grid_size is None:
            # Use default position embedding
            return x + self.pos_embed[:, :N]
        
        # Interpolate position embedding to target size
        if self.cls_token:
            cls_pos = self.pos_embed[:, :1]  # Keep CLS token position
            patch_pos = self.pos_embed[:, 1:]  # Patch positions to interpolate
            N_patches = N - 1
        else:
            patch_pos = self.pos_embed
            N_patches = N
            
        # Reshape to 2D grid
        patch_pos = patch_pos.reshape(1, self.default_grid_size, 
                                    self.default_grid_size, D)
        patch_pos = patch_pos.permute(0, 3, 1, 2)  # (1, D, H, W)
        
        # Interpolate
        patch_pos = torch.nn.functional.interpolate(
            patch_pos, size=(grid_size, grid_size), 
            mode='bicubic', align_corners=False
        )
        
        # Reshape back
        patch_pos = patch_pos.permute(0, 2, 3, 1).reshape(1, grid_size * grid_size, D)
        
        # Combine with CLS token if needed
        if self.cls_token:
            pos_embed = torch.cat([cls_pos, patch_pos], dim=1)
        else:
            pos_embed = patch_pos
            
        return x + pos_embed


def create_position_embedding(
    num_patches: int,
    embed_dim: int,
    embedding_type: str = 'learnable',
    **kwargs
) -> nn.Module:
    """
    Factory function to create position embedding modules.
    
    Args:
        num_patches: Number of patches
        embed_dim: Embedding dimension
        embedding_type: Type of embedding ('learnable', 'sinusoidal', '2d', 
                       'byte_aware', 'relative', 'interpolated')
        **kwargs: Additional arguments for specific embedding types
        
    Returns:
        Position embedding module
    """
    cls_token = kwargs.get('cls_token', True)
    
    if embedding_type in ['learnable', 'sinusoidal', '2d']:
        return PositionEmbedding(num_patches, embed_dim, embedding_type, cls_token)
    
    elif embedding_type == 'byte_aware':
        patch_size = kwargs.get('patch_size', 16)
        image_size = kwargs.get('image_size', 224)
        return ByteOrderAwarePositionEmbedding(
            num_patches, embed_dim, patch_size, image_size, cls_token
        )
    
    elif embedding_type == 'relative':
        num_heads = kwargs.get('num_heads', 12)
        max_relative_position = kwargs.get('max_relative_position', None)
        return RelativePositionEmbedding(num_patches, num_heads, max_relative_position)
    
    elif embedding_type == 'interpolated':
        default_grid_size = kwargs.get('default_grid_size', 14)
        return InterpolatedPositionEmbedding(embed_dim, default_grid_size, cls_token)
    
    else:
        raise ValueError(f"Unknown position embedding type: {embedding_type}")