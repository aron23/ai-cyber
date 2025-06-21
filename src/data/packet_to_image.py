"""
Packet to Image Encoding Module

This module provides various encoding strategies to convert network packet byte data
into image representations suitable for Vision Transformer models.
"""

import numpy as np
from typing import Tuple, List, Union, Optional
import torch
from torch import nn
import torch.nn.functional as F


class PacketImageEncoder:
    """
    Encoder for converting network packet bytes into images.
    
    Supports multiple encoding strategies:
    - Sequential: Direct byte-to-pixel mapping
    - RGB: 3-byte to RGB channel mapping  
    - Hilbert Curve: Preserves spatial locality
    - Spiral: Maintains sequential relationships
    - Block-based: Captures local patterns
    """
    
    def __init__(self, image_size: Tuple[int, int] = (64, 64), encoding_type: str = 'grayscale'):
        """
        Initialize the packet encoder.
        
        Args:
            image_size: Target image dimensions (height, width)
            encoding_type: Type of encoding ('grayscale' or 'rgb')
        """
        self.image_size = image_size
        self.encoding_type = encoding_type
        self.total_pixels = image_size[0] * image_size[1]
        
    def encode_sequential(self, packet_bytes: np.ndarray) -> np.ndarray:
        """
        Encode packet bytes as grayscale image using sequential filling.
        
        Args:
            packet_bytes: Array of byte values (0-255)
            
        Returns:
            2D numpy array representing the encoded image
        """
        # Handle padding/truncation
        if len(packet_bytes) > self.total_pixels:
            packet_bytes = packet_bytes[:self.total_pixels]
        elif len(packet_bytes) < self.total_pixels:
            packet_bytes = np.pad(packet_bytes, (0, self.total_pixels - len(packet_bytes)), 'constant')
        
        # Reshape to 2D image
        image = packet_bytes.reshape(self.image_size)
        return image.astype(np.uint8)
    
    def encode_hilbert_curve(self, packet_bytes: np.ndarray) -> np.ndarray:
        """
        Encode packet bytes using Hilbert curve to preserve locality.
        
        Args:
            packet_bytes: Array of byte values (0-255)
            
        Returns:
            2D numpy array with Hilbert curve mapping
        """
        # Generate Hilbert curve coordinates
        coords = self._generate_hilbert_curve(self.image_size[0])
        
        # Pad/truncate packet bytes
        if len(packet_bytes) > len(coords):
            packet_bytes = packet_bytes[:len(coords)]
        elif len(packet_bytes) < len(coords):
            packet_bytes = np.pad(packet_bytes, (0, len(coords) - len(packet_bytes)), 'constant')
        
        # Create image using Hilbert curve mapping
        image = np.zeros(self.image_size, dtype=np.uint8)
        for i, (x, y) in enumerate(coords[:len(packet_bytes)]):
            image[y, x] = packet_bytes[i]
        
        return image
    
    def encode_spiral(self, packet_bytes: np.ndarray) -> np.ndarray:
        """
        Encode packet bytes in a spiral pattern from center outward.
        
        Args:
            packet_bytes: Array of byte values (0-255)
            
        Returns:
            2D numpy array with spiral mapping
        """
        # Generate spiral coordinates
        coords = self._generate_spiral_coords(self.image_size)
        
        # Pad/truncate packet bytes
        if len(packet_bytes) > len(coords):
            packet_bytes = packet_bytes[:len(coords)]
        elif len(packet_bytes) < len(coords):
            packet_bytes = np.pad(packet_bytes, (0, len(coords) - len(packet_bytes)), 'constant')
        
        # Create image using spiral mapping
        image = np.zeros(self.image_size, dtype=np.uint8)
        for i, (x, y) in enumerate(coords[:len(packet_bytes)]):
            image[y, x] = packet_bytes[i]
        
        return image
    
    def encode_block_based(self, packet_bytes: np.ndarray, block_size: int = 8) -> np.ndarray:
        """
        Encode packet bytes in blocks to capture local patterns.
        
        Args:
            packet_bytes: Array of byte values (0-255)
            block_size: Size of each block
            
        Returns:
            2D numpy array with block-based layout
        """
        # Calculate number of blocks
        blocks_per_row = self.image_size[0] // block_size
        blocks_per_col = self.image_size[1] // block_size
        bytes_per_block = block_size * block_size
        
        # Pad packet bytes to fit complete blocks
        total_bytes_needed = blocks_per_row * blocks_per_col * bytes_per_block
        if len(packet_bytes) > total_bytes_needed:
            packet_bytes = packet_bytes[:total_bytes_needed]
        else:
            packet_bytes = np.pad(packet_bytes, (0, total_bytes_needed - len(packet_bytes)), 'constant')
        
        # Create image block by block
        image = np.zeros(self.image_size, dtype=np.uint8)
        byte_idx = 0
        
        for block_row in range(blocks_per_col):
            for block_col in range(blocks_per_row):
                for i in range(block_size):
                    for j in range(block_size):
                        if byte_idx < len(packet_bytes):
                            row = block_row * block_size + i
                            col = block_col * block_size + j
                            image[row, col] = packet_bytes[byte_idx]
                            byte_idx += 1
        
        return image
    
    def encode_batch(self, packet_batch: np.ndarray, method: str = 'sequential') -> np.ndarray:
        """
        Encode a batch of packets efficiently.
        
        Args:
            packet_batch: Array of shape (batch_size, packet_length)
            method: Encoding method to use
            
        Returns:
            Array of shape (batch_size, height, width)
        """
        batch_size = packet_batch.shape[0]
        encoded_batch = np.zeros((batch_size, *self.image_size), dtype=np.uint8)
        
        encoding_func = {
            'sequential': self.encode_sequential,
            'hilbert': self.encode_hilbert_curve,
            'spiral': self.encode_spiral,
            'block': self.encode_block_based
        }.get(method, self.encode_sequential)
        
        for i in range(batch_size):
            encoded_batch[i] = encoding_func(packet_batch[i])
        
        return encoded_batch
    
    def _generate_hilbert_curve(self, n: int) -> List[Tuple[int, int]]:
        """Generate 2D Hilbert curve coordinates."""
        def hilbert(x, y, xi, xj, yi, yj, n):
            if n <= 0:
                yield (x + (xi + yi) // 2, y + (xj + yj) // 2)
            else:
                yield from hilbert(x, y, yi // 2, yj // 2, xi // 2, xj // 2, n - 1)
                yield from hilbert(x + xi // 2, y + xj // 2, xi // 2, xj // 2, yi // 2, yj // 2, n - 1)
                yield from hilbert(x + xi // 2 + yi // 2, y + xj // 2 + yj // 2, xi // 2, xj // 2, yi // 2, yj // 2, n - 1)
                yield from hilbert(x + xi // 2 + yi, y + xj // 2 + yj, -yi // 2, -yj // 2, -xi // 2, -xj // 2, n - 1)
        
        # Calculate order needed for n x n grid
        order = int(np.log2(n))
        return list(hilbert(0, 0, n, 0, 0, n, order))
    
    def _generate_spiral_coords(self, shape: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Generate spiral coordinates from center outward."""
        coords = []
        h, w = shape
        cy, cx = h // 2, w // 2
        x, y = cx, cy
        dx, dy = 0, -1
        
        for _ in range(max(h, w) ** 2):
            if 0 <= x < w and 0 <= y < h:
                coords.append((x, y))
                
            if x == y or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
                dx, dy = -dy, dx
                
            x, y = x + dx, y + dy
            
        return coords[:h * w]


class TorchPacketEncoder(nn.Module):
    """
    PyTorch-based packet encoder for GPU acceleration and integration with neural networks.
    """
    
    def __init__(self, image_size: Tuple[int, int] = (64, 64), encoding_type: str = 'sequential'):
        super().__init__()
        self.image_size = image_size
        self.encoding_type = encoding_type
        self.total_pixels = image_size[0] * image_size[1]
        
        # Pre-compute encoding masks for efficiency
        if encoding_type == 'hilbert':
            self._setup_hilbert_mapping()
        elif encoding_type == 'spiral':
            self._setup_spiral_mapping()
    
    def forward(self, packet_batch: torch.Tensor) -> torch.Tensor:
        """
        Encode a batch of packets.
        
        Args:
            packet_batch: Tensor of shape (batch_size, packet_length)
            
        Returns:
            Tensor of shape (batch_size, 1, height, width) for grayscale
        """
        batch_size = packet_batch.shape[0]
        device = packet_batch.device
        
        # Pad or truncate to match total pixels
        if packet_batch.shape[1] > self.total_pixels:
            packet_batch = packet_batch[:, :self.total_pixels]
        elif packet_batch.shape[1] < self.total_pixels:
            padding = self.total_pixels - packet_batch.shape[1]
            packet_batch = F.pad(packet_batch, (0, padding), value=0)
        
        if self.encoding_type == 'sequential':
            # Simple reshape for sequential encoding
            images = packet_batch.view(batch_size, 1, *self.image_size)
        else:
            # Use pre-computed mapping for other encodings
            images = self._apply_mapping(packet_batch)
        
        # Normalize to [0, 1] range
        images = images.float() / 255.0
        
        return images
    
    def _setup_hilbert_mapping(self):
        """Pre-compute Hilbert curve mapping indices."""
        encoder = PacketImageEncoder(self.image_size)
        coords = encoder._generate_hilbert_curve(self.image_size[0])
        
        # Create mapping tensor
        self.register_buffer('mapping_indices', 
                           torch.tensor([y * self.image_size[1] + x for x, y in coords]))
    
    def _setup_spiral_mapping(self):
        """Pre-compute spiral mapping indices."""
        encoder = PacketImageEncoder(self.image_size)
        coords = encoder._generate_spiral_coords(self.image_size)
        
        # Create mapping tensor
        self.register_buffer('mapping_indices',
                           torch.tensor([y * self.image_size[1] + x for x, y in coords]))
    
    def _apply_mapping(self, packet_batch: torch.Tensor) -> torch.Tensor:
        """Apply pre-computed coordinate mapping."""
        batch_size = packet_batch.shape[0]
        device = packet_batch.device
        
        # Create output tensor
        images = torch.zeros(batch_size, self.total_pixels, device=device)
        
        # Apply mapping
        valid_indices = self.mapping_indices[:packet_batch.shape[1]]
        images[:, valid_indices] = packet_batch
        
        # Reshape to image format
        images = images.view(batch_size, 1, *self.image_size)
        
        return images


def create_patch_embeddings(packet_image: np.ndarray, patch_size: int = 16) -> np.ndarray:
    """
    Extract patches from packet image for Vision Transformer input.
    
    Args:
        packet_image: 2D array representing the packet image
        patch_size: Size of each patch
        
    Returns:
        Array of flattened patches
    """
    h, w = packet_image.shape
    assert h % patch_size == 0 and w % patch_size == 0, \
        f"Image size {(h, w)} must be divisible by patch_size {patch_size}"
    
    # Calculate number of patches
    n_patches_h = h // patch_size
    n_patches_w = w // patch_size
    n_patches = n_patches_h * n_patches_w
    
    # Extract patches
    patches = np.zeros((n_patches, patch_size * patch_size))
    
    patch_idx = 0
    for i in range(0, h, patch_size):
        for j in range(0, w, patch_size):
            patch = packet_image[i:i+patch_size, j:j+patch_size]
            patches[patch_idx] = patch.flatten()
            patch_idx += 1
    
    return patches


# Utility functions for handling variable-length packets
def pad_packet(packet: np.ndarray, target_length: int, strategy: str = 'zero') -> np.ndarray:
    """
    Pad packet to target length.
    
    Args:
        packet: Original packet bytes
        target_length: Desired length after padding
        strategy: Padding strategy ('zero', 'random', 'cyclic')
        
    Returns:
        Padded packet array
    """
    if len(packet) >= target_length:
        return packet[:target_length]
    
    padding_needed = target_length - len(packet)
    
    if strategy == 'zero':
        return np.pad(packet, (0, padding_needed), 'constant', constant_values=0)
    elif strategy == 'random':
        padding = np.random.randint(0, 256, padding_needed, dtype=np.uint8)
        return np.concatenate([packet, padding])
    elif strategy == 'cyclic':
        if len(packet) == 0:
            return np.zeros(target_length, dtype=np.uint8)
        cycles = padding_needed // len(packet) + 1
        cyclic_pad = np.tile(packet, cycles)[:padding_needed]
        return np.concatenate([packet, cyclic_pad])
    else:
        raise ValueError(f"Unknown padding strategy: {strategy}")


def calculate_optimal_image_size(packet_lengths: List[int], 
                               coverage_threshold: float = 0.95) -> Tuple[int, int]:
    """
    Calculate optimal image size based on packet length distribution.
    
    Args:
        packet_lengths: List of packet lengths in the dataset
        coverage_threshold: Percentage of packets to cover without truncation
        
    Returns:
        Tuple of (height, width) for optimal image size
    """
    # Calculate percentile length
    target_length = int(np.percentile(packet_lengths, coverage_threshold * 100))
    
    # Find closest square image size
    sqrt_size = int(np.sqrt(target_length))
    
    # Common image sizes for Vision Transformers
    common_sizes = [32, 64, 96, 128, 160, 192, 224, 256]
    
    # Find the smallest common size that can accommodate the target length
    for size in common_sizes:
        if size * size >= target_length:
            return (size, size)
    
    # If no common size fits, use custom size
    return (sqrt_size + 1, sqrt_size + 1)