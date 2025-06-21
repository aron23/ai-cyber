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
from torch.utils.data import Dataset, DataLoader
from enum import Enum
import time


class EncodingStrategy(Enum):
    """Enumeration of available encoding strategies."""
    SEQUENTIAL = "sequential"
    HILBERT = "hilbert"
    SPIRAL = "spiral"
    ZIGZAG = "zigzag"
    BLOCK = "block"


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


class PacketToImageConverter:
    """
    Converter class for transforming network packets into image representations.
    Provides the API expected by the notebook.
    """
    
    def __init__(self, default_image_size: Tuple[int, int] = (224, 224)):
        """
        Initialize the packet to image converter.
        
        Args:
            default_image_size: Default target image dimensions (height, width)
        """
        self.default_image_size = default_image_size
        self.encoder = PacketImageEncoder(default_image_size)
        
    def packet_to_image(self, 
                       packet_bytes: Union[np.ndarray, List[int]],
                       image_size: Optional[Tuple[int, int]] = None,
                       strategy: EncodingStrategy = EncodingStrategy.SEQUENTIAL,
                       normalize: bool = True) -> torch.Tensor:
        """
        Convert a single packet to image representation.
        
        Args:
            packet_bytes: Packet byte data
            image_size: Target image size (uses default if None)
            strategy: Encoding strategy to use
            normalize: Whether to normalize to [0, 1] range
            
        Returns:
            Tensor of shape (1, H, W) representing the packet image
        """
        if image_size is None:
            image_size = self.default_image_size
            
        # Ensure numpy array
        if not isinstance(packet_bytes, np.ndarray):
            packet_bytes = np.array(packet_bytes, dtype=np.uint8)
            
        # Update encoder image size if needed
        if image_size != self.encoder.image_size:
            self.encoder = PacketImageEncoder(image_size)
            
        # Encode based on strategy
        if strategy == EncodingStrategy.SEQUENTIAL:
            image = self.encoder.encode_sequential(packet_bytes)
        elif strategy == EncodingStrategy.HILBERT:
            image = self.encoder.encode_hilbert_curve(packet_bytes)
        elif strategy == EncodingStrategy.SPIRAL:
            image = self.encoder.encode_spiral(packet_bytes)
        elif strategy == EncodingStrategy.BLOCK:
            image = self.encoder.encode_block_based(packet_bytes)
        elif strategy == EncodingStrategy.ZIGZAG:
            image = self.encode_zigzag(packet_bytes, image_size)
        else:
            raise ValueError(f"Unknown encoding strategy: {strategy}")
            
        # Convert to tensor
        tensor = torch.from_numpy(image).float().unsqueeze(0)  # Add channel dimension
        
        # Normalize if requested
        if normalize:
            tensor = tensor / 255.0
            
        return tensor
    
    def packets_to_images(self,
                         packet_list: List[Union[np.ndarray, List[int]]],
                         image_size: Optional[Tuple[int, int]] = None,
                         strategy: EncodingStrategy = EncodingStrategy.SEQUENTIAL,
                         normalize: bool = True) -> torch.Tensor:
        """
        Convert a batch of packets to image representations.
        
        Args:
            packet_list: List of packet byte arrays
            image_size: Target image size (uses default if None)
            strategy: Encoding strategy to use
            normalize: Whether to normalize to [0, 1] range
            
        Returns:
            Tensor of shape (N, 1, H, W) representing the packet images
        """
        if image_size is None:
            image_size = self.default_image_size
            
        images = []
        for packet in packet_list:
            image_tensor = self.packet_to_image(packet, image_size, strategy, normalize)
            images.append(image_tensor)
            
        return torch.stack(images)
    
    def encode_zigzag(self, packet_bytes: np.ndarray, image_size: Tuple[int, int]) -> np.ndarray:
        """
        Encode packet bytes using zigzag pattern (like JPEG).
        
        Args:
            packet_bytes: Array of byte values (0-255)
            image_size: Target image dimensions
            
        Returns:
            2D numpy array with zigzag mapping
        """
        h, w = image_size
        total_pixels = h * w
        
        # Pad/truncate packet bytes
        if len(packet_bytes) > total_pixels:
            packet_bytes = packet_bytes[:total_pixels]
        elif len(packet_bytes) < total_pixels:
            packet_bytes = np.pad(packet_bytes, (0, total_pixels - len(packet_bytes)), 'constant')
        
        # Generate zigzag coordinates
        coords = self._generate_zigzag_coords(image_size)
        
        # Create image using zigzag mapping
        image = np.zeros(image_size, dtype=np.uint8)
        for i, (x, y) in enumerate(coords[:len(packet_bytes)]):
            image[y, x] = packet_bytes[i]
        
        return image
    
    def _generate_zigzag_coords(self, image_size: Tuple[int, int]) -> List[Tuple[int, int]]:
        """Generate zigzag coordinates for JPEG-like encoding."""
        h, w = image_size
        coords = []
        
        for i in range(h):
            if i % 2 == 0:
                # Left to right
                for j in range(w):
                    coords.append((j, i))
            else:
                # Right to left
                for j in range(w-1, -1, -1):
                    coords.append((j, i))
        
        return coords
    
    def benchmark_encoding_speeds(self, 
                                 packet_sizes: List[int] = [256, 512, 1024, 2048, 4096],
                                 num_trials: int = 100) -> dict:
        """
        Benchmark encoding speeds for different strategies and packet sizes.
        
        Args:
            packet_sizes: List of packet sizes to test
            num_trials: Number of trials per configuration
            
        Returns:
            Dictionary of results
        """
        results = {}
        
        for size in packet_sizes:
            size_key = f"packet_{size}"
            results[size_key] = {}
            
            # Generate test packet
            test_packet = np.random.randint(0, 256, size, dtype=np.uint8)
            
            for strategy in EncodingStrategy:
                times = []
                
                for _ in range(num_trials):
                    start_time = time.time()
                    _ = self.packet_to_image(test_packet, strategy=strategy)
                    end_time = time.time()
                    times.append(end_time - start_time)
                
                avg_time = np.mean(times)
                results[size_key][strategy.value] = avg_time
        
        return results
    
    def optimize_memory_usage(self, batch_size: int) -> dict:
        """
        Calculate memory usage statistics for a given batch size.
        
        Args:
            batch_size: Number of packets in a batch
            
        Returns:
            Dictionary with memory statistics
        """
        h, w = self.default_image_size
        bytes_per_image = h * w
        batch_memory_bytes = batch_size * bytes_per_image * 4  # float32
        batch_memory_mb = batch_memory_bytes / (1024 * 1024)
        
        return {
            'batch_size': batch_size,
            'bytes_per_image': bytes_per_image,
            'batch_memory_bytes': batch_memory_bytes,
            'batch_memory_mb': batch_memory_mb
        }


class PacketImageDataset(Dataset):
    """PyTorch Dataset for packet images."""
    
    def __init__(self, 
                 packets: List[np.ndarray], 
                 labels: List[int],
                 image_size: Tuple[int, int] = (224, 224),
                 strategy: EncodingStrategy = EncodingStrategy.SEQUENTIAL,
                 normalize: bool = True):
        """
        Initialize dataset.
        
        Args:
            packets: List of packet byte arrays
            labels: List of corresponding labels
            image_size: Target image size
            strategy: Encoding strategy
            normalize: Whether to normalize images
        """
        self.packets = packets
        self.labels = labels
        self.converter = PacketToImageConverter(image_size)
        self.strategy = strategy
        self.normalize = normalize
        
    def __len__(self):
        return len(self.packets)
    
    def __getitem__(self, idx):
        packet = self.packets[idx]
        label = self.labels[idx]
        
        # Convert packet to image
        image = self.converter.packet_to_image(
            packet, 
            strategy=self.strategy, 
            normalize=self.normalize
        )
        
        return image, torch.tensor(label, dtype=torch.long)


def create_batch_generator(packets: List[np.ndarray],
                          labels: List[int],
                          batch_size: int = 32,
                          image_size: Tuple[int, int] = (224, 224),
                          strategy: EncodingStrategy = EncodingStrategy.SEQUENTIAL,
                          shuffle: bool = True,
                          num_workers: int = 0) -> DataLoader:
    """
    Create a PyTorch DataLoader for packet images.
    
    Args:
        packets: List of packet byte arrays
        labels: List of corresponding labels
        batch_size: Batch size for DataLoader
        image_size: Target image size
        strategy: Encoding strategy
        shuffle: Whether to shuffle data
        num_workers: Number of worker processes
        
    Returns:
        PyTorch DataLoader
    """
    dataset = PacketImageDataset(
        packets=packets,
        labels=labels,
        image_size=image_size,
        strategy=strategy,
        normalize=True
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available()
    )