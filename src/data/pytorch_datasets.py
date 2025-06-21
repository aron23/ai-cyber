"""
PyTorch Dataset Classes for Payload-Byte Data

This module provides PyTorch Dataset implementations for efficient
data loading and preprocessing in machine learning pipelines.
"""

import torch
from torch.utils.data import Dataset, DataLoader, Sampler
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Union, Dict, Any, Callable
import h5py
import logging
from collections import defaultdict
import random


logger = logging.getLogger(__name__)


class PayloadByteDataset(Dataset):
    """
    PyTorch Dataset for Payload-Byte packet data.
    
    Supports multiple data formats, transformations, and efficient loading.
    """
    
    def __init__(self,
                 data_path: Union[str, Path],
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 mode: str = 'train',
                 cache_size: int = 1000,
                 preload: bool = False):
        """
        Initialize the dataset.
        
        Args:
            data_path: Path to data directory or file
            transform: Optional transform to apply to packets
            target_transform: Optional transform to apply to labels
            mode: Dataset mode ('train', 'val', 'test')
            cache_size: Number of samples to cache in memory
            preload: Whether to preload all data into memory
        """
        self.data_path = Path(data_path)
        self.transform = transform
        self.target_transform = target_transform
        self.mode = mode
        self.cache_size = cache_size
        self.preload = preload
        
        # Initialize data storage
        self.packets = []
        self.labels = []
        self.metadata = {}
        
        # Cache for recently accessed items
        self.cache = {}
        self.cache_order = []
        
        # Load data
        self._load_data()
        
    def _load_data(self):
        """Load data based on file format and mode"""
        if self.data_path.suffix == '.h5':
            self._load_h5_data()
        elif self.data_path.suffix == '.npz':
            self._load_npz_data()
        elif self.data_path.is_dir():
            self._load_directory_data()
        else:
            raise ValueError(f"Unsupported data format: {self.data_path}")
            
        logger.info(f"Loaded {len(self)} samples for {self.mode} mode")
        
    def _load_h5_data(self):
        """Load data from HDF5 file"""
        with h5py.File(self.data_path, 'r') as f:
            # Load based on mode
            if self.mode in f:
                group = f[self.mode]
                
                if self.preload:
                    self.packets = group['packets'][:]
                    self.labels = group['labels'][:]
                else:
                    # Keep references for lazy loading
                    self.h5_file = self.data_path
                    self.packets = None
                    self.labels = None
                    self._length = len(group['packets'])
                    
                # Load metadata if available
                if 'metadata' in group.attrs:
                    self.metadata = dict(group.attrs)
            else:
                raise ValueError(f"Mode '{self.mode}' not found in HDF5 file")
                
    def _load_npz_data(self):
        """Load data from NumPy compressed file"""
        data = np.load(self.data_path, allow_pickle=True)
        
        prefix = f"{self.mode}_"
        self.packets = data[f"{prefix}packets"]
        self.labels = data[f"{prefix}labels"]
        
        # Load metadata if available
        if f"{prefix}metadata" in data:
            self.metadata = data[f"{prefix}metadata"].item()
            
    def _load_directory_data(self):
        """Load data from directory structure"""
        mode_dir = self.data_path / self.mode
        
        if not mode_dir.exists():
            raise ValueError(f"Mode directory not found: {mode_dir}")
            
        # Load packets and labels from separate files
        benign_dir = mode_dir / 'benign'
        malicious_dir = mode_dir / 'malicious'
        
        # Load benign packets
        if benign_dir.exists():
            for packet_file in benign_dir.glob('*.npy'):
                packet = np.load(packet_file)
                self.packets.append(packet)
                self.labels.append(0)
                
        # Load malicious packets
        if malicious_dir.exists():
            for packet_file in malicious_dir.glob('*.npy'):
                packet = np.load(packet_file)
                self.packets.append(packet)
                self.labels.append(1)
                
        # Convert to arrays if preloading
        if self.preload:
            self.packets = np.array(self.packets)
            self.labels = np.array(self.labels)
            
    def __len__(self) -> int:
        """Return the number of samples"""
        if hasattr(self, '_length'):
            return self._length
        return len(self.labels)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """
        Get a sample from the dataset.
        
        Args:
            idx: Sample index
            
        Returns:
            Tuple of (packet_data, label)
        """
        # Check cache first
        if idx in self.cache:
            return self.cache[idx]
            
        # Load data
        if self.packets is None:
            # Lazy loading from HDF5
            with h5py.File(self.h5_file, 'r') as f:
                packet = f[self.mode]['packets'][idx]
                label = f[self.mode]['labels'][idx]
        else:
            packet = self.packets[idx]
            label = self.labels[idx]
            
        # Apply transforms
        if self.transform:
            packet = self.transform(packet)
        else:
            packet = torch.from_numpy(packet).float()
            
        if self.target_transform:
            label = self.target_transform(label)
        else:
            label = int(label)
            
        # Update cache
        self._update_cache(idx, (packet, label))
        
        return packet, label
    
    def _update_cache(self, idx: int, data: Tuple):
        """Update the LRU cache"""
        if len(self.cache) >= self.cache_size:
            # Remove oldest item
            oldest_idx = self.cache_order.pop(0)
            del self.cache[oldest_idx]
            
        self.cache[idx] = data
        self.cache_order.append(idx)
        
    def get_label_distribution(self) -> Dict[int, int]:
        """Get the distribution of labels in the dataset"""
        if self.labels is None:
            # Load all labels for statistics
            with h5py.File(self.h5_file, 'r') as f:
                labels = f[self.mode]['labels'][:]
        else:
            labels = self.labels
            
        unique, counts = np.unique(labels, return_counts=True)
        return dict(zip(unique.tolist(), counts.tolist()))
    
    def get_metadata(self) -> Dict[str, Any]:
        """Get dataset metadata"""
        metadata = self.metadata.copy()
        metadata['num_samples'] = len(self)
        metadata['label_distribution'] = self.get_label_distribution()
        return metadata


class PacketImageDataset(PayloadByteDataset):
    """
    Dataset that converts packets to images on-the-fly.
    
    Extends PayloadByteDataset with image conversion capabilities.
    """
    
    def __init__(self,
                 data_path: Union[str, Path],
                 image_converter: Optional[Any] = None,
                 image_size: Tuple[int, int] = (224, 224),
                 encoding_strategy: str = 'sequential',
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None,
                 mode: str = 'train',
                 cache_size: int = 1000,
                 preload: bool = False):
        """
        Initialize the packet image dataset.
        
        Args:
            data_path: Path to data
            image_converter: PacketToImageConverter instance
            image_size: Target image size
            encoding_strategy: Image encoding strategy
            transform: Additional transforms for images
            target_transform: Transform for labels
            mode: Dataset mode
            cache_size: Cache size
            preload: Whether to preload data
        """
        # Initialize parent class
        super().__init__(
            data_path=data_path,
            transform=None,  # We'll handle transforms differently
            target_transform=target_transform,
            mode=mode,
            cache_size=cache_size,
            preload=preload
        )
        
        self.image_converter = image_converter
        self.image_size = image_size
        self.encoding_strategy = encoding_strategy
        self.image_transform = transform
        
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample and convert to image"""
        # Get raw packet data
        if self.packets is None:
            with h5py.File(self.h5_file, 'r') as f:
                packet = f[self.mode]['packets'][idx]
                label = f[self.mode]['labels'][idx]
        else:
            packet = self.packets[idx]
            label = self.labels[idx]
            
        # Convert to image
        if self.image_converter:
            image = self.image_converter.packet_to_image(
                packet,
                image_size=self.image_size,
                strategy=self.encoding_strategy,
                normalize=True
            )
        else:
            # Simple conversion if no converter provided
            image = self._simple_packet_to_image(packet)
            
        # Apply additional transforms
        if self.image_transform:
            image = self.image_transform(image)
            
        # Transform label
        if self.target_transform:
            label = self.target_transform(label)
        else:
            label = int(label)
            
        return image, label
    
    def _simple_packet_to_image(self, packet: np.ndarray) -> torch.Tensor:
        """Simple packet to image conversion"""
        height, width = self.image_size
        total_pixels = height * width
        
        # Pad or truncate
        if len(packet) < total_pixels:
            padded = np.zeros(total_pixels, dtype=np.float32)
            padded[:len(packet)] = packet / 255.0
            packet = padded
        else:
            packet = packet[:total_pixels].astype(np.float32) / 255.0
            
        # Reshape and convert to tensor
        image = packet.reshape(1, height, width)
        return torch.from_numpy(image).float()


class StreamingDataset(Dataset):
    """
    Dataset for streaming large packet captures that don't fit in memory.
    """
    
    def __init__(self,
                 data_files: List[Union[str, Path]],
                 chunk_size: int = 1000,
                 transform: Optional[Callable] = None,
                 target_transform: Optional[Callable] = None):
        """
        Initialize streaming dataset.
        
        Args:
            data_files: List of data files to stream
            chunk_size: Number of samples to load at once
            transform: Transform for packets
            target_transform: Transform for labels
        """
        self.data_files = [Path(f) for f in data_files]
        self.chunk_size = chunk_size
        self.transform = transform
        self.target_transform = target_transform
        
        # Build index of file locations
        self._build_index()
        
    def _build_index(self):
        """Build an index of sample locations across files"""
        self.file_index = []
        self.cumulative_sizes = [0]
        
        for file_path in self.data_files:
            if file_path.suffix == '.h5':
                with h5py.File(file_path, 'r') as f:
                    size = len(f['packets'])
                    self.file_index.extend([(file_path, i) for i in range(size)])
                    self.cumulative_sizes.append(self.cumulative_sizes[-1] + size)
                    
        logger.info(f"Built index for {len(self.file_index)} samples across {len(self.data_files)} files")
        
    def __len__(self) -> int:
        """Return total number of samples"""
        return len(self.file_index)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a sample from the appropriate file"""
        file_path, local_idx = self.file_index[idx]
        
        with h5py.File(file_path, 'r') as f:
            packet = f['packets'][local_idx]
            label = f['labels'][local_idx]
            
        # Apply transforms
        if self.transform:
            packet = self.transform(packet)
        else:
            packet = torch.from_numpy(packet).float()
            
        if self.target_transform:
            label = self.target_transform(label)
        else:
            label = int(label)
            
        return packet, label


class BalancedBatchSampler(Sampler):
    """
    Sampler that ensures balanced classes in each batch.
    """
    
    def __init__(self, dataset: Dataset, batch_size: int, drop_last: bool = False):
        """
        Initialize balanced batch sampler.
        
        Args:
            dataset: Dataset to sample from
            batch_size: Batch size
            drop_last: Whether to drop last incomplete batch
        """
        self.dataset = dataset
        self.batch_size = batch_size
        self.drop_last = drop_last
        
        # Get label distribution
        self.label_to_indices = defaultdict(list)
        for idx in range(len(dataset)):
            _, label = dataset[idx]
            self.label_to_indices[label].append(idx)
            
        self.labels = list(self.label_to_indices.keys())
        self.n_classes = len(self.labels)
        self.samples_per_class = batch_size // self.n_classes
        
    def __iter__(self):
        """Generate balanced batches"""
        # Shuffle indices for each class
        for label in self.labels:
            random.shuffle(self.label_to_indices[label])
            
        # Create batches
        batch = []
        class_counters = {label: 0 for label in self.labels}
        
        while True:
            # Add samples from each class
            for label in self.labels:
                indices = self.label_to_indices[label]
                counter = class_counters[label]
                
                for _ in range(self.samples_per_class):
                    if counter < len(indices):
                        batch.append(indices[counter])
                        counter += 1
                    else:
                        # Reshuffle and reset counter
                        random.shuffle(indices)
                        counter = 0
                        batch.append(indices[counter])
                        counter += 1
                        
                class_counters[label] = counter
                
            # Yield batch if full
            if len(batch) >= self.batch_size:
                yield batch[:self.batch_size]
                batch = batch[self.batch_size:]
                
            # Check if we've used all samples
            if all(class_counters[label] >= len(self.label_to_indices[label]) 
                   for label in self.labels):
                if batch and not self.drop_last:
                    yield batch
                break
                
    def __len__(self) -> int:
        """Return number of batches"""
        total_samples = sum(len(indices) for indices in self.label_to_indices.values())
        if self.drop_last:
            return total_samples // self.batch_size
        else:
            return (total_samples + self.batch_size - 1) // self.batch_size


def create_data_loaders(data_path: Union[str, Path],
                       batch_size: int = 32,
                       num_workers: int = 4,
                       transform: Optional[Callable] = None,
                       target_transform: Optional[Callable] = None,
                       balanced_sampling: bool = False,
                       pin_memory: bool = True,
                       persistent_workers: bool = True) -> Dict[str, DataLoader]:
    """
    Create DataLoaders for train, validation, and test sets.
    
    Args:
        data_path: Path to data
        batch_size: Batch size
        num_workers: Number of worker processes
        transform: Transform for data
        target_transform: Transform for labels
        balanced_sampling: Whether to use balanced sampling
        pin_memory: Whether to pin memory for GPU
        persistent_workers: Whether to keep workers alive
        
    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders
    """
    loaders = {}
    
    for mode in ['train', 'val', 'test']:
        # Create dataset
        dataset = PayloadByteDataset(
            data_path=data_path,
            transform=transform,
            target_transform=target_transform,
            mode=mode,
            preload=False
        )
        
        # Create sampler if needed
        sampler = None
        if mode == 'train' and balanced_sampling:
            sampler = BalancedBatchSampler(dataset, batch_size)
            # When using custom sampler, set batch_size=1 and sampler handles batching
            loader_batch_size = 1
            shuffle = False
        else:
            loader_batch_size = batch_size
            shuffle = (mode == 'train')
            
        # Create DataLoader
        loaders[mode] = DataLoader(
            dataset,
            batch_size=loader_batch_size,
            shuffle=shuffle,
            sampler=sampler,
            num_workers=num_workers,
            pin_memory=pin_memory,
            persistent_workers=persistent_workers and num_workers > 0,
            drop_last=(mode == 'train')
        )
        
        logger.info(f"Created {mode} DataLoader with {len(dataset)} samples")
        
    return loaders


def create_streaming_dataloader(data_files: List[Union[str, Path]],
                               batch_size: int = 32,
                               num_workers: int = 4,
                               transform: Optional[Callable] = None,
                               shuffle: bool = True) -> DataLoader:
    """
    Create a DataLoader for streaming large datasets.
    
    Args:
        data_files: List of data files
        batch_size: Batch size
        num_workers: Number of workers
        transform: Data transform
        shuffle: Whether to shuffle data
        
    Returns:
        DataLoader instance
    """
    dataset = StreamingDataset(
        data_files=data_files,
        transform=transform
    )
    
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=True
    )