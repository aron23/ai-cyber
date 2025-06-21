"""
Episode sampling for few-shot learning tasks.
Handles creation of N-way K-shot episodes from datasets.
"""

import torch
import numpy as np
from torch.utils.data import Dataset, Sampler
from typing import List, Tuple, Dict, Optional, Iterator
import random


class Episode:
    """Container for a single few-shot learning episode."""
    
    def __init__(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                 query_data: torch.Tensor, query_labels: torch.Tensor,
                 class_mapping: Optional[Dict[int, int]] = None):
        """
        Args:
            support_data: Support set examples
            support_labels: Support set labels (0 to N-1)
            query_data: Query set examples
            query_labels: Query set labels (0 to N-1)
            class_mapping: Mapping from episode labels to original labels
        """
        self.support_data = support_data
        self.support_labels = support_labels
        self.query_data = query_data
        self.query_labels = query_labels
        self.class_mapping = class_mapping
        
    def to(self, device):
        """Move episode to device."""
        self.support_data = self.support_data.to(device)
        self.support_labels = self.support_labels.to(device)
        self.query_data = self.query_data.to(device)
        self.query_labels = self.query_labels.to(device)
        return self
    
    @property
    def n_way(self) -> int:
        return len(torch.unique(self.support_labels))
    
    @property
    def k_shot(self) -> int:
        return len(self.support_data) // self.n_way
    
    @property
    def n_query(self) -> int:
        return len(self.query_data) // self.n_way


class EpisodeSampler:
    """
    Samples episodes from a dataset for few-shot learning.
    """
    
    def __init__(self, dataset: Dataset, n_way: int = 5, k_shot: int = 5, 
                 n_query: int = 15, min_samples_per_class: Optional[int] = None):
        """
        Args:
            dataset: PyTorch dataset with __getitem__ returning (data, label)
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
            min_samples_per_class: Minimum samples required per class
        """
        self.dataset = dataset
        self.n_way = n_way
        self.k_shot = k_shot
        self.n_query = n_query
        self.min_samples = min_samples_per_class or (k_shot + n_query)
        
        # Build class-to-indices mapping
        self.class_indices = self._build_class_indices()
        
        # Filter classes with enough samples
        self.valid_classes = [
            cls for cls, indices in self.class_indices.items()
            if len(indices) >= self.min_samples
        ]
        
        if len(self.valid_classes) < n_way:
            raise ValueError(
                f"Not enough classes with {self.min_samples} samples. "
                f"Found {len(self.valid_classes)} classes, need {n_way}."
            )
            
    def _build_class_indices(self) -> Dict[int, List[int]]:
        """Build mapping from class labels to dataset indices."""
        class_indices = {}
        
        for idx in range(len(self.dataset)):
            _, label = self.dataset[idx]
            if label not in class_indices:
                class_indices[label] = []
            class_indices[label].append(idx)
            
        return class_indices
    
    def sample_episode(self) -> Episode:
        """Sample a single N-way K-shot episode."""
        # Randomly select N classes
        selected_classes = random.sample(self.valid_classes, self.n_way)
        
        support_data = []
        support_labels = []
        query_data = []
        query_labels = []
        class_mapping = {}
        
        for episode_label, true_label in enumerate(selected_classes):
            # Get indices for this class
            class_idx = self.class_indices[true_label]
            
            # Sample K+Q examples
            selected_idx = random.sample(class_idx, self.k_shot + self.n_query)
            
            # Split into support and query
            support_idx = selected_idx[:self.k_shot]
            query_idx = selected_idx[self.k_shot:]
            
            # Collect data
            for idx in support_idx:
                data, _ = self.dataset[idx]
                support_data.append(data)
                support_labels.append(episode_label)
                
            for idx in query_idx:
                data, _ = self.dataset[idx]
                query_data.append(data)
                query_labels.append(episode_label)
                
            class_mapping[episode_label] = true_label
            
        # Convert to tensors
        support_data = torch.stack(support_data)
        support_labels = torch.tensor(support_labels, dtype=torch.long)
        query_data = torch.stack(query_data)
        query_labels = torch.tensor(query_labels, dtype=torch.long)
        
        return Episode(support_data, support_labels, query_data, query_labels, class_mapping)
    
    def generate_episodes(self, num_episodes: int) -> List[Episode]:
        """Generate multiple episodes."""
        return [self.sample_episode() for _ in range(num_episodes)]


class EpisodicBatchSampler(Sampler):
    """
    Batch sampler that creates episodes for DataLoader.
    """
    
    def __init__(self, dataset: Dataset, n_way: int, k_shot: int, 
                 n_query: int, episodes_per_epoch: int):
        """
        Args:
            dataset: Dataset to sample from
            n_way: Number of classes per episode
            k_shot: Number of support examples per class
            n_query: Number of query examples per class
            episodes_per_epoch: Number of episodes to generate per epoch
        """
        self.sampler = EpisodeSampler(dataset, n_way, k_shot, n_query)
        self.episodes_per_epoch = episodes_per_epoch
        
    def __iter__(self) -> Iterator[List[int]]:
        """Yield batch indices for episodes."""
        for _ in range(self.episodes_per_epoch):
            episode = self.sampler.sample_episode()
            
            # For DataLoader compatibility, we return indices
            # This is a placeholder - actual implementation would need
            # to handle episode structure differently
            indices = list(range(len(episode.support_data) + len(episode.query_data)))
            yield indices
            
    def __len__(self) -> int:
        return self.episodes_per_epoch


class TaskSampler:
    """
    Samples tasks for meta-learning (e.g., MAML).
    Each task is a classification problem with its own train/test split.
    """
    
    def __init__(self, dataset: Dataset, n_way: int = 5, 
                 k_shot_train: int = 5, k_shot_test: int = 15):
        """
        Args:
            dataset: Dataset to sample from
            n_way: Number of classes per task
            k_shot_train: Number of training examples per class
            k_shot_test: Number of test examples per class
        """
        self.sampler = EpisodeSampler(
            dataset, n_way, k_shot_train, k_shot_test
        )
        
    def sample_task(self) -> Tuple[Episode, Episode]:
        """
        Sample a task with train and test episodes.
        
        Returns:
            train_episode: Episode for inner loop training
            test_episode: Episode for outer loop evaluation
        """
        # For simplicity, we use support as train and query as test
        episode = self.sampler.sample_episode()
        
        train_episode = Episode(
            episode.support_data, episode.support_labels,
            episode.support_data, episode.support_labels,  # Duplicate for compatibility
            episode.class_mapping
        )
        
        test_episode = Episode(
            episode.support_data, episode.support_labels,  # Same support
            episode.query_data, episode.query_labels,
            episode.class_mapping
        )
        
        return train_episode, test_episode
    
    def sample_batch(self, batch_size: int) -> List[Tuple[Episode, Episode]]:
        """Sample a batch of tasks."""
        return [self.sample_task() for _ in range(batch_size)] 