"""
Few-shot learning implementations for malware detection.
"""

from .prototypical_net import PrototypicalNetwork, LearnedDistanceMetric
from .episode_sampler import Episode, EpisodeSampler, EpisodicBatchSampler, TaskSampler
from .maml import MAML, MAMLTrainer

__all__ = [
    'PrototypicalNetwork',
    'LearnedDistanceMetric',
    'Episode',
    'EpisodeSampler',
    'EpisodicBatchSampler',
    'TaskSampler',
    'MAML',
    'MAMLTrainer'
] 