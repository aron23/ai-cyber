"""
Prototypical Networks for Few-Shot Learning
Based on Snell et al., 2017: https://arxiv.org/abs/1703.05175
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple, Optional


class PrototypicalNetwork(nn.Module):
    """
    Prototypical Networks for few-shot learning.
    Uses prototype representations of classes for classification.
    """
    
    def __init__(self, encoder: nn.Module, distance_metric: str = 'euclidean'):
        """
        Args:
            encoder: Feature extraction network (e.g., ViT backbone)
            distance_metric: Distance metric to use ('euclidean' or 'cosine')
        """
        super().__init__()
        self.encoder = encoder
        self.distance_metric = distance_metric
        
    def compute_prototypes(self, support_features: torch.Tensor, 
                          support_labels: torch.Tensor) -> torch.Tensor:
        """
        Calculate class prototypes from support set.
        
        Args:
            support_features: Features of support examples [n_support, feature_dim]
            support_labels: Labels of support examples [n_support]
            
        Returns:
            prototypes: Class prototypes [n_classes, feature_dim]
        """
        n_classes = len(torch.unique(support_labels))
        feature_dim = support_features.shape[1]
        
        prototypes = torch.zeros(n_classes, feature_dim).to(support_features.device)
        
        for class_idx in range(n_classes):
            # Get all support examples for this class
            class_mask = support_labels == class_idx
            class_features = support_features[class_mask]
            
            # Compute prototype as mean of class features
            prototypes[class_idx] = class_features.mean(dim=0)
            
        return prototypes
    
    def compute_distances(self, query_features: torch.Tensor, 
                         prototypes: torch.Tensor) -> torch.Tensor:
        """
        Compute distances between query features and prototypes.
        
        Args:
            query_features: Features of query examples [n_query, feature_dim]
            prototypes: Class prototypes [n_classes, feature_dim]
            
        Returns:
            distances: Distance matrix [n_query, n_classes]
        """
        if self.distance_metric == 'euclidean':
            # Euclidean distance: ||q - p||^2
            distances = torch.cdist(query_features, prototypes, p=2)
            
        elif self.distance_metric == 'cosine':
            # Cosine similarity (converted to distance)
            query_norm = F.normalize(query_features, p=2, dim=1)
            proto_norm = F.normalize(prototypes, p=2, dim=1)
            similarities = torch.mm(query_norm, proto_norm.t())
            distances = 1 - similarities
            
        else:
            raise ValueError(f"Unknown distance metric: {self.distance_metric}")
            
        return distances
    
    def classify_query(self, query_features: torch.Tensor, 
                      prototypes: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Classify query examples based on distances to prototypes.
        
        Args:
            query_features: Features of query examples [n_query, feature_dim]
            prototypes: Class prototypes [n_classes, feature_dim]
            
        Returns:
            predictions: Predicted class indices [n_query]
            confidence: Confidence scores [n_query, n_classes]
        """
        # Compute distances
        distances = self.compute_distances(query_features, prototypes)
        
        # Convert distances to log probabilities
        log_probs = F.log_softmax(-distances, dim=1)
        confidence = torch.exp(log_probs)
        
        # Get predictions
        predictions = torch.argmin(distances, dim=1)
        
        return predictions, confidence
    
    def forward(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                query_data: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Forward pass for an episode.
        
        Args:
            support_data: Support set data [n_support, ...]
            support_labels: Support set labels [n_support]
            query_data: Query set data [n_query, ...]
            
        Returns:
            log_probs: Log probabilities for query set [n_query, n_classes]
            predictions: Predicted classes [n_query]
            prototypes: Computed prototypes [n_classes, feature_dim]
        """
        # Extract features
        support_features = self.encoder(support_data)
        query_features = self.encoder(query_data)
        
        # Compute prototypes
        prototypes = self.compute_prototypes(support_features, support_labels)
        
        # Compute distances and log probabilities
        distances = self.compute_distances(query_features, prototypes)
        log_probs = F.log_softmax(-distances, dim=1)
        
        # Get predictions
        predictions = torch.argmin(distances, dim=1)
        
        return log_probs, predictions, prototypes
    
    def adapt(self, support_data: torch.Tensor, support_labels: torch.Tensor) -> dict:
        """
        Adapt to new classes (compute prototypes).
        
        Args:
            support_data: Support examples for new classes
            support_labels: Labels for support examples
            
        Returns:
            adaptation_state: Dictionary containing prototypes
        """
        with torch.no_grad():
            support_features = self.encoder(support_data)
            prototypes = self.compute_prototypes(support_features, support_labels)
            
        return {'prototypes': prototypes}
    
    def predict_with_adaptation(self, query_data: torch.Tensor, 
                               adaptation_state: dict) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Make predictions using pre-computed adaptation state.
        
        Args:
            query_data: Query examples to classify
            adaptation_state: Dictionary containing prototypes
            
        Returns:
            predictions: Predicted classes
            confidence: Confidence scores
        """
        with torch.no_grad():
            query_features = self.encoder(query_data)
            prototypes = adaptation_state['prototypes']
            predictions, confidence = self.classify_query(query_features, prototypes)
            
        return predictions, confidence


class LearnedDistanceMetric(nn.Module):
    """
    Learnable distance metric for Prototypical Networks.
    """
    
    def __init__(self, feature_dim: int, hidden_dim: Optional[int] = None):
        """
        Args:
            feature_dim: Dimension of feature vectors
            hidden_dim: Hidden dimension for metric network
        """
        super().__init__()
        hidden_dim = hidden_dim or feature_dim
        
        self.metric_net = nn.Sequential(
            nn.Linear(feature_dim * 2, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, 1)
        )
        
    def forward(self, query_features: torch.Tensor, prototypes: torch.Tensor) -> torch.Tensor:
        """
        Compute learned distances between queries and prototypes.
        
        Args:
            query_features: [n_query, feature_dim]
            prototypes: [n_classes, feature_dim]
            
        Returns:
            distances: [n_query, n_classes]
        """
        n_query = query_features.shape[0]
        n_classes = prototypes.shape[0]
        
        # Expand dimensions for pairwise computation
        query_exp = query_features.unsqueeze(1).expand(n_query, n_classes, -1)
        proto_exp = prototypes.unsqueeze(0).expand(n_query, n_classes, -1)
        
        # Concatenate query-prototype pairs
        pairs = torch.cat([query_exp, proto_exp], dim=-1)
        
        # Compute distances
        distances = self.metric_net(pairs).squeeze(-1)
        
        return distances 