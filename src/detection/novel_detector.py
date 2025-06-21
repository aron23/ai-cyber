"""
Novel Malware Detection System
Combines anomaly detection with few-shot learning for detecting and adapting to new malware families.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Tuple, Optional, Union
from collections import defaultdict
import copy


class NovelMalwareDetector:
    """
    Detects novel malware families and adapts to them using few-shot learning.
    """
    
    def __init__(self, base_model: nn.Module, few_shot_learner: Union['PrototypicalNetwork', 'MAML'],
                 confidence_threshold: float = 0.95, distance_threshold: Optional[float] = None,
                 device: str = 'cpu'):
        """
        Args:
            base_model: Pre-trained classifier for known malware families
            few_shot_learner: Few-shot learning model (Prototypical Networks or MAML)
            confidence_threshold: Threshold for confidence-based rejection
            distance_threshold: Threshold for distance-based rejection (if None, auto-computed)
            device: Device to run computations on
        """
        self.base_model = base_model.to(device)
        self.few_shot_learner = few_shot_learner
        self.confidence_threshold = confidence_threshold
        self.distance_threshold = distance_threshold
        self.device = device
        
        # Store known class information
        self.known_classes = []
        self.class_prototypes = {}
        self.class_stats = defaultdict(lambda: {'mean': None, 'std': None, 'count': 0})
        
        # Novel class detection history
        self.novel_detections = []
        self.adaptation_history = []
        
    def extract_features(self, data: torch.Tensor) -> torch.Tensor:
        """Extract features using the base model's encoder."""
        self.base_model.eval()
        with torch.no_grad():
            # If base_model has an encoder attribute, use it
            if hasattr(self.base_model, 'encoder'):
                features = self.base_model.encoder(data)
            else:
                # Otherwise, use intermediate layer output
                features = self._get_intermediate_features(data)
        return features
    
    def _get_intermediate_features(self, data: torch.Tensor) -> torch.Tensor:
        """Get features from intermediate layer of base model."""
        features = []
        
        def hook_fn(module, input, output):
            features.append(output)
        
        # Register hook on second-to-last layer
        layers = list(self.base_model.children())
        if len(layers) > 1:
            handle = layers[-2].register_forward_hook(hook_fn)
        else:
            handle = self.base_model.register_forward_hook(hook_fn)
            
        with torch.no_grad():
            _ = self.base_model(data)
            
        handle.remove()
        return features[0] if features else data
    
    def calibrate_on_known_classes(self, data_loader: torch.utils.data.DataLoader):
        """
        Calibrate the detector on known classes to establish baselines.
        
        Args:
            data_loader: DataLoader containing known class samples
        """
        print("Calibrating on known classes...")
        
        all_features = []
        all_labels = []
        all_predictions = []
        all_confidences = []
        
        self.base_model.eval()
        with torch.no_grad():
            for batch_data, batch_labels in data_loader:
                batch_data = batch_data.to(self.device)
                batch_labels = batch_labels.to(self.device)
                
                # Get features
                features = self.extract_features(batch_data)
                
                # Get predictions and confidence
                logits = self.base_model(batch_data)
                probs = F.softmax(logits, dim=1)
                confidences, predictions = torch.max(probs, dim=1)
                
                all_features.append(features.cpu())
                all_labels.append(batch_labels.cpu())
                all_predictions.append(predictions.cpu())
                all_confidences.append(confidences.cpu())
        
        # Concatenate all batches
        all_features = torch.cat(all_features)
        all_labels = torch.cat(all_labels)
        all_confidences = torch.cat(all_confidences)
        
        # Compute class prototypes and statistics
        unique_labels = torch.unique(all_labels)
        self.known_classes = unique_labels.tolist()
        
        for class_id in self.known_classes:
            class_mask = all_labels == class_id
            class_features = all_features[class_mask]
            class_confidences = all_confidences[class_mask]
            
            # Compute prototype
            self.class_prototypes[class_id] = class_features.mean(dim=0)
            
            # Compute confidence statistics
            self.class_stats[class_id]['mean'] = class_confidences.mean().item()
            self.class_stats[class_id]['std'] = class_confidences.std().item()
            self.class_stats[class_id]['count'] = len(class_features)
        
        # Auto-compute distance threshold if not provided
        if self.distance_threshold is None:
            self._compute_distance_threshold(all_features, all_labels)
            
        print(f"Calibration complete. Known classes: {len(self.known_classes)}")
    
    def _compute_distance_threshold(self, features: torch.Tensor, labels: torch.Tensor):
        """Compute distance threshold based on intra-class distances."""
        distances = []
        
        for class_id in self.known_classes:
            class_mask = labels == class_id
            class_features = features[class_mask]
            prototype = self.class_prototypes[class_id]
            
            # Compute distances to prototype
            class_distances = torch.cdist(class_features, prototype.unsqueeze(0)).squeeze()
            distances.extend(class_distances.tolist())
        
        # Set threshold as 95th percentile of distances
        self.distance_threshold = np.percentile(distances, 95)
        print(f"Auto-computed distance threshold: {self.distance_threshold:.4f}")
    
    def detect_novel_class(self, samples: torch.Tensor, return_details: bool = False) -> Union[torch.Tensor, Dict]:
        """
        Detect if samples belong to novel (unknown) classes.
        
        Args:
            samples: Input samples to classify
            return_details: Whether to return detailed detection information
            
        Returns:
            is_novel: Boolean tensor indicating novel samples
            details (optional): Dictionary with detection details
        """
        samples = samples.to(self.device)
        
        with torch.no_grad():
            # Get features
            features = self.extract_features(samples)
            
            # Get base model predictions
            logits = self.base_model(samples)
            probs = F.softmax(logits, dim=1)
            confidences, predictions = torch.max(probs, dim=1)
            
            # Method 1: Confidence-based detection
            low_confidence = confidences < self.confidence_threshold
            
            # Method 2: Distance-based detection
            min_distances = torch.full((len(samples),), float('inf')).to(self.device)
            nearest_class = torch.zeros(len(samples), dtype=torch.long).to(self.device)
            
            for class_id, prototype in self.class_prototypes.items():
                prototype = prototype.to(self.device)
                distances = torch.cdist(features, prototype.unsqueeze(0)).squeeze()
                mask = distances < min_distances
                min_distances[mask] = distances[mask]
                nearest_class[mask] = class_id
            
            far_from_prototypes = min_distances > self.distance_threshold
            
            # Combine detection methods
            is_novel = low_confidence | far_from_prototypes
            
        if return_details:
            details = {
                'is_novel': is_novel,
                'confidences': confidences,
                'predictions': predictions,
                'min_distances': min_distances,
                'nearest_class': nearest_class,
                'low_confidence': low_confidence,
                'far_from_prototypes': far_from_prototypes
            }
            return details
        
        return is_novel
    
    def adapt_to_novel_class(self, novel_samples: torch.Tensor, labels: torch.Tensor,
                            adaptation_method: str = 'prototypical') -> Dict:
        """
        Adapt to novel class using few-shot learning.
        
        Args:
            novel_samples: Samples from novel class
            labels: Labels for novel samples (can be pseudo-labels)
            adaptation_method: 'prototypical' or 'maml'
            
        Returns:
            adaptation_info: Dictionary with adaptation results
        """
        novel_samples = novel_samples.to(self.device)
        labels = labels.to(self.device)
        
        # Extract features
        features = self.extract_features(novel_samples)
        
        if adaptation_method == 'prototypical':
            # Compute prototype for novel class
            novel_prototype = features.mean(dim=0)
            
            # Assign new class ID
            new_class_id = max(self.known_classes) + 1 if self.known_classes else 0
            
            # Update prototypes
            self.class_prototypes[new_class_id] = novel_prototype
            self.known_classes.append(new_class_id)
            
            adaptation_info = {
                'method': 'prototypical',
                'new_class_id': new_class_id,
                'num_samples': len(novel_samples),
                'prototype_norm': torch.norm(novel_prototype).item()
            }
            
        elif adaptation_method == 'maml':
            # Use MAML for adaptation
            if hasattr(self.few_shot_learner, 'adapt'):
                adapted_model = self.few_shot_learner.adapt(novel_samples, labels)
                
                adaptation_info = {
                    'method': 'maml',
                    'num_samples': len(novel_samples),
                    'adapted_model': adapted_model
                }
            else:
                raise ValueError("Few-shot learner doesn't support MAML adaptation")
        
        else:
            raise ValueError(f"Unknown adaptation method: {adaptation_method}")
        
        # Record adaptation
        self.adaptation_history.append({
            'timestamp': pd.Timestamp.now(),
            'num_samples': len(novel_samples),
            'method': adaptation_method,
            **adaptation_info
        })
        
        return adaptation_info
    
    def classify_with_novel_detection(self, samples: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Classify samples with novel class detection.
        
        Args:
            samples: Input samples
            
        Returns:
            predictions: Class predictions (-1 for novel)
            confidences: Prediction confidences
            is_novel: Boolean mask for novel samples
        """
        detection_details = self.detect_novel_class(samples, return_details=True)
        
        predictions = detection_details['predictions']
        confidences = detection_details['confidences']
        is_novel = detection_details['is_novel']
        
        # Mark novel samples with -1
        predictions[is_novel] = -1
        
        return predictions, confidences, is_novel
    
    def update_statistics(self, features: torch.Tensor, labels: torch.Tensor, 
                         confidences: torch.Tensor):
        """Update class statistics with new data."""
        unique_labels = torch.unique(labels)
        
        for class_id in unique_labels:
            class_id = class_id.item()
            class_mask = labels == class_id
            
            if class_id in self.class_stats:
                # Update existing statistics
                old_mean = self.class_stats[class_id]['mean']
                old_count = self.class_stats[class_id]['count']
                new_count = class_mask.sum().item()
                
                # Update confidence statistics incrementally
                if old_mean is not None:
                    new_mean = ((old_mean * old_count + 
                               confidences[class_mask].mean().item() * new_count) / 
                               (old_count + new_count))
                    self.class_stats[class_id]['mean'] = new_mean
                else:
                    self.class_stats[class_id]['mean'] = confidences[class_mask].mean().item()
                
                self.class_stats[class_id]['count'] += new_count
    
    def get_detection_report(self) -> Dict:
        """Generate a report of novel detections and adaptations."""
        report = {
            'num_known_classes': len(self.known_classes),
            'num_adaptations': len(self.adaptation_history),
            'confidence_threshold': self.confidence_threshold,
            'distance_threshold': self.distance_threshold,
            'known_classes': self.known_classes,
            'adaptation_history': self.adaptation_history
        }
        
        return report 