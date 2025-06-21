"""
Model-Agnostic Meta-Learning (MAML) Implementation
Based on Finn et al., 2017: https://arxiv.org/abs/1703.03400
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from typing import List, Tuple, Dict, Optional, Callable
import copy
from collections import OrderedDict


class MAML:
    """
    Model-Agnostic Meta-Learning for few-shot learning.
    Learns model initialization that can quickly adapt to new tasks.
    """
    
    def __init__(self, model: nn.Module, inner_lr: float = 0.01, 
                 outer_lr: float = 0.001, first_order: bool = True,
                 inner_steps: int = 5, device: str = 'cpu'):
        """
        Args:
            model: Base model to be meta-learned
            inner_lr: Learning rate for inner loop (task adaptation)
            outer_lr: Learning rate for outer loop (meta-optimization)
            first_order: Use first-order approximation (FOMAML)
            inner_steps: Number of gradient steps in inner loop
            device: Device to run computations on
        """
        self.model = model.to(device)
        self.meta_model = copy.deepcopy(model).to(device)
        self.inner_lr = inner_lr
        self.outer_lr = outer_lr
        self.first_order = first_order
        self.inner_steps = inner_steps
        self.device = device
        
        # Meta-optimizer
        self.meta_optimizer = optim.Adam(
            self.meta_model.parameters(), 
            lr=outer_lr
        )
        
    def inner_loop(self, support_data: torch.Tensor, support_labels: torch.Tensor,
                   initial_params: Optional[OrderedDict] = None,
                   num_steps: Optional[int] = None) -> OrderedDict:
        """
        Perform task-specific adaptation (inner loop).
        
        Args:
            support_data: Support set data for the task
            support_labels: Support set labels
            initial_params: Initial parameters (if None, use meta_model params)
            num_steps: Number of gradient steps (if None, use self.inner_steps)
            
        Returns:
            adapted_params: Task-adapted parameters
        """
        if initial_params is None:
            initial_params = OrderedDict(self.meta_model.named_parameters())
        
        if num_steps is None:
            num_steps = self.inner_steps
            
        # Create a functional version of the model
        adapted_params = OrderedDict()
        for name, param in initial_params.items():
            adapted_params[name] = param.clone().requires_grad_(True)
            
        # Perform gradient steps
        for step in range(num_steps):
            # Forward pass with current parameters
            logits = self._functional_forward(
                support_data, adapted_params
            )
            
            # Compute loss
            inner_loss = F.cross_entropy(logits, support_labels)
            
            # Compute gradients
            grads = torch.autograd.grad(
                inner_loss, 
                adapted_params.values(),
                create_graph=not self.first_order
            )
            
            # Update parameters
            adapted_params = OrderedDict()
            for (name, param), grad in zip(initial_params.items(), grads):
                adapted_params[name] = param - self.inner_lr * grad
                
        return adapted_params
    
    def _functional_forward(self, x: torch.Tensor, 
                           params: OrderedDict) -> torch.Tensor:
        """
        Forward pass using provided parameters.
        
        Args:
            x: Input data
            params: Model parameters
            
        Returns:
            output: Model output
        """
        # This is a simplified version - actual implementation would need
        # to handle different layer types properly
        
        # Get model with temporary parameters
        model = copy.deepcopy(self.meta_model)
        for name, param in params.items():
            rsetattr(model, name, param)
            
        return model(x)
    
    def outer_loop(self, tasks: List[Tuple[torch.Tensor, torch.Tensor, 
                                          torch.Tensor, torch.Tensor]]) -> float:
        """
        Perform meta-optimization (outer loop).
        
        Args:
            tasks: List of (support_data, support_labels, query_data, query_labels)
            
        Returns:
            mean_loss: Average loss across tasks
        """
        self.meta_optimizer.zero_grad()
        
        task_losses = []
        
        for support_data, support_labels, query_data, query_labels in tasks:
            # Move data to device
            support_data = support_data.to(self.device)
            support_labels = support_labels.to(self.device)
            query_data = query_data.to(self.device)
            query_labels = query_labels.to(self.device)
            
            # Inner loop: adapt to task
            adapted_params = self.inner_loop(
                support_data, support_labels
            )
            
            # Evaluate on query set
            query_logits = self._functional_forward(
                query_data, adapted_params
            )
            task_loss = F.cross_entropy(query_logits, query_labels)
            
            task_losses.append(task_loss)
            
        # Compute meta-gradient
        meta_loss = torch.stack(task_losses).mean()
        meta_loss.backward()
        
        # Update meta-parameters
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def adapt(self, support_data: torch.Tensor, 
              support_labels: torch.Tensor) -> nn.Module:
        """
        Adapt model to new task and return adapted model.
        
        Args:
            support_data: Support examples for new task
            support_labels: Support labels
            
        Returns:
            adapted_model: Model adapted to the task
        """
        support_data = support_data.to(self.device)
        support_labels = support_labels.to(self.device)
        
        # Get adapted parameters
        adapted_params = self.inner_loop(support_data, support_labels)
        
        # Create model with adapted parameters
        adapted_model = copy.deepcopy(self.meta_model)
        for name, param in adapted_params.items():
            rsetattr(adapted_model, name, param.data)
            
        return adapted_model
    
    def save_checkpoint(self, path: str):
        """Save meta-learned model."""
        torch.save({
            'meta_model_state_dict': self.meta_model.state_dict(),
            'meta_optimizer_state_dict': self.meta_optimizer.state_dict(),
            'inner_lr': self.inner_lr,
            'outer_lr': self.outer_lr,
            'first_order': self.first_order,
            'inner_steps': self.inner_steps
        }, path)
        
    def load_checkpoint(self, path: str):
        """Load meta-learned model."""
        checkpoint = torch.load(path, map_location=self.device)
        self.meta_model.load_state_dict(checkpoint['meta_model_state_dict'])
        self.meta_optimizer.load_state_dict(checkpoint['meta_optimizer_state_dict'])
        self.inner_lr = checkpoint['inner_lr']
        self.outer_lr = checkpoint['outer_lr']
        self.first_order = checkpoint['first_order']
        self.inner_steps = checkpoint['inner_steps']


class MAMLTrainer:
    """
    Training wrapper for MAML with logging and validation.
    """
    
    def __init__(self, maml: MAML, task_sampler: Callable,
                 tasks_per_batch: int = 4):
        """
        Args:
            maml: MAML instance
            task_sampler: Function that returns tasks
            tasks_per_batch: Number of tasks per meta-batch
        """
        self.maml = maml
        self.task_sampler = task_sampler
        self.tasks_per_batch = tasks_per_batch
        self.train_losses = []
        self.val_losses = []
        
    def train_epoch(self, num_batches: int) -> float:
        """Train for one epoch."""
        epoch_losses = []
        
        for _ in range(num_batches):
            # Sample batch of tasks
            tasks = []
            for _ in range(self.tasks_per_batch):
                task = self.task_sampler()
                tasks.append(task)
                
            # Meta-update
            loss = self.maml.outer_loop(tasks)
            epoch_losses.append(loss)
            
        mean_loss = np.mean(epoch_losses)
        self.train_losses.append(mean_loss)
        return mean_loss
    
    def validate(self, num_tasks: int = 100) -> Tuple[float, float]:
        """
        Validate on new tasks.
        
        Returns:
            mean_loss: Average loss on validation tasks
            mean_accuracy: Average accuracy on validation tasks
        """
        losses = []
        accuracies = []
        
        for _ in range(num_tasks):
            # Sample validation task
            support_data, support_labels, query_data, query_labels = self.task_sampler()
            
            # Adapt to task
            adapted_model = self.maml.adapt(support_data, support_labels)
            
            # Evaluate
            with torch.no_grad():
                query_logits = adapted_model(query_data.to(self.maml.device))
                loss = F.cross_entropy(query_logits, query_labels.to(self.maml.device))
                
                predictions = torch.argmax(query_logits, dim=1)
                accuracy = (predictions == query_labels.to(self.maml.device)).float().mean()
                
            losses.append(loss.item())
            accuracies.append(accuracy.item())
            
        mean_loss = np.mean(losses)
        mean_accuracy = np.mean(accuracies)
        
        self.val_losses.append(mean_loss)
        
        return mean_loss, mean_accuracy


# Helper function for setting nested attributes
def rsetattr(obj, attr, val):
    """Recursively set attribute."""
    pre, _, post = attr.rpartition('.')
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

def rgetattr(obj, attr, *args):
    """Recursively get attribute."""
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)
    return functools.reduce(_getattr, [obj] + attr.split('.'))


# Import at the end to avoid circular dependency
import functools
import numpy as np 