"""
Few-Shot Performance Analysis
Comprehensive benchmarking and analysis tools for few-shot learning methods.
"""

import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Optional, Callable
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from scipy import stats
import time


class FewShotBenchmark:
    """
    Comprehensive benchmarking suite for few-shot learning methods.
    """
    
    def __init__(self, methods: Dict[str, object], dataset, device='cpu'):
        """
        Args:
            methods: Dictionary of method_name -> few_shot_learner
            dataset: Dataset to evaluate on
            device: Device for computation
        """
        self.methods = methods
        self.dataset = dataset
        self.device = device
        self.results = {}
        
    def benchmark_k_shot(self, n_way: int = 5, k_shots: List[int] = [1, 5, 10, 20],
                        n_query: int = 15, num_episodes: int = 100) -> pd.DataFrame:
        """
        Benchmark performance across different k-shot values.
        
        Returns:
            DataFrame with results for each method and k-shot value
        """
        results = []
        
        for method_name, learner in self.methods.items():
            print(f"\nBenchmarking {method_name}...")
            
            for k_shot in k_shots:
                print(f"  Testing {k_shot}-shot...")
                
                # Run episodes
                accuracies = []
                times = []
                
                for _ in tqdm(range(num_episodes), leave=False):
                    # Sample episode
                    episode = self._sample_episode(n_way, k_shot, n_query)
                    
                    # Time the adaptation and inference
                    start_time = time.time()
                    accuracy = self._evaluate_episode(learner, episode)
                    end_time = time.time()
                    
                    accuracies.append(accuracy)
                    times.append(end_time - start_time)
                
                # Store results
                results.append({
                    'method': method_name,
                    'k_shot': k_shot,
                    'n_way': n_way,
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'mean_time': np.mean(times),
                    'std_time': np.std(times),
                    'accuracies': accuracies
                })
        
        return pd.DataFrame(results)
    
    def learning_curve_analysis(self, n_way: int = 5, max_k: int = 50,
                               step_size: int = 5, num_episodes: int = 50) -> pd.DataFrame:
        """
        Analyze learning curves as k increases.
        """
        k_values = list(range(1, max_k + 1, step_size))
        results = []
        
        for method_name, learner in self.methods.items():
            print(f"\nAnalyzing learning curve for {method_name}...")
            
            for k in tqdm(k_values):
                accuracies = []
                
                for _ in range(num_episodes):
                    episode = self._sample_episode(n_way, k, n_query=15)
                    accuracy = self._evaluate_episode(learner, episode)
                    accuracies.append(accuracy)
                
                results.append({
                    'method': method_name,
                    'k': k,
                    'mean_accuracy': np.mean(accuracies),
                    'std_accuracy': np.std(accuracies),
                    'confidence_95': 1.96 * np.std(accuracies) / np.sqrt(len(accuracies))
                })
        
        return pd.DataFrame(results)
    
    def ablation_study(self, base_config: Dict, ablations: Dict[str, List],
                      num_episodes: int = 100) -> pd.DataFrame:
        """
        Perform ablation study on different components.
        
        Args:
            base_config: Base configuration
            ablations: Dict of parameter_name -> list of values to test
            num_episodes: Number of episodes per configuration
        """
        results = []
        
        for ablation_param, values in ablations.items():
            print(f"\nAblating {ablation_param}...")
            
            for value in values:
                # Create config with ablation
                config = base_config.copy()
                config[ablation_param] = value
                
                # Run evaluation
                accuracies = []
                for _ in tqdm(range(num_episodes), leave=False):
                    episode = self._sample_episode(**config)
                    
                    # Evaluate each method
                    for method_name, learner in self.methods.items():
                        accuracy = self._evaluate_episode(learner, episode)
                        accuracies.append({
                            'method': method_name,
                            'ablation_param': ablation_param,
                            'ablation_value': str(value),
                            'accuracy': accuracy,
                            **config
                        })
                
                results.extend(accuracies)
        
        return pd.DataFrame(results)
    
    def statistical_analysis(self, results_df: pd.DataFrame, 
                           metric: str = 'mean_accuracy') -> Dict:
        """
        Perform statistical analysis on results.
        
        Returns:
            Dictionary with statistical test results
        """
        analysis = {}
        
        # Pairwise comparisons between methods
        methods = results_df['method'].unique()
        
        if len(methods) > 1:
            # Perform pairwise t-tests
            pairwise_tests = []
            
            for i, method1 in enumerate(methods):
                for method2 in methods[i+1:]:
                    data1 = results_df[results_df['method'] == method1][metric]
                    data2 = results_df[results_df['method'] == method2][metric]
                    
                    # T-test
                    t_stat, p_value = stats.ttest_ind(data1, data2)
                    
                    # Effect size (Cohen's d)
                    pooled_std = np.sqrt((np.std(data1)**2 + np.std(data2)**2) / 2)
                    cohens_d = (np.mean(data1) - np.mean(data2)) / pooled_std
                    
                    pairwise_tests.append({
                        'method1': method1,
                        'method2': method2,
                        't_statistic': t_stat,
                        'p_value': p_value,
                        'cohens_d': cohens_d,
                        'significant': p_value < 0.05
                    })
            
            analysis['pairwise_tests'] = pd.DataFrame(pairwise_tests)
            
            # ANOVA if more than 2 methods
            if len(methods) > 2:
                groups = [results_df[results_df['method'] == m][metric] for m in methods]
                f_stat, p_value = stats.f_oneway(*groups)
                analysis['anova'] = {
                    'f_statistic': f_stat,
                    'p_value': p_value,
                    'significant': p_value < 0.05
                }
        
        return analysis
    
    def visualize_results(self, results_df: pd.DataFrame, save_path: Optional[str] = None):
        """
        Create comprehensive visualizations of results.
        """
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        
        # 1. K-shot performance comparison
        if 'k_shot' in results_df.columns:
            ax = axes[0, 0]
            for method in results_df['method'].unique():
                method_data = results_df[results_df['method'] == method]
                ax.errorbar(method_data['k_shot'], 
                           method_data['mean_accuracy'],
                           yerr=method_data['std_accuracy'],
                           marker='o', label=method, capsize=5)
            ax.set_xlabel('K-shot')
            ax.set_ylabel('Accuracy')
            ax.set_title('Performance vs K-shot')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        # 2. Time efficiency
        if 'mean_time' in results_df.columns:
            ax = axes[0, 1]
            methods = results_df['method'].unique()
            times = [results_df[results_df['method'] == m]['mean_time'].mean() 
                    for m in methods]
            ax.bar(methods, times)
            ax.set_ylabel('Average Time (s)')
            ax.set_title('Inference Speed Comparison')
            ax.grid(True, alpha=0.3, axis='y')
        
        # 3. Accuracy distribution
        if 'accuracies' in results_df.columns:
            ax = axes[1, 0]
            all_accs = []
            labels = []
            for _, row in results_df.iterrows():
                if isinstance(row['accuracies'], list):
                    all_accs.extend(row['accuracies'])
                    labels.extend([row['method']] * len(row['accuracies']))
            
            if all_accs:
                df_accs = pd.DataFrame({'accuracy': all_accs, 'method': labels})
                sns.violinplot(data=df_accs, x='method', y='accuracy', ax=ax)
                ax.set_title('Accuracy Distribution')
                ax.set_ylim(0, 1)
        
        # 4. Learning curve
        if 'k' in results_df.columns:
            ax = axes[1, 1]
            for method in results_df['method'].unique():
                method_data = results_df[results_df['method'] == method]
                ax.plot(method_data['k'], method_data['mean_accuracy'], 
                       marker='o', label=method)
                # Add confidence bands
                ax.fill_between(method_data['k'],
                              method_data['mean_accuracy'] - method_data['confidence_95'],
                              method_data['mean_accuracy'] + method_data['confidence_95'],
                              alpha=0.2)
            ax.set_xlabel('Number of Support Examples (K)')
            ax.set_ylabel('Accuracy')
            ax.set_title('Learning Curves')
            ax.legend()
            ax.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        plt.show()
    
    def _sample_episode(self, n_way: int, k_shot: int, n_query: int, **kwargs):
        """Sample an episode from the dataset."""
        # This is a placeholder - actual implementation depends on dataset format
        # In practice, would use EpisodeSampler
        pass
    
    def _evaluate_episode(self, learner, episode) -> float:
        """Evaluate a learner on an episode."""
        # This is a placeholder - actual implementation depends on learner type
        # Would call learner's forward/adapt methods
        pass


class AttentionAnalyzer:
    """
    Analyze attention patterns in few-shot learning models.
    """
    
    def __init__(self, model: nn.Module):
        self.model = model
        self.attention_maps = []
        
    def extract_attention_patterns(self, support_data: torch.Tensor,
                                 query_data: torch.Tensor) -> Dict:
        """
        Extract and analyze attention patterns.
        
        Returns:
            Dictionary with attention analysis results
        """
        # Register hooks to capture attention weights
        hooks = []
        attention_weights = []
        
        def hook_fn(module, input, output):
            if hasattr(output, 'attention_weights'):
                attention_weights.append(output.attention_weights)
        
        # Register hooks on attention layers
        for module in self.model.modules():
            if 'attention' in module.__class__.__name__.lower():
                hooks.append(module.register_forward_hook(hook_fn))
        
        # Forward pass
        with torch.no_grad():
            _ = self.model(torch.cat([support_data, query_data]))
        
        # Remove hooks
        for hook in hooks:
            hook.remove()
        
        # Analyze patterns
        analysis = {
            'num_attention_layers': len(attention_weights),
            'attention_stats': []
        }
        
        for i, attn in enumerate(attention_weights):
            stats = {
                'layer': i,
                'mean_attention': attn.mean().item(),
                'std_attention': attn.std().item(),
                'entropy': self._compute_entropy(attn)
            }
            analysis['attention_stats'].append(stats)
        
        return analysis
    
    def _compute_entropy(self, attention_weights: torch.Tensor) -> float:
        """Compute entropy of attention distribution."""
        # Normalize to probabilities
        probs = F.softmax(attention_weights.flatten(), dim=0)
        # Compute entropy
        entropy = -(probs * torch.log(probs + 1e-8)).sum()
        return entropy.item()
    
    def visualize_attention_evolution(self, episodes: List, save_path: Optional[str] = None):
        """
        Visualize how attention patterns evolve across episodes.
        """
        # Placeholder for attention visualization
        pass 