"""
Pipeline Optimization Module

Provides tools for profiling, optimizing, and scaling the data pipeline
with support for both CPU and GPU environments.
"""

import time
import psutil
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging
from dataclasses import dataclass
import torch
from pathlib import Path
import json


logger = logging.getLogger(__name__)


@dataclass
class PerformanceMetrics:
    """Container for performance metrics"""
    throughput_samples_per_sec: float
    latency_ms: float
    memory_usage_mb: float
    cpu_usage_percent: float
    gpu_usage_percent: Optional[float] = None
    gpu_memory_mb: Optional[float] = None


class PipelineOptimizer:
    """
    Optimizer for data pipeline performance.
    
    Handles both CPU-only and GPU-accelerated environments.
    """
    
    def __init__(self, use_gpu: bool = False):
        """
        Initialize the pipeline optimizer.
        
        Args:
            use_gpu: Whether to use GPU acceleration (if available)
        """
        self.use_gpu = use_gpu and torch.cuda.is_available()
        
        if self.use_gpu:
            logger.info(f"GPU acceleration enabled: {torch.cuda.get_device_name()}")
        else:
            logger.info("Running in CPU-only mode")
            
        self.process = psutil.Process()
        self.cpu_count = mp.cpu_count()
        
    def profile_pipeline(self,
                        pipeline_func: Callable,
                        data_samples: List[Any],
                        batch_size: int = 32,
                        num_iterations: int = 10) -> PerformanceMetrics:
        """
        Profile a data pipeline function.
        
        Args:
            pipeline_func: Function to profile
            data_samples: Sample data for testing
            batch_size: Batch size to test
            num_iterations: Number of iterations for averaging
            
        Returns:
            Performance metrics
        """
        # Warm up
        _ = pipeline_func(data_samples[:batch_size])
        
        # Collect metrics
        throughputs = []
        latencies = []
        memory_usages = []
        cpu_usages = []
        gpu_usages = []
        gpu_memories = []
        
        for i in range(num_iterations):
            # Select batch
            start_idx = (i * batch_size) % (len(data_samples) - batch_size)
            batch = data_samples[start_idx:start_idx + batch_size]
            
            # Measure performance
            start_time = time.time()
            start_memory = self.process.memory_info().rss / 1024 / 1024
            
            # Get initial CPU usage
            self.process.cpu_percent(interval=None)
            
            # Process batch
            _ = pipeline_func(batch)
            
            # Collect metrics
            elapsed_time = time.time() - start_time
            end_memory = self.process.memory_info().rss / 1024 / 1024
            cpu_usage = self.process.cpu_percent(interval=None)
            
            # Calculate metrics
            throughput = batch_size / elapsed_time
            latency = (elapsed_time / batch_size) * 1000  # ms per sample
            memory_usage = end_memory - start_memory
            
            throughputs.append(throughput)
            latencies.append(latency)
            memory_usages.append(memory_usage)
            cpu_usages.append(cpu_usage)
            
            # GPU metrics if available
            if self.use_gpu:
                gpu_usage = torch.cuda.utilization()
                gpu_memory = torch.cuda.memory_allocated() / 1024 / 1024
                gpu_usages.append(gpu_usage)
                gpu_memories.append(gpu_memory)
                
        # Average metrics
        metrics = PerformanceMetrics(
            throughput_samples_per_sec=np.mean(throughputs),
            latency_ms=np.mean(latencies),
            memory_usage_mb=np.mean(memory_usages),
            cpu_usage_percent=np.mean(cpu_usages),
            gpu_usage_percent=np.mean(gpu_usages) if gpu_usages else None,
            gpu_memory_mb=np.mean(gpu_memories) if gpu_memories else None
        )
        
        return metrics
    
    def optimize_batch_size(self,
                           pipeline_func: Callable,
                           data_samples: List[Any],
                           batch_sizes: List[int] = [8, 16, 32, 64, 128],
                           target_memory_mb: Optional[float] = None) -> Dict[str, Any]:
        """
        Find optimal batch size for the pipeline.
        
        Args:
            pipeline_func: Pipeline function to optimize
            data_samples: Sample data
            batch_sizes: Batch sizes to test
            target_memory_mb: Maximum memory usage target
            
        Returns:
            Optimization results with recommended batch size
        """
        results = []
        
        for batch_size in batch_sizes:
            logger.info(f"Testing batch size: {batch_size}")
            
            try:
                metrics = self.profile_pipeline(
                    pipeline_func,
                    data_samples,
                    batch_size=batch_size,
                    num_iterations=5
                )
                
                results.append({
                    'batch_size': batch_size,
                    'metrics': metrics,
                    'efficiency_score': self._calculate_efficiency_score(metrics)
                })
                
                # Check memory constraint
                if target_memory_mb and metrics.memory_usage_mb > target_memory_mb:
                    logger.warning(f"Batch size {batch_size} exceeds memory target")
                    
            except Exception as e:
                logger.error(f"Failed to test batch size {batch_size}: {e}")
                
        # Find optimal batch size
        if not results:
            raise RuntimeError("No valid batch sizes found")
            
        # Sort by efficiency score
        results.sort(key=lambda x: x['efficiency_score'], reverse=True)
        
        # Apply memory constraint if specified
        if target_memory_mb:
            valid_results = [r for r in results 
                           if r['metrics'].memory_usage_mb <= target_memory_mb]
            if valid_results:
                results = valid_results
                
        optimal = results[0]
        
        return {
            'optimal_batch_size': optimal['batch_size'],
            'optimal_metrics': optimal['metrics'],
            'all_results': results,
            'recommendation': self._generate_recommendation(optimal)
        }
    
    def create_parallel_pipeline(self,
                               process_func: Callable,
                               num_workers: Optional[int] = None,
                               use_processes: bool = True) -> 'ParallelPipeline':
        """
        Create a parallel processing pipeline.
        
        Args:
            process_func: Function to parallelize
            num_workers: Number of workers (auto-detect if None)
            use_processes: Use processes (True) or threads (False)
            
        Returns:
            ParallelPipeline instance
        """
        if num_workers is None:
            num_workers = self.cpu_count
            
        return ParallelPipeline(
            process_func=process_func,
            num_workers=num_workers,
            use_processes=use_processes,
            use_gpu=self.use_gpu
        )
    
    def benchmark_parallel_processing(self,
                                    process_func: Callable,
                                    data_samples: List[Any],
                                    worker_counts: List[int] = [1, 2, 4, 8],
                                    batch_size: int = 32) -> Dict[str, Any]:
        """
        Benchmark parallel processing with different worker counts.
        
        Args:
            process_func: Function to benchmark
            data_samples: Sample data
            worker_counts: Number of workers to test
            batch_size: Batch size
            
        Returns:
            Benchmark results
        """
        results = []
        
        # Test sequential processing
        seq_metrics = self.profile_pipeline(
            process_func,
            data_samples,
            batch_size=batch_size
        )
        
        results.append({
            'workers': 1,
            'parallel': False,
            'metrics': seq_metrics,
            'speedup': 1.0
        })
        
        # Test parallel processing
        for num_workers in worker_counts:
            if num_workers == 1:
                continue
                
            logger.info(f"Testing with {num_workers} workers")
            
            # Create parallel pipeline
            pipeline = self.create_parallel_pipeline(
                process_func,
                num_workers=num_workers
            )
            
            # Profile parallel processing
            par_metrics = self.profile_pipeline(
                lambda batch: pipeline.process_batch(batch),
                data_samples,
                batch_size=batch_size
            )
            
            speedup = par_metrics.throughput_samples_per_sec / seq_metrics.throughput_samples_per_sec
            
            results.append({
                'workers': num_workers,
                'parallel': True,
                'metrics': par_metrics,
                'speedup': speedup
            })
            
        # Find optimal configuration
        results.sort(key=lambda x: x['metrics'].throughput_samples_per_sec, reverse=True)
        optimal = results[0]
        
        return {
            'optimal_workers': optimal['workers'],
            'optimal_speedup': optimal['speedup'],
            'all_results': results,
            'recommendation': f"Use {optimal['workers']} workers for {optimal['speedup']:.1f}x speedup"
        }
    
    def _calculate_efficiency_score(self, metrics: PerformanceMetrics) -> float:
        """Calculate efficiency score based on multiple metrics"""
        # Normalize metrics (higher is better)
        throughput_score = metrics.throughput_samples_per_sec / 1000  # Normalize to ~1
        latency_score = 1 / (metrics.latency_ms + 1)  # Lower is better
        memory_score = 1 / (metrics.memory_usage_mb + 1)  # Lower is better
        
        # Weighted combination
        score = (
            0.5 * throughput_score +
            0.3 * latency_score +
            0.2 * memory_score
        )
        
        return score
    
    def _generate_recommendation(self, result: Dict[str, Any]) -> str:
        """Generate optimization recommendation"""
        metrics = result['metrics']
        batch_size = result['batch_size']
        
        recommendations = [
            f"Recommended batch size: {batch_size}",
            f"Expected throughput: {metrics.throughput_samples_per_sec:.0f} samples/sec",
            f"Memory usage: {metrics.memory_usage_mb:.1f} MB"
        ]
        
        if metrics.latency_ms > 10:
            recommendations.append("Consider GPU acceleration for lower latency")
            
        if metrics.cpu_usage_percent > 80:
            recommendations.append("High CPU usage - consider parallel processing")
            
        return "\n".join(recommendations)


class ParallelPipeline:
    """
    Parallel processing pipeline for CPU-intensive operations.
    """
    
    def __init__(self,
                 process_func: Callable,
                 num_workers: int = 4,
                 use_processes: bool = True,
                 use_gpu: bool = False):
        """
        Initialize parallel pipeline.
        
        Args:
            process_func: Function to parallelize
            num_workers: Number of workers
            use_processes: Use processes vs threads
            use_gpu: Whether GPU is available
        """
        self.process_func = process_func
        self.num_workers = num_workers
        self.use_processes = use_processes
        self.use_gpu = use_gpu
        
        # Create executor
        if use_processes:
            self.executor = ProcessPoolExecutor(max_workers=num_workers)
        else:
            self.executor = ThreadPoolExecutor(max_workers=num_workers)
            
    def process_batch(self, batch: List[Any]) -> List[Any]:
        """
        Process a batch in parallel.
        
        Args:
            batch: Input batch
            
        Returns:
            Processed results
        """
        # Split batch among workers
        chunk_size = len(batch) // self.num_workers
        chunks = []
        
        for i in range(self.num_workers):
            start_idx = i * chunk_size
            if i == self.num_workers - 1:
                # Last worker gets remaining items
                chunk = batch[start_idx:]
            else:
                chunk = batch[start_idx:start_idx + chunk_size]
            chunks.append(chunk)
            
        # Process chunks in parallel
        futures = [self.executor.submit(self.process_func, chunk) 
                  for chunk in chunks]
        
        # Collect results
        results = []
        for future in futures:
            chunk_results = future.result()
            results.extend(chunk_results)
            
        return results
    
    def shutdown(self):
        """Shutdown the executor"""
        self.executor.shutdown(wait=True)
        
    def __enter__(self):
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        self.shutdown()


class MemoryOptimizer:
    """
    Utilities for optimizing memory usage in data pipelines.
    """
    
    @staticmethod
    def estimate_memory_usage(data_shape: Tuple[int, ...],
                            dtype: np.dtype,
                            batch_size: int,
                            overhead_factor: float = 1.2) -> float:
        """
        Estimate memory usage for a given data configuration.
        
        Args:
            data_shape: Shape of single data sample
            dtype: Data type
            batch_size: Batch size
            overhead_factor: Overhead multiplier
            
        Returns:
            Estimated memory usage in MB
        """
        # Calculate single sample size
        sample_size = np.prod(data_shape) * dtype.itemsize
        
        # Calculate batch size
        batch_memory = sample_size * batch_size
        
        # Add overhead
        total_memory = batch_memory * overhead_factor
        
        # Convert to MB
        return total_memory / (1024 * 1024)
    
    @staticmethod
    def optimize_data_loading(file_path: Path,
                            target_memory_mb: float = 1000) -> Dict[str, Any]:
        """
        Optimize data loading strategy based on file size and memory constraints.
        
        Args:
            file_path: Path to data file
            target_memory_mb: Target memory usage
            
        Returns:
            Loading strategy recommendations
        """
        file_size_mb = file_path.stat().st_size / (1024 * 1024)
        
        recommendations = {
            'file_size_mb': file_size_mb,
            'target_memory_mb': target_memory_mb,
            'strategy': None,
            'chunk_size': None
        }
        
        if file_size_mb < target_memory_mb * 0.5:
            # Can load entire file
            recommendations['strategy'] = 'full_load'
            recommendations['preload'] = True
        elif file_size_mb < target_memory_mb * 2:
            # Use memory mapping
            recommendations['strategy'] = 'memory_map'
            recommendations['preload'] = False
        else:
            # Use chunked loading
            recommendations['strategy'] = 'chunked'
            recommendations['chunk_size'] = int(target_memory_mb * 0.3 * 1024 * 1024)
            recommendations['preload'] = False
            
        return recommendations


def create_optimized_pipeline(data_path: Path,
                            batch_size: Optional[int] = None,
                            target_memory_mb: float = 4000,
                            use_gpu: bool = False) -> Dict[str, Any]:
    """
    Create an optimized data pipeline configuration.
    
    Args:
        data_path: Path to data
        batch_size: Desired batch size (auto-optimize if None)
        target_memory_mb: Target memory usage
        use_gpu: Whether to use GPU
        
    Returns:
        Optimized pipeline configuration
    """
    optimizer = PipelineOptimizer(use_gpu=use_gpu)
    
    # Analyze data
    memory_strategy = MemoryOptimizer.optimize_data_loading(
        data_path,
        target_memory_mb
    )
    
    # Determine optimal configuration
    config = {
        'data_path': str(data_path),
        'memory_strategy': memory_strategy,
        'use_gpu': use_gpu,
        'num_workers': mp.cpu_count() // 2,  # Conservative default
        'pin_memory': use_gpu,
        'persistent_workers': True
    }
    
    # Set batch size
    if batch_size is None:
        if use_gpu:
            config['batch_size'] = 64  # Larger batches for GPU
        else:
            config['batch_size'] = 32  # Smaller batches for CPU
    else:
        config['batch_size'] = batch_size
        
    # Add device-specific optimizations
    if use_gpu:
        config['gpu_optimizations'] = {
            'prefetch_factor': 2,
            'num_workers': min(8, mp.cpu_count()),
            'pin_memory': True
        }
    else:
        config['cpu_optimizations'] = {
            'prefetch_factor': 1,
            'num_workers': mp.cpu_count() // 2,
            'use_shared_memory': True
        }
        
    return config