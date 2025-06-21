import time
import psutil
import numpy as np
import torch
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
import logging
from pathlib import Path
import json

logger = logging.getLogger(__name__)


@dataclass
class TestResult:
    test_name: str
    passed: bool
    metrics: Dict[str, Any]
    message: str = ""
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "test_name": self.test_name,
            "passed": self.passed,
            "metrics": self.metrics,
            "message": self.message
        }


class ModelTestSuite:
    def __init__(self, device: str = 'cpu'):
        self.device = device
        self.results: List[TestResult] = []
        
    def test_model_performance(self, model: Any, test_data: Any, 
                               accuracy_threshold: float = 0.9) -> TestResult:
        logger.info("Running model performance test...")
        
        try:
            model.eval()
            correct = 0
            total = 0
            
            with torch.no_grad():
                for batch_data, batch_labels in test_data:
                    if hasattr(batch_data, 'to'):
                        batch_data = batch_data.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                    
                    outputs = model(batch_data)
                    if hasattr(outputs, 'logits'):
                        outputs = outputs.logits
                    
                    _, predicted = torch.max(outputs, 1)
                    total += batch_labels.size(0)
                    correct += (predicted == batch_labels).sum().item()
            
            accuracy = correct / total if total > 0 else 0
            passed = accuracy >= accuracy_threshold
            
            result = TestResult(
                test_name="model_performance",
                passed=passed,
                metrics={
                    "accuracy": accuracy,
                    "threshold": accuracy_threshold,
                    "total_samples": total,
                    "correct_predictions": correct
                },
                message=f"Model accuracy: {accuracy:.4f} (threshold: {accuracy_threshold})"
            )
            
        except Exception as e:
            result = TestResult(
                test_name="model_performance",
                passed=False,
                metrics={},
                message=f"Performance test failed: {str(e)}"
            )
        
        self.results.append(result)
        return result
    
    def test_inference_speed(self, model: Any, batch_sizes: List[int] = [1, 8, 16, 32],
                            latency_threshold_ms: float = 100) -> TestResult:
        logger.info("Running inference speed test...")
        
        try:
            model.eval()
            speed_results = {}
            all_passed = True
            
            for batch_size in batch_sizes:
                dummy_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
                
                # Warmup
                for _ in range(10):
                    _ = model(dummy_input)
                
                # Measure inference time
                if self.device == 'cuda':
                    torch.cuda.synchronize()
                
                times = []
                for _ in range(100):
                    start_time = time.perf_counter()
                    _ = model(dummy_input)
                    if self.device == 'cuda':
                        torch.cuda.synchronize()
                    end_time = time.perf_counter()
                    times.append((end_time - start_time) * 1000)  # Convert to ms
                
                avg_time = np.mean(times)
                p95_time = np.percentile(times, 95)
                p99_time = np.percentile(times, 99)
                
                speed_results[f"batch_{batch_size}"] = {
                    "avg_latency_ms": avg_time,
                    "p95_latency_ms": p95_time,
                    "p99_latency_ms": p99_time,
                    "throughput_samples_per_sec": batch_size / (avg_time / 1000)
                }
                
                if p99_time > latency_threshold_ms:
                    all_passed = False
            
            result = TestResult(
                test_name="inference_speed",
                passed=all_passed,
                metrics={
                    "batch_results": speed_results,
                    "latency_threshold_ms": latency_threshold_ms
                },
                message=f"Inference speed test {'passed' if all_passed else 'failed'}"
            )
            
        except Exception as e:
            result = TestResult(
                test_name="inference_speed",
                passed=False,
                metrics={},
                message=f"Speed test failed: {str(e)}"
            )
        
        self.results.append(result)
        return result
    
    def test_model_robustness(self, model: Any, test_data: Any,
                             perturbations: Dict[str, float] = None) -> TestResult:
        logger.info("Running model robustness test...")
        
        if perturbations is None:
            perturbations = {
                "gaussian_noise": 0.1,
                "uniform_noise": 0.05,
                "adversarial_epsilon": 0.01
            }
        
        try:
            model.eval()
            robustness_results = {}
            
            # Get clean accuracy first
            clean_correct = 0
            total = 0
            
            for batch_data, batch_labels in test_data:
                if hasattr(batch_data, 'to'):
                    batch_data = batch_data.to(self.device)
                    batch_labels = batch_labels.to(self.device)
                
                outputs = model(batch_data)
                if hasattr(outputs, 'logits'):
                    outputs = outputs.logits
                
                _, predicted = torch.max(outputs, 1)
                total += batch_labels.size(0)
                clean_correct += (predicted == batch_labels).sum().item()
                
                if total >= 100:  # Test on first 100 samples
                    break
            
            clean_accuracy = clean_correct / total if total > 0 else 0
            robustness_results["clean_accuracy"] = clean_accuracy
            
            # Test with Gaussian noise
            if "gaussian_noise" in perturbations:
                noisy_correct = 0
                for batch_data, batch_labels in test_data:
                    if hasattr(batch_data, 'to'):
                        batch_data = batch_data.to(self.device)
                        batch_labels = batch_labels.to(self.device)
                    
                    noise = torch.randn_like(batch_data) * perturbations["gaussian_noise"]
                    noisy_data = batch_data + noise
                    
                    outputs = model(noisy_data)
                    if hasattr(outputs, 'logits'):
                        outputs = outputs.logits
                    
                    _, predicted = torch.max(outputs, 1)
                    noisy_correct += (predicted == batch_labels).sum().item()
                    
                    if noisy_correct >= 100:
                        break
                
                noisy_accuracy = noisy_correct / total if total > 0 else 0
                robustness_results["gaussian_noise_accuracy"] = noisy_accuracy
                robustness_results["gaussian_noise_drop"] = clean_accuracy - noisy_accuracy
            
            # Test passes if accuracy drop is less than 10%
            max_drop = max(robustness_results.get("gaussian_noise_drop", 0),
                          robustness_results.get("uniform_noise_drop", 0))
            passed = max_drop < 0.1
            
            result = TestResult(
                test_name="model_robustness",
                passed=passed,
                metrics=robustness_results,
                message=f"Max accuracy drop: {max_drop:.4f}"
            )
            
        except Exception as e:
            result = TestResult(
                test_name="model_robustness",
                passed=False,
                metrics={},
                message=f"Robustness test failed: {str(e)}"
            )
        
        self.results.append(result)
        return result
    
    def test_memory_usage(self, model: Any, max_memory_gb: float = 4.0) -> TestResult:
        logger.info("Running memory usage test...")
        
        try:
            process = psutil.Process()
            
            # Baseline memory
            baseline_memory = process.memory_info().rss / 1024 / 1024 / 1024  # GB
            
            # Model memory
            model_size = 0
            for param in model.parameters():
                model_size += param.nelement() * param.element_size()
            model_size_gb = model_size / 1024 / 1024 / 1024
            
            # Test inference memory
            batch_sizes = [1, 8, 16, 32]
            memory_results = {
                "baseline_memory_gb": baseline_memory,
                "model_size_gb": model_size_gb,
                "inference_memory": {}
            }
            
            for batch_size in batch_sizes:
                dummy_input = torch.randn(batch_size, 3, 224, 224).to(self.device)
                
                # Force garbage collection
                import gc
                gc.collect()
                if self.device == 'cuda':
                    torch.cuda.empty_cache()
                
                mem_before = process.memory_info().rss / 1024 / 1024 / 1024
                
                # Run inference
                with torch.no_grad():
                    _ = model(dummy_input)
                
                mem_after = process.memory_info().rss / 1024 / 1024 / 1024
                memory_used = mem_after - mem_before
                
                memory_results["inference_memory"][f"batch_{batch_size}"] = {
                    "memory_used_gb": memory_used,
                    "total_memory_gb": mem_after
                }
            
            # Check if peak memory exceeds limit
            peak_memory = max(
                memory_results["inference_memory"][k]["total_memory_gb"] 
                for k in memory_results["inference_memory"]
            )
            passed = peak_memory <= max_memory_gb
            
            result = TestResult(
                test_name="memory_usage",
                passed=passed,
                metrics=memory_results,
                message=f"Peak memory: {peak_memory:.2f}GB (limit: {max_memory_gb}GB)"
            )
            
        except Exception as e:
            result = TestResult(
                test_name="memory_usage",
                passed=False,
                metrics={},
                message=f"Memory test failed: {str(e)}"
            )
        
        self.results.append(result)
        return result
    
    def run_all_tests(self, model: Any, test_data: Any = None) -> Dict[str, Any]:
        logger.info("Running comprehensive model test suite...")
        
        # Run all tests
        if test_data is not None:
            self.test_model_performance(model, test_data)
            self.test_model_robustness(model, test_data)
        
        self.test_inference_speed(model)
        self.test_memory_usage(model)
        
        # Generate summary
        total_tests = len(self.results)
        passed_tests = sum(1 for r in self.results if r.passed)
        
        summary = {
            "total_tests": total_tests,
            "passed_tests": passed_tests,
            "failed_tests": total_tests - passed_tests,
            "pass_rate": passed_tests / total_tests if total_tests > 0 else 0,
            "results": [r.to_dict() for r in self.results]
        }
        
        return summary
    
    def save_results(self, output_path: str):
        output_file = Path(output_path)
        output_file.parent.mkdir(parents=True, exist_ok=True)
        
        summary = {
            "total_tests": len(self.results),
            "passed_tests": sum(1 for r in self.results if r.passed),
            "results": [r.to_dict() for r in self.results]
        }
        
        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2)
        
        logger.info(f"Test results saved to {output_file}")