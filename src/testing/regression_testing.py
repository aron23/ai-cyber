import json
import logging
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import pandas as pd
import numpy as np
from src.testing.model_test_suite import ModelTestSuite, TestResult

logger = logging.getLogger(__name__)


@dataclass
class PerformanceBenchmark:
    model_name: str
    model_version: str
    timestamp: str
    accuracy: float
    inference_latency_ms: Dict[str, float]  # batch_size -> latency
    memory_usage_gb: float
    throughput_samples_per_sec: float
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


@dataclass
class RegressionTestConfig:
    accuracy_threshold: float = 0.9
    accuracy_regression_tolerance: float = 0.02  # Max allowed accuracy drop
    latency_threshold_ms: float = 100
    latency_regression_tolerance: float = 1.1  # Max 10% latency increase
    memory_threshold_gb: float = 4.0
    memory_regression_tolerance: float = 1.2  # Max 20% memory increase
    min_throughput_samples_per_sec: float = 10.0


class RegressionTestRunner:
    def __init__(self, config: RegressionTestConfig = None):
        self.config = config or RegressionTestConfig()
        self.benchmarks_dir = Path("benchmarks")
        self.benchmarks_dir.mkdir(exist_ok=True)
        self.baseline_benchmarks: Dict[str, PerformanceBenchmark] = {}
        self._load_baselines()
    
    def _load_baselines(self):
        baseline_file = self.benchmarks_dir / "baseline_benchmarks.json"
        if baseline_file.exists():
            with open(baseline_file, 'r') as f:
                data = json.load(f)
                for model_name, benchmark_data in data.items():
                    self.baseline_benchmarks[model_name] = PerformanceBenchmark(**benchmark_data)
    
    def _save_baselines(self):
        baseline_file = self.benchmarks_dir / "baseline_benchmarks.json"
        data = {
            name: benchmark.to_dict() 
            for name, benchmark in self.baseline_benchmarks.items()
        }
        with open(baseline_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def run_regression_tests(self, model: Any, model_name: str, model_version: str,
                           test_data: Any = None) -> Dict[str, Any]:
        logger.info(f"Running regression tests for {model_name} v{model_version}")
        
        # Run comprehensive tests
        test_suite = ModelTestSuite()
        test_results = test_suite.run_all_tests(model, test_data)
        
        # Extract metrics from test results
        current_benchmark = self._extract_benchmark(
            model_name, model_version, test_results
        )
        
        # Compare with baseline if exists
        regression_results = {
            "model_name": model_name,
            "model_version": model_version,
            "timestamp": current_benchmark.timestamp,
            "current_metrics": current_benchmark.to_dict(),
            "regression_tests": [],
            "overall_passed": True
        }
        
        if model_name in self.baseline_benchmarks:
            baseline = self.baseline_benchmarks[model_name]
            regression_tests = self._compare_with_baseline(current_benchmark, baseline)
            regression_results["regression_tests"] = regression_tests
            regression_results["overall_passed"] = all(
                test["passed"] for test in regression_tests
            )
            regression_results["baseline_metrics"] = baseline.to_dict()
        else:
            logger.info(f"No baseline found for {model_name}. Setting current as baseline.")
            self.baseline_benchmarks[model_name] = current_benchmark
            self._save_baselines()
        
        # Save regression test results
        self._save_regression_results(regression_results)
        
        return regression_results
    
    def _extract_benchmark(self, model_name: str, model_version: str,
                          test_results: Dict[str, Any]) -> PerformanceBenchmark:
        # Extract metrics from test results
        accuracy = 0.0
        latency_ms = {}
        memory_gb = 0.0
        throughput = 0.0
        
        for result in test_results["results"]:
            if result["test_name"] == "model_performance":
                accuracy = result["metrics"].get("accuracy", 0.0)
            elif result["test_name"] == "inference_speed":
                batch_results = result["metrics"].get("batch_results", {})
                for batch_key, metrics in batch_results.items():
                    latency_ms[batch_key] = metrics.get("p99_latency_ms", 0)
                    throughput = max(throughput, 
                                   metrics.get("throughput_samples_per_sec", 0))
            elif result["test_name"] == "memory_usage":
                memory_gb = result["metrics"].get("model_size_gb", 0.0)
        
        return PerformanceBenchmark(
            model_name=model_name,
            model_version=model_version,
            timestamp=datetime.now().isoformat(),
            accuracy=accuracy,
            inference_latency_ms=latency_ms,
            memory_usage_gb=memory_gb,
            throughput_samples_per_sec=throughput
        )
    
    def _compare_with_baseline(self, current: PerformanceBenchmark,
                              baseline: PerformanceBenchmark) -> List[Dict[str, Any]]:
        regression_tests = []
        
        # Accuracy regression test
        accuracy_drop = baseline.accuracy - current.accuracy
        accuracy_test = {
            "test_name": "accuracy_regression",
            "passed": accuracy_drop <= self.config.accuracy_regression_tolerance,
            "baseline_value": baseline.accuracy,
            "current_value": current.accuracy,
            "change": -accuracy_drop,
            "tolerance": self.config.accuracy_regression_tolerance,
            "message": f"Accuracy: {current.accuracy:.4f} (baseline: {baseline.accuracy:.4f})"
        }
        regression_tests.append(accuracy_test)
        
        # Latency regression test
        for batch_key in current.inference_latency_ms:
            if batch_key in baseline.inference_latency_ms:
                current_latency = current.inference_latency_ms[batch_key]
                baseline_latency = baseline.inference_latency_ms[batch_key]
                latency_ratio = current_latency / baseline_latency if baseline_latency > 0 else float('inf')
                
                latency_test = {
                    "test_name": f"latency_regression_{batch_key}",
                    "passed": latency_ratio <= self.config.latency_regression_tolerance,
                    "baseline_value": baseline_latency,
                    "current_value": current_latency,
                    "change_ratio": latency_ratio,
                    "tolerance": self.config.latency_regression_tolerance,
                    "message": f"Latency {batch_key}: {current_latency:.2f}ms (baseline: {baseline_latency:.2f}ms)"
                }
                regression_tests.append(latency_test)
        
        # Memory regression test
        memory_ratio = current.memory_usage_gb / baseline.memory_usage_gb if baseline.memory_usage_gb > 0 else float('inf')
        memory_test = {
            "test_name": "memory_regression",
            "passed": memory_ratio <= self.config.memory_regression_tolerance,
            "baseline_value": baseline.memory_usage_gb,
            "current_value": current.memory_usage_gb,
            "change_ratio": memory_ratio,
            "tolerance": self.config.memory_regression_tolerance,
            "message": f"Memory: {current.memory_usage_gb:.2f}GB (baseline: {baseline.memory_usage_gb:.2f}GB)"
        }
        regression_tests.append(memory_test)
        
        # Throughput regression test
        throughput_ratio = current.throughput_samples_per_sec / baseline.throughput_samples_per_sec if baseline.throughput_samples_per_sec > 0 else 0
        throughput_test = {
            "test_name": "throughput_regression",
            "passed": current.throughput_samples_per_sec >= self.config.min_throughput_samples_per_sec,
            "baseline_value": baseline.throughput_samples_per_sec,
            "current_value": current.throughput_samples_per_sec,
            "change_ratio": throughput_ratio,
            "min_required": self.config.min_throughput_samples_per_sec,
            "message": f"Throughput: {current.throughput_samples_per_sec:.2f} samples/sec"
        }
        regression_tests.append(throughput_test)
        
        return regression_tests
    
    def _save_regression_results(self, results: Dict[str, Any]):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        results_file = self.benchmarks_dir / f"regression_test_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save to latest results
        latest_file = self.benchmarks_dir / "latest_regression_results.json"
        with open(latest_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def update_baseline(self, model_name: str, benchmark: PerformanceBenchmark):
        self.baseline_benchmarks[model_name] = benchmark
        self._save_baselines()
        logger.info(f"Updated baseline for {model_name}")
    
    def generate_regression_report(self, model_name: str) -> pd.DataFrame:
        # Load all regression test results for the model
        regression_files = list(self.benchmarks_dir.glob("regression_test_*.json"))
        
        data = []
        for file in regression_files:
            with open(file, 'r') as f:
                result = json.load(f)
                if result.get("model_name") == model_name:
                    metrics = result["current_metrics"]
                    data.append({
                        "timestamp": metrics["timestamp"],
                        "version": metrics["model_version"],
                        "accuracy": metrics["accuracy"],
                        "latency_ms": metrics.get("inference_latency_ms", {}).get("batch_8", 0),
                        "memory_gb": metrics["memory_usage_gb"],
                        "throughput": metrics["throughput_samples_per_sec"],
                        "passed": result["overall_passed"]
                    })
        
        if data:
            df = pd.DataFrame(data)
            df["timestamp"] = pd.to_datetime(df["timestamp"])
            df = df.sort_values("timestamp")
            return df
        else:
            return pd.DataFrame()
    
    def visualize_trends(self, model_name: str, output_path: str = None):
        try:
            import matplotlib.pyplot as plt
            
            df = self.generate_regression_report(model_name)
            if df.empty:
                logger.warning(f"No regression data found for {model_name}")
                return
            
            fig, axes = plt.subplots(2, 2, figsize=(12, 8))
            fig.suptitle(f"Performance Trends for {model_name}")
            
            # Accuracy trend
            axes[0, 0].plot(df["timestamp"], df["accuracy"], marker='o')
            axes[0, 0].set_title("Model Accuracy")
            axes[0, 0].set_ylabel("Accuracy")
            axes[0, 0].grid(True)
            
            # Latency trend
            axes[0, 1].plot(df["timestamp"], df["latency_ms"], marker='o', color='orange')
            axes[0, 1].set_title("Inference Latency (batch=8)")
            axes[0, 1].set_ylabel("Latency (ms)")
            axes[0, 1].grid(True)
            
            # Memory trend
            axes[1, 0].plot(df["timestamp"], df["memory_gb"], marker='o', color='green')
            axes[1, 0].set_title("Memory Usage")
            axes[1, 0].set_ylabel("Memory (GB)")
            axes[1, 0].grid(True)
            
            # Throughput trend
            axes[1, 1].plot(df["timestamp"], df["throughput"], marker='o', color='purple')
            axes[1, 1].set_title("Throughput")
            axes[1, 1].set_ylabel("Samples/sec")
            axes[1, 1].grid(True)
            
            # Adjust layout
            plt.tight_layout()
            
            if output_path:
                plt.savefig(output_path)
            else:
                plt.savefig(self.benchmarks_dir / f"{model_name}_trends.png")
            
            plt.close()
            
        except ImportError:
            logger.warning("Matplotlib not available. Skipping visualization.")