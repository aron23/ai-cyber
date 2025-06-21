import logging
import numpy as np
from scipy import stats
from typing import Dict, List, Tuple, Any, Optional
from dataclasses import dataclass
from datetime import datetime
import json
from pathlib import Path
import torch
from concurrent.futures import ThreadPoolExecutor
import time

logger = logging.getLogger(__name__)


@dataclass
class ABTestResult:
    model_a_name: str
    model_b_name: str
    metric_name: str
    model_a_value: float
    model_b_value: float
    p_value: float
    is_significant: bool
    confidence_level: float
    sample_size: int
    winner: Optional[str] = None
    relative_improvement: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return {
            "model_a_name": self.model_a_name,
            "model_b_name": self.model_b_name,
            "metric_name": self.metric_name,
            "model_a_value": self.model_a_value,
            "model_b_value": self.model_b_value,
            "p_value": self.p_value,
            "is_significant": self.is_significant,
            "confidence_level": self.confidence_level,
            "sample_size": self.sample_size,
            "winner": self.winner,
            "relative_improvement": self.relative_improvement
        }


class ABTestingFramework:
    def __init__(self, confidence_level: float = 0.95, min_sample_size: int = 100):
        self.confidence_level = confidence_level
        self.min_sample_size = min_sample_size
        self.results_dir = Path("ab_test_results")
        self.results_dir.mkdir(exist_ok=True)
        self.rollback_configs: Dict[str, Any] = {}
    
    def compare_models(self, model_a: Any, model_b: Any, 
                      model_a_name: str, model_b_name: str,
                      test_data: Any, metrics: List[str] = None) -> Dict[str, Any]:
        logger.info(f"Starting A/B test between {model_a_name} and {model_b_name}")
        
        if metrics is None:
            metrics = ["accuracy", "latency", "throughput"]
        
        results = {
            "test_id": f"ab_test_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
            "model_a": model_a_name,
            "model_b": model_b_name,
            "timestamp": datetime.now().isoformat(),
            "metrics": {},
            "overall_winner": None,
            "recommendation": None
        }
        
        # Store rollback configuration
        self.rollback_configs[results["test_id"]] = {
            "previous_model": model_a_name,
            "timestamp": results["timestamp"]
        }
        
        # Run tests for each metric
        for metric in metrics:
            if metric == "accuracy":
                result = self._compare_accuracy(model_a, model_b, model_a_name, 
                                              model_b_name, test_data)
            elif metric == "latency":
                result = self._compare_latency(model_a, model_b, model_a_name,
                                             model_b_name)
            elif metric == "throughput":
                result = self._compare_throughput(model_a, model_b, model_a_name,
                                                model_b_name)
            else:
                logger.warning(f"Unknown metric: {metric}")
                continue
            
            results["metrics"][metric] = result.to_dict()
        
        # Determine overall winner
        results["overall_winner"] = self._determine_overall_winner(results["metrics"])
        results["recommendation"] = self._generate_recommendation(results)
        
        # Save results
        self._save_results(results)
        
        return results
    
    def _compare_accuracy(self, model_a: Any, model_b: Any,
                         model_a_name: str, model_b_name: str,
                         test_data: Any) -> ABTestResult:
        logger.info("Comparing model accuracy...")
        
        # Collect predictions from both models
        model_a_correct = []
        model_b_correct = []
        
        model_a.eval()
        model_b.eval()
        
        with torch.no_grad():
            for batch_data, batch_labels in test_data:
                # Model A predictions
                outputs_a = model_a(batch_data)
                if hasattr(outputs_a, 'logits'):
                    outputs_a = outputs_a.logits
                _, predicted_a = torch.max(outputs_a, 1)
                correct_a = (predicted_a == batch_labels).cpu().numpy()
                model_a_correct.extend(correct_a)
                
                # Model B predictions
                outputs_b = model_b(batch_data)
                if hasattr(outputs_b, 'logits'):
                    outputs_b = outputs_b.logits
                _, predicted_b = torch.max(outputs_b, 1)
                correct_b = (predicted_b == batch_labels).cpu().numpy()
                model_b_correct.extend(correct_b)
                
                if len(model_a_correct) >= self.min_sample_size:
                    break
        
        # Calculate accuracies
        accuracy_a = np.mean(model_a_correct)
        accuracy_b = np.mean(model_b_correct)
        
        # Perform statistical test (McNemar's test for paired samples)
        # Create contingency table
        both_correct = sum(1 for a, b in zip(model_a_correct, model_b_correct) if a and b)
        a_only_correct = sum(1 for a, b in zip(model_a_correct, model_b_correct) if a and not b)
        b_only_correct = sum(1 for a, b in zip(model_a_correct, model_b_correct) if not a and b)
        
        # McNemar's test
        if a_only_correct + b_only_correct > 0:
            statistic = (abs(a_only_correct - b_only_correct) - 1) ** 2 / (a_only_correct + b_only_correct)
            p_value = 1 - stats.chi2.cdf(statistic, df=1)
        else:
            p_value = 1.0
        
        is_significant = p_value < (1 - self.confidence_level)
        
        # Determine winner
        winner = None
        relative_improvement = None
        if is_significant:
            if accuracy_b > accuracy_a:
                winner = model_b_name
                relative_improvement = (accuracy_b - accuracy_a) / accuracy_a if accuracy_a > 0 else 0
            else:
                winner = model_a_name
                relative_improvement = (accuracy_a - accuracy_b) / accuracy_b if accuracy_b > 0 else 0
        
        return ABTestResult(
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            metric_name="accuracy",
            model_a_value=accuracy_a,
            model_b_value=accuracy_b,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            sample_size=len(model_a_correct),
            winner=winner,
            relative_improvement=relative_improvement
        )
    
    def _compare_latency(self, model_a: Any, model_b: Any,
                        model_a_name: str, model_b_name: str) -> ABTestResult:
        logger.info("Comparing model latency...")
        
        batch_size = 8
        num_samples = max(self.min_sample_size, 100)
        
        model_a_latencies = []
        model_b_latencies = []
        
        dummy_input = torch.randn(batch_size, 3, 224, 224)
        
        # Warmup
        for _ in range(10):
            model_a(dummy_input)
            model_b(dummy_input)
        
        # Measure latencies
        for _ in range(num_samples):
            # Model A
            start = time.perf_counter()
            model_a(dummy_input)
            model_a_latencies.append((time.perf_counter() - start) * 1000)
            
            # Model B
            start = time.perf_counter()
            model_b(dummy_input)
            model_b_latencies.append((time.perf_counter() - start) * 1000)
        
        # Calculate statistics
        latency_a = np.mean(model_a_latencies)
        latency_b = np.mean(model_b_latencies)
        
        # Perform t-test
        statistic, p_value = stats.ttest_ind(model_a_latencies, model_b_latencies)
        is_significant = p_value < (1 - self.confidence_level)
        
        # Determine winner (lower latency is better)
        winner = None
        relative_improvement = None
        if is_significant:
            if latency_b < latency_a:
                winner = model_b_name
                relative_improvement = (latency_a - latency_b) / latency_a
            else:
                winner = model_a_name
                relative_improvement = (latency_b - latency_a) / latency_b
        
        return ABTestResult(
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            metric_name="latency",
            model_a_value=latency_a,
            model_b_value=latency_b,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            sample_size=num_samples,
            winner=winner,
            relative_improvement=relative_improvement
        )
    
    def _compare_throughput(self, model_a: Any, model_b: Any,
                           model_a_name: str, model_b_name: str) -> ABTestResult:
        logger.info("Comparing model throughput...")
        
        batch_sizes = [1, 8, 16, 32]
        model_a_throughputs = []
        model_b_throughputs = []
        
        for batch_size in batch_sizes:
            dummy_input = torch.randn(batch_size, 3, 224, 224)
            
            # Model A throughput
            start = time.perf_counter()
            for _ in range(50):
                model_a(dummy_input)
            elapsed_a = time.perf_counter() - start
            throughput_a = (50 * batch_size) / elapsed_a
            model_a_throughputs.append(throughput_a)
            
            # Model B throughput
            start = time.perf_counter()
            for _ in range(50):
                model_b(dummy_input)
            elapsed_b = time.perf_counter() - start
            throughput_b = (50 * batch_size) / elapsed_b
            model_b_throughputs.append(throughput_b)
        
        # Use maximum throughput
        max_throughput_a = max(model_a_throughputs)
        max_throughput_b = max(model_b_throughputs)
        
        # For throughput comparison, we'll use the ratio test
        throughput_ratio = max_throughput_b / max_throughput_a if max_throughput_a > 0 else 0
        
        # Consider significant if difference is > 10%
        is_significant = abs(throughput_ratio - 1.0) > 0.1
        p_value = 0.001 if is_significant else 0.5  # Simplified for throughput
        
        # Determine winner (higher throughput is better)
        winner = None
        relative_improvement = None
        if is_significant:
            if max_throughput_b > max_throughput_a:
                winner = model_b_name
                relative_improvement = (max_throughput_b - max_throughput_a) / max_throughput_a
            else:
                winner = model_a_name
                relative_improvement = (max_throughput_a - max_throughput_b) / max_throughput_b
        
        return ABTestResult(
            model_a_name=model_a_name,
            model_b_name=model_b_name,
            metric_name="throughput",
            model_a_value=max_throughput_a,
            model_b_value=max_throughput_b,
            p_value=p_value,
            is_significant=is_significant,
            confidence_level=self.confidence_level,
            sample_size=len(batch_sizes) * 50,
            winner=winner,
            relative_improvement=relative_improvement
        )
    
    def _determine_overall_winner(self, metrics: Dict[str, Dict]) -> Optional[str]:
        # Count wins for each model
        model_wins = {}
        
        for metric_result in metrics.values():
            winner = metric_result.get("winner")
            if winner:
                model_wins[winner] = model_wins.get(winner, 0) + 1
        
        if not model_wins:
            return None
        
        # Return model with most wins
        return max(model_wins.items(), key=lambda x: x[1])[0]
    
    def _generate_recommendation(self, results: Dict[str, Any]) -> str:
        overall_winner = results.get("overall_winner")
        
        if not overall_winner:
            return "No significant difference detected between models. Recommend keeping current model."
        
        # Check if new model (B) is better
        if overall_winner == results["model_b"]:
            improvements = []
            for metric, result in results["metrics"].items():
                if result.get("winner") == results["model_b"] and result.get("relative_improvement"):
                    improvements.append(f"{metric}: {result['relative_improvement']*100:.1f}%")
            
            if improvements:
                return f"Recommend deploying {results['model_b']}. Improvements: {', '.join(improvements)}"
            else:
                return f"Recommend deploying {results['model_b']} based on test results."
        else:
            return f"Recommend keeping {results['model_a']}. New model shows no significant improvement."
    
    def rollback(self, test_id: str) -> Dict[str, Any]:
        logger.info(f"Initiating rollback for test {test_id}")
        
        if test_id not in self.rollback_configs:
            return {
                "status": "error",
                "message": f"No rollback configuration found for test {test_id}"
            }
        
        config = self.rollback_configs[test_id]
        
        # In a real system, this would trigger deployment rollback
        rollback_result = {
            "status": "success",
            "test_id": test_id,
            "rolled_back_to": config["previous_model"],
            "timestamp": datetime.now().isoformat(),
            "original_test_timestamp": config["timestamp"]
        }
        
        # Save rollback event
        rollback_file = self.results_dir / f"rollback_{test_id}.json"
        with open(rollback_file, 'w') as f:
            json.dump(rollback_result, f, indent=2)
        
        return rollback_result
    
    def _save_results(self, results: Dict[str, Any]):
        results_file = self.results_dir / f"{results['test_id']}.json"
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Also save as latest
        latest_file = self.results_dir / "latest_ab_test.json"
        with open(latest_file, 'w') as f:
            json.dump(results, f, indent=2)
    
    def get_test_history(self, model_name: str = None) -> List[Dict[str, Any]]:
        history = []
        
        for results_file in self.results_dir.glob("ab_test_*.json"):
            with open(results_file, 'r') as f:
                result = json.load(f)
                
                if model_name is None or model_name in [result["model_a"], result["model_b"]]:
                    history.append({
                        "test_id": result["test_id"],
                        "timestamp": result["timestamp"],
                        "models": [result["model_a"], result["model_b"]],
                        "winner": result.get("overall_winner"),
                        "recommendation": result.get("recommendation")
                    })
        
        # Sort by timestamp
        history.sort(key=lambda x: x["timestamp"], reverse=True)
        return history