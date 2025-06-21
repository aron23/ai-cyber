"""Prometheus metrics collector for ML models."""

import time
from contextlib import contextmanager
from typing import Any, Dict, Optional

from prometheus_client import (
    CollectorRegistry,
    Counter,
    Gauge,
    Histogram,
    Summary,
    generate_latest,
    start_http_server,
)


class MLMetricsCollector:
    """Collects and exposes metrics for ML models."""
    
    def __init__(self, port: int = 8000, registry: Optional[CollectorRegistry] = None):
        """Initialize metrics collector.
        
        Args:
            port: Port to expose metrics on
            registry: Prometheus collector registry
        """
        self.port = port
        self.registry = registry or CollectorRegistry()
        
        # Define metrics
        self.prediction_counter = Counter(
            "ml_model_predictions_total",
            "Total number of predictions made",
            ["model_name", "model_version", "prediction_type"],
            registry=self.registry
        )
        
        self.prediction_duration = Histogram(
            "ml_model_prediction_duration_seconds",
            "Time spent processing prediction request",
            ["model_name", "model_version"],
            buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1.0, 2.5, 5.0),
            registry=self.registry
        )
        
        self.error_counter = Counter(
            "ml_model_errors_total",
            "Total number of prediction errors",
            ["model_name", "model_version", "error_type"],
            registry=self.registry
        )
        
        self.model_load_duration = Histogram(
            "ml_model_load_duration_seconds",
            "Time spent loading model",
            ["model_name", "model_version"],
            registry=self.registry
        )
        
        self.active_models = Gauge(
            "ml_active_models",
            "Number of active models in memory",
            registry=self.registry
        )
        
        self.model_memory_usage = Gauge(
            "ml_model_memory_usage_bytes",
            "Memory usage per model",
            ["model_name", "model_version"],
            registry=self.registry
        )
        
        self.batch_size = Summary(
            "ml_batch_size",
            "Batch size for predictions",
            ["model_name", "model_version"],
            registry=self.registry
        )
        
        self.feature_processing_time = Histogram(
            "ml_feature_processing_duration_seconds",
            "Time spent on feature processing",
            ["processing_type"],
            registry=self.registry
        )
        
        self.model_accuracy = Gauge(
            "ml_model_accuracy",
            "Model accuracy metric",
            ["model_name", "model_version", "dataset"],
            registry=self.registry
        )
        
        self.data_drift_score = Gauge(
            "ml_data_drift_score",
            "Data drift detection score",
            ["model_name", "feature_name"],
            registry=self.registry
        )
        
    def start_server(self):
        """Start the metrics HTTP server."""
        start_http_server(self.port, registry=self.registry)
        
    def record_prediction(
        self,
        model_name: str,
        model_version: str,
        prediction_type: str = "single",
        duration: Optional[float] = None
    ):
        """Record a prediction event.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            prediction_type: Type of prediction (single, batch)
            duration: Time taken for prediction
        """
        self.prediction_counter.labels(
            model_name=model_name,
            model_version=model_version,
            prediction_type=prediction_type
        ).inc()
        
        if duration is not None:
            self.prediction_duration.labels(
                model_name=model_name,
                model_version=model_version
            ).observe(duration)
    
    def record_error(
        self,
        model_name: str,
        model_version: str,
        error_type: str
    ):
        """Record a prediction error.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            error_type: Type of error
        """
        self.error_counter.labels(
            model_name=model_name,
            model_version=model_version,
            error_type=error_type
        ).inc()
    
    def record_model_load(
        self,
        model_name: str,
        model_version: str,
        duration: float
    ):
        """Record model loading time.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            duration: Time taken to load model
        """
        self.model_load_duration.labels(
            model_name=model_name,
            model_version=model_version
        ).observe(duration)
    
    def set_active_models(self, count: int):
        """Set the number of active models.
        
        Args:
            count: Number of active models
        """
        self.active_models.set(count)
    
    def set_model_memory(
        self,
        model_name: str,
        model_version: str,
        memory_bytes: int
    ):
        """Set memory usage for a model.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            memory_bytes: Memory usage in bytes
        """
        self.model_memory_usage.labels(
            model_name=model_name,
            model_version=model_version
        ).set(memory_bytes)
    
    def record_batch_size(
        self,
        model_name: str,
        model_version: str,
        batch_size: int
    ):
        """Record batch size for predictions.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            batch_size: Size of the batch
        """
        self.batch_size.labels(
            model_name=model_name,
            model_version=model_version
        ).observe(batch_size)
    
    def record_feature_processing(
        self,
        processing_type: str,
        duration: float
    ):
        """Record feature processing time.
        
        Args:
            processing_type: Type of processing
            duration: Time taken
        """
        self.feature_processing_time.labels(
            processing_type=processing_type
        ).observe(duration)
    
    def set_model_accuracy(
        self,
        model_name: str,
        model_version: str,
        dataset: str,
        accuracy: float
    ):
        """Set model accuracy metric.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            dataset: Dataset name
            accuracy: Accuracy value
        """
        self.model_accuracy.labels(
            model_name=model_name,
            model_version=model_version,
            dataset=dataset
        ).set(accuracy)
    
    def set_data_drift(
        self,
        model_name: str,
        feature_name: str,
        drift_score: float
    ):
        """Set data drift score.
        
        Args:
            model_name: Name of the model
            feature_name: Name of the feature
            drift_score: Drift score value
        """
        self.data_drift_score.labels(
            model_name=model_name,
            feature_name=feature_name
        ).set(drift_score)
    
    @contextmanager
    def measure_prediction(
        self,
        model_name: str,
        model_version: str,
        prediction_type: str = "single"
    ):
        """Context manager to measure prediction time.
        
        Args:
            model_name: Name of the model
            model_version: Version of the model
            prediction_type: Type of prediction
        """
        start_time = time.time()
        try:
            yield
            duration = time.time() - start_time
            self.record_prediction(
                model_name=model_name,
                model_version=model_version,
                prediction_type=prediction_type,
                duration=duration
            )
        except Exception as e:
            duration = time.time() - start_time
            self.record_error(
                model_name=model_name,
                model_version=model_version,
                error_type=type(e).__name__
            )
            raise
    
    @contextmanager
    def measure_feature_processing(self, processing_type: str):
        """Context manager to measure feature processing time.
        
        Args:
            processing_type: Type of processing
        """
        start_time = time.time()
        yield
        duration = time.time() - start_time
        self.record_feature_processing(
            processing_type=processing_type,
            duration=duration
        )
    
    def get_metrics(self) -> bytes:
        """Get current metrics in Prometheus format.
        
        Returns:
            Metrics in Prometheus text format
        """
        return generate_latest(self.registry)


# Global metrics collector instance
metrics_collector = MLMetricsCollector()


def init_metrics(port: int = 8000):
    """Initialize and start metrics server.
    
    Args:
        port: Port to expose metrics on
    """
    metrics_collector.port = port
    metrics_collector.start_server()