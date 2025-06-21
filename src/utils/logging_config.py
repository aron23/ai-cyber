"""Centralized logging configuration for the ML project."""

import json
import logging
import logging.handlers
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml
from pythonjsonlogger import jsonlogger


class MLProjectLogger:
    """Custom logger for ML project with structured logging support."""

    _instance = None
    _initialized = False

    def __new__(cls):
        """Singleton pattern to ensure single logger instance."""
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __init__(self):
        """Initialize logger configuration."""
        if not self._initialized:
            self.log_dir = Path("logs")
            self.log_dir.mkdir(exist_ok=True)
            self._setup_logger()
            MLProjectLogger._initialized = True

    def _setup_logger(self):
        """Set up logger with multiple handlers."""
        # Create logger
        self.logger = logging.getLogger("ml_project")
        self.logger.setLevel(logging.DEBUG)
        self.logger.propagate = False

        # Remove existing handlers
        self.logger.handlers = []

        # Add handlers
        self._add_console_handler()
        self._add_file_handler()
        self._add_json_handler()
        self._add_error_handler()

    def _add_console_handler(self):
        """Add colored console handler."""
        console_handler = logging.StreamHandler(sys.stdout)
        console_handler.setLevel(logging.INFO)
        
        # Create colored formatter
        class ColoredFormatter(logging.Formatter):
            """Colored formatter for console output."""
            
            grey = "\x1b[38;21m"
            green = "\x1b[32m"
            yellow = "\x1b[33m"
            red = "\x1b[31m"
            bold_red = "\x1b[31;1m"
            reset = "\x1b[0m"
            
            FORMATS = {
                logging.DEBUG: grey + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
                logging.INFO: green + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
                logging.WARNING: yellow + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
                logging.ERROR: red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset,
                logging.CRITICAL: bold_red + "%(asctime)s - %(name)s - %(levelname)s - %(message)s" + reset
            }
            
            def format(self, record):
                log_fmt = self.FORMATS.get(record.levelno)
                formatter = logging.Formatter(log_fmt, datefmt="%Y-%m-%d %H:%M:%S")
                return formatter.format(record)
        
        console_handler.setFormatter(ColoredFormatter())
        self.logger.addHandler(console_handler)

    def _add_file_handler(self):
        """Add rotating file handler."""
        file_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "ml_project.log",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        file_handler.setLevel(logging.DEBUG)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        file_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)

    def _add_json_handler(self):
        """Add JSON file handler for structured logging."""
        json_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "ml_project.json",
            maxBytes=10 * 1024 * 1024,  # 10MB
            backupCount=5,
            encoding="utf-8"
        )
        json_handler.setLevel(logging.DEBUG)
        
        # Custom JSON formatter
        class CustomJsonFormatter(jsonlogger.JsonFormatter):
            def add_fields(self, log_record, record, message_dict):
                super().add_fields(log_record, record, message_dict)
                log_record["timestamp"] = datetime.utcnow().isoformat()
                log_record["level"] = record.levelname
                log_record["logger_name"] = record.name
                log_record["function"] = record.funcName
                log_record["line_number"] = record.lineno
                
                # Add custom fields if present
                if hasattr(record, "experiment_id"):
                    log_record["experiment_id"] = record.experiment_id
                if hasattr(record, "model_name"):
                    log_record["model_name"] = record.model_name
                if hasattr(record, "metrics"):
                    log_record["metrics"] = record.metrics
        
        json_formatter = CustomJsonFormatter()
        json_handler.setFormatter(json_formatter)
        self.logger.addHandler(json_handler)

    def _add_error_handler(self):
        """Add separate error log handler."""
        error_handler = logging.handlers.RotatingFileHandler(
            filename=self.log_dir / "errors.log",
            maxBytes=5 * 1024 * 1024,  # 5MB
            backupCount=3,
            encoding="utf-8"
        )
        error_handler.setLevel(logging.ERROR)
        
        formatter = logging.Formatter(
            "%(asctime)s - %(name)s - %(levelname)s - %(funcName)s:%(lineno)d - %(message)s\n"
            "Exception: %(exc_info)s\n",
            datefmt="%Y-%m-%d %H:%M:%S"
        )
        error_handler.setFormatter(formatter)
        self.logger.addHandler(error_handler)

    def get_logger(self, name: Optional[str] = None) -> logging.Logger:
        """Get a logger instance.
        
        Args:
            name: Logger name (module name)
            
        Returns:
            Logger instance
        """
        if name:
            return self.logger.getChild(name)
        return self.logger


class MetricsLogger:
    """Logger specifically for ML metrics and performance tracking."""
    
    def __init__(self, experiment_name: str, run_id: Optional[str] = None):
        """Initialize metrics logger.
        
        Args:
            experiment_name: Name of the experiment
            run_id: Optional MLflow run ID
        """
        self.experiment_name = experiment_name
        self.run_id = run_id or datetime.now().strftime("%Y%m%d_%H%M%S")
        self.metrics_file = Path("logs/metrics") / f"{experiment_name}_{self.run_id}.json"
        self.metrics_file.parent.mkdir(parents=True, exist_ok=True)
        self.logger = MLProjectLogger().get_logger("metrics")
        
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        """Log metrics to file and logger.
        
        Args:
            metrics: Dictionary of metrics
            step: Optional step/epoch number
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "experiment": self.experiment_name,
            "run_id": self.run_id,
            "step": step,
            "metrics": metrics
        }
        
        # Append to metrics file
        with open(self.metrics_file, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")
        
        # Log to main logger
        self.logger.info(
            f"Metrics logged for {self.experiment_name}",
            extra={"metrics": metrics, "experiment_id": self.run_id}
        )
    
    def log_hyperparameters(self, params: Dict[str, Any]):
        """Log hyperparameters.
        
        Args:
            params: Dictionary of hyperparameters
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "experiment": self.experiment_name,
            "run_id": self.run_id,
            "hyperparameters": params
        }
        
        with open(self.metrics_file, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")
    
    def log_model_info(self, model_info: Dict[str, Any]):
        """Log model information.
        
        Args:
            model_info: Dictionary containing model details
        """
        log_entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "experiment": self.experiment_name,
            "run_id": self.run_id,
            "model_info": model_info
        }
        
        with open(self.metrics_file, "a") as f:
            json.dump(log_entry, f)
            f.write("\n")


def setup_logging(
    level: int = logging.INFO,
    log_dir: Optional[str] = None,
    config_file: Optional[str] = None
) -> logging.Logger:
    """Set up logging configuration.
    
    Args:
        level: Logging level
        log_dir: Custom log directory
        config_file: Optional YAML config file
        
    Returns:
        Configured logger instance
    """
    # Load config from file if provided
    if config_file and Path(config_file).exists():
        with open(config_file, "r") as f:
            config = yaml.safe_load(f)
            level = getattr(logging, config.get("level", "INFO").upper())
            log_dir = config.get("log_dir", log_dir)
    
    # Update log directory if provided
    ml_logger = MLProjectLogger()
    if log_dir:
        ml_logger.log_dir = Path(log_dir)
        ml_logger.log_dir.mkdir(exist_ok=True)
        ml_logger._setup_logger()  # Reinitialize with new directory
    
    # Set level
    ml_logger.logger.setLevel(level)
    
    return ml_logger.get_logger()


def get_logger(name: str) -> logging.Logger:
    """Get a logger instance for a specific module.
    
    Args:
        name: Module name (usually __name__)
        
    Returns:
        Logger instance
    """
    return MLProjectLogger().get_logger(name)


# Performance logging decorators
def log_execution_time(func):
    """Decorator to log function execution time."""
    import functools
    import time
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        start_time = time.time()
        
        try:
            result = func(*args, **kwargs)
            execution_time = time.time() - start_time
            logger.info(
                f"{func.__name__} completed in {execution_time:.2f} seconds",
                extra={"execution_time": execution_time, "function": func.__name__}
            )
            return result
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(
                f"{func.__name__} failed after {execution_time:.2f} seconds: {str(e)}",
                exc_info=True,
                extra={"execution_time": execution_time, "function": func.__name__}
            )
            raise
    
    return wrapper


def log_memory_usage(func):
    """Decorator to log memory usage."""
    import functools
    import psutil
    
    @functools.wraps(func)
    def wrapper(*args, **kwargs):
        logger = get_logger(func.__module__)
        process = psutil.Process()
        
        # Memory before
        mem_before = process.memory_info().rss / 1024 / 1024  # MB
        
        result = func(*args, **kwargs)
        
        # Memory after
        mem_after = process.memory_info().rss / 1024 / 1024  # MB
        mem_used = mem_after - mem_before
        
        logger.info(
            f"{func.__name__} memory usage: {mem_used:.2f} MB "
            f"(before: {mem_before:.2f} MB, after: {mem_after:.2f} MB)",
            extra={
                "memory_before_mb": mem_before,
                "memory_after_mb": mem_after,
                "memory_used_mb": mem_used,
                "function": func.__name__
            }
        )
        
        return result
    
    return wrapper