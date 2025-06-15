# Data Configuration for SMS Spam Filter Project
# Created: 15/06/2025
# Task: DE-001 Environment Setup

import os
from pathlib import Path
from datetime import datetime

# Project Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
NOTEBOOKS_DIR = PROJECT_ROOT / "notebooks"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Dataset Configuration
DATASET_CONFIG = {
    "raw_data_path": DATA_DIR / "SMSSPamCollection",
    "processed_data_path": DATA_DIR / "processed",
    "train_split": 0.8,
    "validation_split": 0.1,
    "test_split": 0.1,
    "random_seed": 42,
    "expected_message_count": 5574,
    "format": "tab_separated",
    "encoding": "utf-8"
}

# Model Performance Targets
PERFORMANCE_TARGETS = {
    "precision": 0.92,
    "recall": 0.88,
    "f1_score": 0.90,
    "inference_time_ms": 50
}

# Data Validation Rules
DATA_VALIDATION = {
    "required_columns": ["label", "message"],
    "allowed_labels": ["ham", "spam"],
    "min_message_length": 1,
    "max_message_length": 1000,
    "encoding_validation": True
}

# Pipeline Configuration
PIPELINE_CONFIG = {
    "preprocessing_steps": [
        "text_cleaning",
        "tokenization", 
        "feature_extraction"
    ],
    "feature_extraction": {
        "tfidf_max_features": 5000,
        "ngram_range": (1, 2),
        "min_df": 2,
        "max_df": 0.95
    },
    "batch_size": 1000,
    "parallel_jobs": -1
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "file_path": LOGS_DIR / f"pipeline_{datetime.now().strftime('%Y%m%d')}.log"
}

# Create directories if they don't exist
for directory in [MODELS_DIR, LOGS_DIR, DATA_DIR / "processed"]:
    directory.mkdir(parents=True, exist_ok=True) 