#!/usr/bin/env python3
"""
COLLAB-001: Ensemble Serving Infrastructure
Date: 16/06/2025 07:41
Engineer: AI Data Engineer
Purpose: Production-ready ensemble serving infrastructure with multi-model support
Phase: Day 1 - Infrastructure Foundation
"""

import os
import sys
import json
import time
import asyncio
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, asdict
from concurrent.futures import ThreadPoolExecutor
import threading
import queue
import logging

# Data processing
import pandas as pd
import numpy as np
import joblib

# FastAPI for serving
from fastapi import FastAPI, HTTPException, Request, BackgroundTasks
from fastapi.responses import JSONResponse
from pydantic import BaseModel
import uvicorn

# Monitoring and performance
import psutil
from collections import defaultdict, deque
import statistics

print("🚀 COLLAB-001: ENSEMBLE SERVING INFRASTRUCTURE")
print("=" * 70)
print(f"📅 Started: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print(f"🎯 Building production-ready ensemble serving platform")
print(f"⚡ Multi-model architecture with confidence scoring")
print("=" * 70)
print()

# Request/Response Models
class PredictionRequest(BaseModel):
    text: str
    model_preference: Optional[str] = None
    confidence_threshold: Optional[float] = 0.8
    return_all_predictions: Optional[bool] = False

class ModelPrediction(BaseModel):
    model_name: str
    prediction: int  # 0 for ham, 1 for spam
    confidence: float
    inference_time_ms: float

class EnsemblePrediction(BaseModel):
    final_prediction: int
    final_confidence: float
    ensemble_method: str
    agreement_score: float
    individual_predictions: List[ModelPrediction]
    total_inference_time_ms: float
    models_used: List[str]

@dataclass
class ModelMetrics:
    """Metrics tracking for individual models"""
    model_name: str
    total_requests: int = 0
    total_inference_time_ms: float = 0.0
    avg_inference_time_ms: float = 0.0
    p95_inference_time_ms: float = 0.0
    error_count: int = 0
    error_rate: float = 0.0
    last_used: Optional[datetime] = None
    
    def update_timing(self, inference_time_ms: float):
        self.total_requests += 1
        self.total_inference_time_ms += inference_time_ms
        self.avg_inference_time_ms = self.total_inference_time_ms / self.total_requests
        self.last_used = datetime.now()

@dataclass
class EnsembleMetrics:
    """Metrics tracking for ensemble operations"""
    total_requests: int = 0
    total_ensemble_time_ms: float = 0.0
    avg_ensemble_time_ms: float = 0.0
    high_agreement_count: int = 0
    low_agreement_count: int = 0
    avg_agreement_score: float = 0.0
    fallback_usage_count: int = 0

class ModelManager:
    """Manages loading and inference for individual models"""
    
    def __init__(self):
        self.models = {}
        self.vectorizers = {}
        self.model_configs = {}
        self.model_metrics = {}
        self.models_dir = Path("models")
        
        # Performance tracking
        self.inference_times = defaultdict(lambda: deque(maxlen=1000))
        
        print("🤖 Model Manager initialized")
    
    def load_available_models(self):
        """Load all available models for ensemble serving"""
        print("📦 LOADING AVAILABLE MODELS FOR SERVING")
        print("=" * 50)
        
        model_configs = {
            "lightgbm": {
                "model_file": "lightgbm_optimized_v1.0.0_16062025_072128.joblib",
                "vectorizer_file": "lightgbm_vectorizer_v1.0.0.joblib",
                "performance": 89.93,
                "type": "sklearn"
            },
            "xgboost": {
                "model_file": "xgboost_advanced_v1.0.0_15062025_192505.joblib",
                "vectorizer_file": "xgboost_vectorizer_v1.0.0.joblib",
                "performance": 89.04,
                "type": "sklearn"
            }
        }
        
        loaded_count = 0
        for model_name, config in model_configs.items():
            try:
                model_path = self.models_dir / config["model_file"]
                vectorizer_path = self.models_dir / config["vectorizer_file"]
                
                if model_path.exists() and vectorizer_path.exists():
                    # Load model and vectorizer
                    model = joblib.load(model_path)
                    vectorizer = joblib.load(vectorizer_path)
                    
                    self.models[model_name] = model
                    self.vectorizers[model_name] = vectorizer
                    self.model_configs[model_name] = config
                    self.model_metrics[model_name] = ModelMetrics(model_name)
                    
                    loaded_count += 1
                    print(f"  ✅ {model_name.upper()}: Loaded ({config['performance']:.2f}% F1)")
                else:
                    print(f"  ❌ {model_name.upper()}: Files not found")
                    
            except Exception as e:
                print(f"  ❌ {model_name.upper()}: Load failed - {e}")
        
        print(f"\n🎯 {loaded_count} models loaded for serving")
        
        # Load ensemble models if available
        self._load_ensemble_models()
        
        return loaded_count > 0
    
    def _load_ensemble_models(self):
        """Load pre-trained ensemble models"""
        ensemble_dir = Path("ensemble_models")
        if ensemble_dir.exists():
            ensemble_files = list(ensemble_dir.glob("*.joblib"))
            for ensemble_file in ensemble_files:
                try:
                    ensemble_name = ensemble_file.stem
                    ensemble_model = joblib.load(ensemble_file)
                    
                    self.models[ensemble_name] = ensemble_model
                    self.model_configs[ensemble_name] = {
                        "type": "ensemble",
                        "performance": 90.0  # Placeholder
                    }
                    self.model_metrics[ensemble_name] = ModelMetrics(ensemble_name)
                    
                    print(f"  🤝 {ensemble_name.upper()}: Ensemble loaded")
                except Exception as e:
                    print(f"  ❌ {ensemble_file.name}: Ensemble load failed - {e}")
    
    def predict_single_model(self, model_name: str, text: str) -> Tuple[int, float, float]:
        """Make prediction with a single model"""
        if model_name not in self.models:
            raise ValueError(f"Model {model_name} not available")
        
        start_time = time.time()
        
        try:
            # Transform text
            if model_name in self.vectorizers:
                X_transformed = self.vectorizers[model_name].transform([text])
            else:
                raise ValueError(f"No vectorizer available for {model_name}")
            
            # Make prediction
            model = self.models[model_name]
            prediction = model.predict(X_transformed)[0]
            
            # Get prediction probability for confidence
            if hasattr(model, 'predict_proba'):
                proba = model.predict_proba(X_transformed)[0]
                confidence = float(max(proba))
            else:
                confidence = 0.8  # Default confidence for models without probability
            
            inference_time_ms = (time.time() - start_time) * 1000
            
            # Update metrics
            self.model_metrics[model_name].update_timing(inference_time_ms)
            self.inference_times[model_name].append(inference_time_ms)
            
            return int(prediction), confidence, inference_time_ms
            
        except Exception as e:
            self.model_metrics[model_name].error_count += 1
            self.model_metrics[model_name].error_rate = (
                self.model_metrics[model_name].error_count / 
                max(self.model_metrics[model_name].total_requests, 1)
            )
            raise Exception(f"Prediction failed for {model_name}: {e}")
    
    def get_model_health(self) -> Dict[str, Any]:
        """Get health status of all models"""
        health_status = {}
        
        for model_name, metrics in self.model_metrics.items():
            recent_times = list(self.inference_times[model_name])
            
            health_status[model_name] = {
                "status": "healthy" if metrics.error_rate < 0.05 else "degraded",
                "total_requests": metrics.total_requests,
                "avg_inference_ms": metrics.avg_inference_time_ms,
                "p95_inference_ms": np.percentile(recent_times, 95) if recent_times else 0,
                "error_rate": metrics.error_rate,
                "last_used": metrics.last_used.isoformat() if metrics.last_used else None
            }
        
        return health_status

class EnsembleEngine:
    """Core ensemble prediction engine with multiple strategies"""
    
    def __init__(self, model_manager: ModelManager):
        self.model_manager = model_manager
        self.ensemble_metrics = EnsembleMetrics()
        
        # Ensemble strategies
        self.strategies = {
            "simple_voting": self._simple_voting,
            "weighted_voting": self._weighted_voting,
            "confidence_weighted": self._confidence_weighted_voting,
            "best_model_fallback": self._best_model_fallback
        }
        
        print("🤖 Ensemble Engine initialized with strategies:")
        for strategy in self.strategies.keys():
            print(f"  📊 {strategy}")
    
    def predict_ensemble(self, text: str, strategy: str = "confidence_weighted", 
                        models_to_use: Optional[List[str]] = None) -> EnsemblePrediction:
        """Make ensemble prediction using specified strategy"""
        start_time = time.time()
        
        # Determine which models to use
        if models_to_use is None:
            models_to_use = [name for name in self.model_manager.models.keys() 
                           if self.model_manager.model_configs[name]["type"] != "ensemble"]
        
        # Get predictions from individual models
        individual_predictions = []
        successful_predictions = []
        
        for model_name in models_to_use:
            try:
                prediction, confidence, inference_time = (
                    self.model_manager.predict_single_model(model_name, text)
                )
                
                model_pred = ModelPrediction(
                    model_name=model_name,
                    prediction=prediction,
                    confidence=confidence,
                    inference_time_ms=inference_time
                )
                
                individual_predictions.append(model_pred)
                successful_predictions.append((prediction, confidence))
                
            except Exception as e:
                print(f"⚠️ Model {model_name} failed: {e}")
                continue
        
        if not successful_predictions:
            raise Exception("No models produced successful predictions")
        
        # Apply ensemble strategy
        if strategy not in self.strategies:
            strategy = "confidence_weighted"  # Fallback
        
        final_prediction, final_confidence, agreement_score = (
            self.strategies[strategy](successful_predictions)
        )
        
        total_time_ms = (time.time() - start_time) * 1000
        
        # Update ensemble metrics
        self.ensemble_metrics.total_requests += 1
        self.ensemble_metrics.total_ensemble_time_ms += total_time_ms
        self.ensemble_metrics.avg_ensemble_time_ms = (
            self.ensemble_metrics.total_ensemble_time_ms / 
            self.ensemble_metrics.total_requests
        )
        
        if agreement_score > 0.8:
            self.ensemble_metrics.high_agreement_count += 1
        else:
            self.ensemble_metrics.low_agreement_count += 1
        
        return EnsemblePrediction(
            final_prediction=final_prediction,
            final_confidence=final_confidence,
            ensemble_method=strategy,
            agreement_score=agreement_score,
            individual_predictions=individual_predictions,
            total_inference_time_ms=total_time_ms,
            models_used=models_to_use
        )
    
    def _simple_voting(self, predictions: List[Tuple[int, float]]) -> Tuple[int, float, float]:
        """Simple majority voting"""
        votes = [pred[0] for pred in predictions]
        spam_votes = sum(votes)
        
        if spam_votes > len(votes) / 2:
            final_pred = 1
            agreement = spam_votes / len(votes)
        else:
            final_pred = 0
            agreement = (len(votes) - spam_votes) / len(votes)
        
        avg_confidence = np.mean([pred[1] for pred in predictions])
        return final_pred, float(avg_confidence), float(agreement)
    
    def _weighted_voting(self, predictions: List[Tuple[int, float]]) -> Tuple[int, float, float]:
        """Weighted voting based on model performance"""
        # Simple implementation - can be enhanced with actual model weights
        model_weights = [0.89, 0.85]  # Example weights based on F1 scores
        
        if len(predictions) != len(model_weights):
            return self._simple_voting(predictions)
        
        weighted_sum = sum(pred[0] * weight for pred, weight in zip(predictions, model_weights))
        total_weight = sum(model_weights)
        
        final_pred = 1 if weighted_sum > total_weight / 2 else 0
        
        # Calculate agreement based on weighted consensus
        agreement = abs(weighted_sum - total_weight/2) / (total_weight/2)
        agreement = min(agreement, 1.0)
        
        avg_confidence = np.mean([pred[1] for pred in predictions])
        return final_pred, float(avg_confidence), float(agreement)
    
    def _confidence_weighted_voting(self, predictions: List[Tuple[int, float]]) -> Tuple[int, float, float]:
        """Voting weighted by prediction confidence"""
        if not predictions:
            return 0, 0.0, 0.0
        
        # Weight votes by confidence
        spam_confidence = sum(pred[1] for pred in predictions if pred[0] == 1)
        ham_confidence = sum(pred[1] for pred in predictions if pred[0] == 0)
        
        total_confidence = spam_confidence + ham_confidence
        
        if total_confidence == 0:
            return self._simple_voting(predictions)
        
        spam_weight = spam_confidence / total_confidence
        
        final_pred = 1 if spam_weight > 0.5 else 0
        final_confidence = max(spam_weight, 1 - spam_weight)
        
        # Agreement based on confidence distribution
        agreement = 2 * abs(spam_weight - 0.5)  # 0 when 50/50, 1 when unanimous
        
        return final_pred, float(final_confidence), float(agreement)
    
    def _best_model_fallback(self, predictions: List[Tuple[int, float]]) -> Tuple[int, float, float]:
        """Use best performing model as primary, others as validation"""
        if not predictions:
            return 0, 0.0, 0.0
        
        # Use first prediction as best model (assumes sorted by performance)
        best_pred, best_conf = predictions[0]
        
        # Check agreement with other models
        if len(predictions) > 1:
            other_votes = [pred[0] for pred in predictions[1:]]
            agreement_count = sum(1 for vote in other_votes if vote == best_pred)
            agreement = agreement_count / len(other_votes)
        else:
            agreement = 1.0
        
        return best_pred, best_conf, float(agreement)

# FastAPI Application
app = FastAPI(title="Ensemble Spam Filter API", version="1.0.0")

# Global instances
model_manager = ModelManager()
ensemble_engine = None

@app.on_event("startup")
async def startup_event():
    """Initialize serving infrastructure on startup"""
    global ensemble_engine
    
    print("🚀 Starting Ensemble Serving Infrastructure")
    
    if not model_manager.load_available_models():
        raise Exception("Failed to load models")
    
    ensemble_engine = EnsembleEngine(model_manager)
    
    print("✅ Ensemble serving infrastructure ready!")

@app.post("/predict", response_model=EnsemblePrediction)
async def predict(request: PredictionRequest):
    """Main prediction endpoint"""
    if not ensemble_engine:
        raise HTTPException(status_code=503, detail="Ensemble engine not initialized")
    
    try:
        result = ensemble_engine.predict_ensemble(
            text=request.text,
            strategy="confidence_weighted"
        )
        return result
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    model_health = model_manager.get_model_health()
    
    return {
        "status": "healthy",
        "timestamp": datetime.now().isoformat(),
        "models": model_health,
        "ensemble_metrics": {
            "total_requests": ensemble_engine.ensemble_metrics.total_requests if ensemble_engine else 0,
            "avg_ensemble_time_ms": ensemble_engine.ensemble_metrics.avg_ensemble_time_ms if ensemble_engine else 0
        }
    }

@app.get("/metrics")
async def get_metrics():
    """Detailed metrics endpoint"""
    if not ensemble_engine:
        return {"status": "not_ready"}
    
    return {
        "model_metrics": {
            name: asdict(metrics) for name, metrics in model_manager.model_metrics.items()
        },
        "ensemble_metrics": asdict(ensemble_engine.ensemble_metrics),
        "system_metrics": {
            "cpu_percent": psutil.cpu_percent(),
            "memory_percent": psutil.virtual_memory().percent,
            "timestamp": datetime.now().isoformat()
        }
    }

def main():
    """Main function to start the ensemble serving infrastructure"""
    print("🚀 LAUNCHING COLLAB-001 ENSEMBLE SERVING INFRASTRUCTURE")
    print("=" * 70)
    
    # Configuration
    config = {
        "host": "0.0.0.0",
        "port": 8000,
        "workers": 1,
        "reload": False,
        "log_level": "info"
    }
    
    print(f"📡 Starting server on {config['host']}:{config['port']}")
    print("🎯 Endpoints available:")
    print("  📊 POST /predict - Main ensemble prediction")
    print("  🏥 GET /health - Health check")
    print("  📈 GET /metrics - Detailed metrics")
    print()
    
    # Start server
    uvicorn.run(app, **config)

if __name__ == "__main__":
    main()