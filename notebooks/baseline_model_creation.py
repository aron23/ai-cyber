#!/usr/bin/env python3
"""
DE-INT-001: Baseline Model Integration Testing
Date: 15/06/2025 18:36
Engineer: AI Data Engineer
"""

import os
import sys
import time
import json
import warnings
from datetime import datetime
from pathlib import Path

# Data processing
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer

# Baseline Models
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Model evaluation
from sklearn.metrics import classification_report, accuracy_score
from sklearn.metrics import precision_recall_fscore_support, roc_auc_score

# Class imbalance handling
from sklearn.utils.class_weight import compute_class_weight

# Performance monitoring
import psutil
import joblib

# Suppress warnings
warnings.filterwarnings('ignore')

def print_header():
    """Print execution header"""
    print("🚀 DE-INT-001: BASELINE MODEL INTEGRATION TESTING")
    print("=" * 60)
    print(f"📅 Started: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
    print(f"🎯 Task: Baseline Model Integration with Serving Infrastructure")
    print(f"⚡ Phase: Creating DS-004 Baseline Models + Integration Testing")
    print("=" * 60)
    print()

def load_data():
    """Load and analyze SMS dataset"""
    print("📊 LOADING SMS SPAM DATASET")
    print("=" * 40)
    
    # Load the dataset
    data_path = '../data/SMSSPamCollection'
    df = pd.read_csv(data_path, sep='\t', header=None, names=['label', 'message'])
    
    print(f"✅ Dataset loaded successfully")
    print(f"📏 Total messages: {len(df):,}")
    print(f"📋 Columns: {list(df.columns)}")
    print()
    
    # Analyze class distribution
    class_counts = df['label'].value_counts()
    class_percentages = df['label'].value_counts(normalize=True) * 100
    
    print("📊 CLASS DISTRIBUTION ANALYSIS")
    print("=" * 40)
    print(f"Ham Messages: {class_counts['ham']:,} ({class_percentages['ham']:.2f}%)")
    print(f"Spam Messages: {class_counts['spam']:,} ({class_percentages['spam']:.2f}%)")
    
    # Calculate imbalance ratio
    imbalance_ratio = class_counts['ham'] / class_counts['spam']
    print(f"\n⚠️ Imbalance Ratio (Ham:Spam): {imbalance_ratio:.1f}:1")
    print(f"🚨 Class imbalance strategy required!")
    print()
    
    return df

def prepare_features(df):
    """Prepare features using TF-IDF"""
    print("🔧 FEATURE ENGINEERING PIPELINE")
    print("=" * 40)
    
    # Text preprocessing function
    def preprocess_text(text):
        """Basic text preprocessing"""
        if pd.isna(text):
            return ""
        return text.lower().strip()
    
    # Apply preprocessing
    df['processed_message'] = df['message'].apply(preprocess_text)
    print("✅ Text preprocessing completed")
    
    # Configure TF-IDF vectorizer
    tfidf_config = {
        'max_features': 5000,
        'ngram_range': (1, 2),
        'min_df': 2,
        'max_df': 0.95,
        'stop_words': 'english',
        'lowercase': True,
        'strip_accents': 'unicode'
    }
    
    print(f"⚙️ TF-IDF Configuration:")
    for key, value in tfidf_config.items():
        print(f"  {key}: {value}")
    print()
    
    # Initialize vectorizer
    vectorizer = TfidfVectorizer(**tfidf_config)
    
    return df, vectorizer

def split_data(df):
    """Split data with stratification"""
    print("📊 STRATIFIED DATA SPLITTING")
    print("=" * 40)
    
    # Prepare features and target
    X = df['processed_message']
    y = df['label']
    
    # Split data with stratification
    X_train, X_temp, y_train, y_temp = train_test_split(
        X, y, test_size=0.3, random_state=42, stratify=y
    )
    
    X_val, X_test, y_val, y_test = train_test_split(
        X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
    )
    
    print(f"✅ Data splitting completed")
    print(f"📊 Training set: {len(X_train):,} samples ({len(X_train)/len(df)*100:.1f}%)")
    print(f"📊 Validation set: {len(X_val):,} samples ({len(X_val)/len(df)*100:.1f}%)")
    print(f"📊 Test set: {len(X_test):,} samples ({len(X_test)/len(df)*100:.1f}%)")
    print()
    
    return X_train, X_val, X_test, y_train, y_val, y_test

def extract_features(vectorizer, X_train, X_val, X_test):
    """Extract TF-IDF features"""
    print("🔢 FEATURE EXTRACTION & CLASS BALANCING")
    print("=" * 40)
    
    # Fit TF-IDF on training data
    start_time = time.time()
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_val_tfidf = vectorizer.transform(X_val)
    X_test_tfidf = vectorizer.transform(X_test)
    feature_time = time.time() - start_time
    
    print(f"✅ TF-IDF features extracted ({feature_time:.2f}s)")
    print(f"📊 Feature matrix shape: {X_train_tfidf.shape}")
    print(f"🎯 Features created: {len(vectorizer.get_feature_names_out()):,}")
    print(f"💾 Matrix sparsity: {(1 - X_train_tfidf.nnz / (X_train_tfidf.shape[0] * X_train_tfidf.shape[1])):.1%}")
    print()
    
    return X_train_tfidf, X_val_tfidf, X_test_tfidf

def train_baseline_models(X_train_tfidf, X_val_tfidf, y_train, y_val):
    """Train and evaluate baseline models"""
    print("🤖 DS-004 BASELINE MODEL TRAINING")
    print("=" * 50)
    
    # Define baseline models with optimal configurations
    baseline_models = {
        'naive_bayes': {
            'model': MultinomialNB(alpha=1.0),
            'description': 'Multinomial Naive Bayes - Excellent for text classification'
        },
        'svm': {
            'model': LinearSVC(
                C=1.0, 
                class_weight='balanced',
                random_state=42,
                max_iter=2000
            ),
            'description': 'Linear Support Vector Machine - Strong for high-dimensional data'
        },
        'logistic_regression': {
            'model': LogisticRegression(
                C=1.0,
                class_weight='balanced',
                solver='liblinear',
                random_state=42,
                max_iter=2000
            ),
            'description': 'Logistic Regression - Fast, interpretable baseline'
        },
        'random_forest': {
            'model': RandomForestClassifier(
                n_estimators=100,
                class_weight='balanced',
                random_state=42,
                n_jobs=-1,
                max_depth=10
            ),
            'description': 'Random Forest - Robust ensemble method'
        }
    }
    
    print(f"📋 Baseline models configured: {len(baseline_models)}")
    for name, config in baseline_models.items():
        model_name = config['model'].__class__.__name__
        print(f"  🔸 {name.title()}: {model_name}")
        print(f"    {config['description']}")
    print()
    
    # Initialize results storage
    training_results = {}
    performance_metrics = {}
    
    total_training_start = time.time()
    
    for model_name, config in baseline_models.items():
        print(f"\n🎯 Training {model_name.upper()}")
        print("-" * 30)
        
        # Record memory usage before training
        memory_before = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        
        # Train model
        start_time = time.time()
        model = config['model']
        model.fit(X_train_tfidf, y_train)
        training_time = time.time() - start_time
        
        memory_after = psutil.Process().memory_info().rss / 1024 / 1024  # MB
        memory_used = memory_after - memory_before
        
        print(f"✅ Training completed in {training_time:.2f} seconds")
        print(f"💾 Memory used: {memory_used:.1f} MB")
        
        # Make predictions
        pred_start = time.time()
        y_val_pred = model.predict(X_val_tfidf)
        prediction_time = time.time() - pred_start
        avg_prediction_time_ms = (prediction_time / len(y_val)) * 1000
        
        print(f"⚡ Validation prediction time: {prediction_time:.3f}s")
        print(f"📊 Average per message: {avg_prediction_time_ms:.2f}ms")
        
        # Calculate performance metrics
        accuracy = accuracy_score(y_val, y_val_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_val_pred, average='weighted'
        )
        
        # Try to get probability predictions for AUC (if supported)
        try:
            if hasattr(model, 'predict_proba'):
                y_val_proba = model.predict_proba(X_val_tfidf)[:, 1]  # Probability of spam
                # Convert labels to binary for AUC calculation
                y_val_binary = (y_val == 'spam').astype(int)
                auc_score = roc_auc_score(y_val_binary, y_val_proba)
            else:
                # For SVM, use decision function
                y_val_scores = model.decision_function(X_val_tfidf)
                y_val_binary = (y_val == 'spam').astype(int)
                auc_score = roc_auc_score(y_val_binary, y_val_scores)
        except Exception as e:
            auc_score = None
            print(f"⚠️ AUC calculation failed: {str(e)}")
        
        print(f"📈 Performance Metrics:")
        print(f"  Accuracy: {accuracy:.4f} ({accuracy*100:.2f}%)")
        print(f"  Precision: {precision:.4f} ({precision*100:.2f}%)")
        print(f"  Recall: {recall:.4f} ({recall*100:.2f}%)")
        print(f"  F1-Score: {f1:.4f} ({f1*100:.2f}%)")
        if auc_score:
            print(f"  ROC-AUC: {auc_score:.4f} ({auc_score*100:.2f}%)")
        
        # Store results
        training_results[model_name] = {
            'model': model,
            'training_time_seconds': training_time,
            'memory_used_mb': memory_used,
            'prediction_time_seconds': prediction_time,
            'avg_prediction_time_ms': avg_prediction_time_ms,
            'validation_samples': len(y_val)
        }
        
        performance_metrics[model_name] = {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': auc_score,
            'avg_inference_time_ms': avg_prediction_time_ms
        }
        
        # Check if meets performance targets
        targets_met = {
            'f1_target': f1 >= 0.85,  # Reasonable baseline target
            'inference_target': avg_prediction_time_ms < 50,  # <50ms target
            'precision_target': precision >= 0.80  # Good spam detection
        }
        
        print(f"🎯 Target Achievement:")
        for target, met in targets_met.items():
            status = "✅" if met else "❌"
            print(f"  {status} {target.replace('_', ' ').title()}: {met}")
    
    total_training_time = time.time() - total_training_start
    
    print(f"\n🏆 BASELINE MODEL TRAINING COMPLETED")
    print("=" * 50)
    print(f"⏱️ Total training time: {total_training_time:.2f} seconds")
    print(f"🤖 Models trained: {len(training_results)}")
    print(f"✅ All baseline models ready for integration testing")
    print()
    
    return training_results, performance_metrics

def save_models(training_results, vectorizer):
    """Save models for integration testing"""
    print("💾 SAVING MODELS FOR INTEGRATION TESTING")
    print("=" * 50)
    
    # Create models directory
    models_dir = Path('../models')
    models_dir.mkdir(exist_ok=True)
    
    saved_models = {}
    
    # Save vectorizer first
    vectorizer_path = models_dir / 'tfidf_vectorizer_v1.0.0.joblib'
    joblib.dump(vectorizer, vectorizer_path)
    print(f"✅ TF-IDF Vectorizer saved: {vectorizer_path}")
    
    # Save vectorizer metadata
    vectorizer_metadata = {
        'component': 'tfidf_vectorizer',
        'version': '1.0.0',
        'config': {
            'max_features': 5000,
            'ngram_range': [1, 2],
            'min_df': 2,
            'max_df': 0.95
        },
        'created_at': datetime.now().isoformat(),
        'feature_count': len(vectorizer.get_feature_names_out())
    }
    
    vectorizer_metadata_path = models_dir / 'tfidf_vectorizer_v1.0.0_metadata.json'
    with open(vectorizer_metadata_path, 'w') as f:
        json.dump(vectorizer_metadata, f, indent=2)
    
    # Save each trained model
    for model_name, results in training_results.items():
        model = results['model']
        model_filename = f"{model_name}_baseline_v1.0.0.joblib"
        model_path = models_dir / model_filename
        
        # Save model
        joblib.dump(model, model_path)
        file_size_mb = model_path.stat().st_size / (1024 * 1024)
        
        print(f"✅ {model_name.title()} saved: {model_path} ({file_size_mb:.2f}MB)")
        
        # Save model metadata
        model_metadata = {
            'model_id': f"{model_name}_baseline_v1.0.0",
            'model_name': model_name,
            'model_type': model.__class__.__name__,
            'version': '1.0.0',
            'format': 'joblib',
            'file_path': str(model_path),
            'file_size_mb': file_size_mb,
            'created_at': datetime.now().isoformat(),
            'training_time_seconds': results['training_time_seconds'],
            'memory_used_mb': results['memory_used_mb'],
            'avg_inference_time_ms': results['avg_prediction_time_ms'],
            'validation_samples': results['validation_samples'],
            'requires_vectorizer': 'tfidf_vectorizer_v1.0.0'
        }
        
        metadata_path = models_dir / f"{model_name}_baseline_v1.0.0_metadata.json"
        with open(metadata_path, 'w') as f:
            json.dump(model_metadata, f, indent=2)
        
        saved_models[model_name] = {
            'model_path': model_path,
            'metadata_path': metadata_path,
            'model_id': model_metadata['model_id']
        }
    
    print(f"\n✅ All models saved successfully!")
    print(f"📁 Models directory: {models_dir.absolute()}")
    print(f"🎯 Ready for integration testing with serving infrastructure")
    
    return saved_models

def main():
    """Main execution function"""
    print_header()
    
    # Phase 1: Data preparation
    df = load_data()
    df, vectorizer = prepare_features(df)
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(df)
    X_train_tfidf, X_val_tfidf, X_test_tfidf = extract_features(
        vectorizer, X_train, X_val, X_test
    )
    
    # Phase 2: Model training
    training_results, performance_metrics = train_baseline_models(
        X_train_tfidf, X_val_tfidf, y_train, y_val
    )
    
    # Phase 3: Model saving
    saved_models = save_models(training_results, vectorizer)
    
    print("🎉 DE-INT-001 PHASE 1 COMPLETED SUCCESSFULLY!")
    print("=" * 50)
    print(f"✅ DS-004 Baseline Models Created: {len(saved_models)}")
    print(f"💾 Models Saved and Ready for Integration Testing")
    print(f"⚡ Next Phase: Integration with Model Serving Infrastructure")
    print()
    
    return saved_models, performance_metrics

if __name__ == "__main__":
    saved_models, performance_metrics = main() 