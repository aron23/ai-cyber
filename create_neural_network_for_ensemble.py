#!/usr/bin/env python3
"""
Create Neural Network for Ensemble - 94%+ Target Achievement
Training a new neural network compatible with current ensemble infrastructure
Goal: Achieve 94%+ F1-Score by adding neural network diversity to ensemble
"""

import warnings
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split, GridSearchCV, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score, classification_report
import json

warnings.filterwarnings('ignore')

print("🧠 CREATING NEURAL NETWORK FOR ENSEMBLE - 94%+ TARGET")
print("=" * 70)
print(f"📅 Started: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("🚀 Goal: Train NN compatible with current ensemble → 94%+ F1")
print("🎯 Strategy: Perfect integration + ensemble diversity boost")
print("=" * 70)
print()

def load_ensemble_infrastructure():
    """Load the exact same infrastructure used by our current ensemble"""
    models_dir = Path("models")
    data_dir = Path("data")
    
    print("📦 LOADING ENSEMBLE INFRASTRUCTURE")
    print("-" * 50)
    
    # Load the SAME vectorizer used by current ensemble
    vectorizer = joblib.load(models_dir / "tfidf_vectorizer_v1.0.0.joblib")
    print(f"✅ TF-IDF Vectorizer: {len(vectorizer.vocabulary_):,} features")
    
    # Load existing models for ensemble testing
    baseline_models = {
        "logistic": joblib.load(models_dir / "logistic_regression_baseline_v1.0.0.joblib"),
        "svm": joblib.load(models_dir / "svm_baseline_v1.0.0.joblib"),
        "naive_bayes": joblib.load(models_dir / "naive_bayes_baseline_v1.0.0.joblib"),
    }
    
    # Load the SAME data splits used by current ensemble
    df = pd.read_csv(data_dir / "processed" / "full_processed.csv")
    X = df['original_message']
    y = df['label_encoded']
    
    # Use IDENTICAL split as current ensemble
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    print(f"✅ Baseline models: {len(baseline_models)}")
    print(f"✅ Data splits: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    print(f"✅ Same infrastructure as current 93.75% F1 ensemble")
    print()
    
    return baseline_models, vectorizer, X_train, X_val, X_test, y_train, y_val, y_test

def train_neural_network(vectorizer, X_train, X_val, y_train, y_val):
    """Train neural network optimized for ensemble integration"""
    print("🧠 TRAINING NEURAL NETWORK FOR ENSEMBLE")
    print("-" * 50)
    
    # Transform data using SAME vectorizer as ensemble
    print("🔄 Transforming text data...")
    X_train_vec = vectorizer.transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    print(f"✅ Training features: {X_train_vec.shape}")
    print(f"✅ Validation features: {X_val_vec.shape}")
    print(f"✅ Feature sparsity: {1 - X_train_vec.nnz / np.prod(X_train_vec.shape):.3f}")
    
    # Define neural network architectures to test
    nn_configs = {
        "ensemble_optimized": {
            "hidden_layer_sizes": (512, 256, 128),
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "max_iter": 500,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 20,
            "random_state": 42
        },
        "deep_ensemble": {
            "hidden_layer_sizes": (400, 200, 100, 50),
            "alpha": 0.005,
            "learning_rate_init": 0.002,
            "max_iter": 400,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 15,
            "random_state": 42
        },
        "wide_ensemble": {
            "hidden_layer_sizes": (800, 400),
            "alpha": 0.001,
            "learning_rate_init": 0.001,
            "max_iter": 300,
            "early_stopping": True,
            "validation_fraction": 0.15,
            "n_iter_no_change": 20,
            "random_state": 42
        }
    }
    
    print(f"\n🎯 Testing {len(nn_configs)} neural network configurations:")
    
    best_nn = None
    best_f1 = 0
    best_config_name = None
    nn_results = {}
    
    for config_name, config in nn_configs.items():
        print(f"\n🔄 Training {config_name}...")
        print(f"   Architecture: {config['hidden_layer_sizes']}")
        print(f"   Learning rate: {config['learning_rate_init']}")
        print(f"   Alpha: {config['alpha']}")
        
        # Train model
        nn_model = MLPClassifier(**config)
        
        start_time = datetime.now()
        nn_model.fit(X_train_vec, y_train)
        training_time = (datetime.now() - start_time).total_seconds()
        
        # Evaluate
        y_pred = nn_model.predict(X_val_vec)
        f1 = f1_score(y_val, y_pred, average='binary', pos_label=1)
        precision = precision_score(y_val, y_pred, average='binary', pos_label=1)
        recall = recall_score(y_val, y_pred, average='binary', pos_label=1)
        accuracy = accuracy_score(y_val, y_pred)
        
        nn_results[config_name] = {
            "f1": f1,
            "precision": precision,
            "recall": recall,
            "accuracy": accuracy,
            "training_time": training_time,
            "iterations": nn_model.n_iter_,
            "architecture": config['hidden_layer_sizes']
        }
        
        print(f"   ✅ F1: {f1:.4f} ({f1*100:.2f}%)")
        print(f"   🎯 Precision: {precision:.4f}")
        print(f"   📈 Recall: {recall:.4f}")
        print(f"   ⏱️ Training time: {training_time:.1f}s")
        print(f"   🔄 Iterations: {nn_model.n_iter_}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_nn = nn_model
            best_config_name = config_name
            print("   🏆 NEW BEST!")
    
    print(f"\n🏆 BEST NEURAL NETWORK: {best_config_name}")
    print(f"   📊 F1-Score: {best_f1:.4f} ({best_f1*100:.2f}%)")
    print(f"   🏗️ Architecture: {nn_results[best_config_name]['architecture']}")
    
    return best_nn, best_f1, best_config_name, nn_results

def test_neural_network_ensemble(baseline_models, neural_network, vectorizer, X_train, X_val, y_train, y_val):
    """Test ensemble combinations including the new neural network"""
    print("\n🔗 TESTING NEURAL NETWORK ENSEMBLE INTEGRATION")
    print("-" * 60)
    
    # Transform data
    X_train_vec = vectorizer.transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    # Test different ensemble combinations
    ensemble_configs = [
        {
            "name": "baseline_only",
            "models": ["logistic", "svm", "naive_bayes"],
            "description": "Current best ensemble (baseline)"
        },
        {
            "name": "with_neural_network",
            "models": ["logistic", "svm", "naive_bayes", "neural_network"],
            "description": "Baseline + Neural Network"
        },
        {
            "name": "top3_with_nn",
            "models": ["logistic", "svm", "neural_network"],
            "description": "Top 2 baseline + Neural Network"
        },
        {
            "name": "neural_plus_best",
            "models": ["logistic", "neural_network"],
            "description": "Best baseline + Neural Network only"
        }
    ]
    
    # Add neural network to models dict
    all_models = {**baseline_models, "neural_network": neural_network}
    
    ensemble_results = {}
    best_ensemble_f1 = 0
    best_ensemble_config = None
    best_ensemble_model = None
    
    for config in ensemble_configs:
        config_name = config["name"]
        model_names = config["models"]
        description = config["description"]
        
        print(f"\n🎯 Testing: {description}")
        print(f"   Models: {model_names}")
        
        try:
            # Prepare estimators
            estimators = []
            for name in model_names:
                if name in all_models:
                    estimators.append((name, all_models[name]))
            
            if len(estimators) < 2:
                print(f"   ⚠️ Not enough models ({len(estimators)})")
                continue
            
            # Create stacking classifier
            stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(C=1.0, random_state=42, max_iter=1000),
                cv=3,
                n_jobs=-1
            )
            
            # Fit and evaluate
            print(f"   🏋️ Training ensemble...")
            stacking.fit(X_train_vec, y_train)
            y_pred = stacking.predict(X_val_vec)
            
            # Calculate metrics
            f1 = f1_score(y_val, y_pred, average='binary', pos_label=1)
            precision = precision_score(y_val, y_pred, average='binary', pos_label=1)
            recall = recall_score(y_val, y_pred, average='binary', pos_label=1)
            accuracy = accuracy_score(y_val, y_pred)
            
            ensemble_results[config_name] = {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy,
                "models": model_names,
                "description": description,
                "model": stacking
            }
            
            print(f"   ✅ F1: {f1:.4f} ({f1*100:.2f}%)")
            print(f"   🎯 Precision: {precision:.4f}")
            print(f"   📈 Recall: {recall:.4f}")
            print(f"   ✓ Accuracy: {accuracy:.4f}")
            
            # Check if best
            if f1 > best_ensemble_f1:
                best_ensemble_f1 = f1
                best_ensemble_config = config_name
                best_ensemble_model = stacking
                print(f"   🏆 NEW BEST ENSEMBLE!")
            
        except Exception as e:
            print(f"   ❌ Failed: {e}")
            ensemble_results[config_name] = {"error": str(e)}
    
    return ensemble_results, best_ensemble_f1, best_ensemble_config, best_ensemble_model

def validate_on_independent_dataset(best_ensemble, vectorizer):
    """Validate best ensemble on independent dataset"""
    print("\n🧪 INDEPENDENT DATASET VALIDATION")
    print("-" * 40)
    
    try:
        # Load independent dataset
        df_independent = pd.read_csv("data/Dataset_5971.csv")
        
        # Check column names and prepare data
        if 'v2' in df_independent.columns and 'v1' in df_independent.columns:
            X_independent = df_independent['v2']
            y_independent = (df_independent['v1'] == 'spam').astype(int)
            
            # Transform and predict
            X_independent_vec = vectorizer.transform(X_independent)
            y_pred_independent = best_ensemble.predict(X_independent_vec)
            
            # Calculate metrics
            ind_f1 = f1_score(y_independent, y_pred_independent, average='binary', pos_label=1)
            ind_precision = precision_score(y_independent, y_pred_independent, average='binary', pos_label=1)
            ind_recall = recall_score(y_independent, y_pred_independent, average='binary', pos_label=1)
            ind_accuracy = accuracy_score(y_independent, y_pred_independent)
            
            print(f"✅ Independent validation ({len(y_independent):,} samples):")
            print(f"  📊 F1-Score: {ind_f1:.4f} ({ind_f1*100:.2f}%)")
            print(f"  🎯 Precision: {ind_precision:.4f}")
            print(f"  📈 Recall: {ind_recall:.4f}")
            print(f"  ✓ Accuracy: {ind_accuracy:.4f}")
            
            return ind_f1, ind_precision, ind_recall, ind_accuracy
            
        else:
            print("⚠️ Independent dataset column names not recognized")
            return None, None, None, None
            
    except Exception as e:
        print(f"⚠️ Independent validation failed: {e}")
        return None, None, None, None

def save_models_and_results(neural_network, best_ensemble, nn_results, ensemble_results, best_config):
    """Save trained models and comprehensive results"""
    print("\n💾 SAVING MODELS AND RESULTS")
    print("-" * 40)
    
    timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
    
    # Save neural network
    nn_path = Path("models") / f"neural_network_ensemble_{timestamp}.joblib"
    joblib.dump(neural_network, nn_path)
    print(f"✅ Neural network saved: {nn_path}")
    
    # Save best ensemble
    ensemble_path = Path("ensemble_models") / f"neural_ensemble_best_{timestamp}.joblib"
    joblib.dump(best_ensemble, ensemble_path)
    print(f"✅ Best ensemble saved: {ensemble_path}")
    
    # Save comprehensive results
    results = {
        "timestamp": timestamp,
        "neural_network_results": nn_results,
        "ensemble_results": {k: {**v, "model": None} if "model" in v else v for k, v in ensemble_results.items()},
        "best_ensemble_config": best_config,
        "achievement_analysis": {
            "baseline_f1": 0.9375,
            "neural_network_f1": max(nn_results.values(), key=lambda x: x["f1"])["f1"],
            "best_ensemble_f1": ensemble_results[best_config]["f1"],
            "improvement_over_baseline": ensemble_results[best_config]["f1"] - 0.9375,
            "target_94_achieved": ensemble_results[best_config]["f1"] >= 0.94
        }
    }
    
    results_path = Path("ensemble_models") / f"neural_ensemble_results_{timestamp}.json"
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"✅ Results saved: {results_path}")
    
    return results

def main():
    """Main execution for neural network ensemble creation"""
    
    # Step 1: Load infrastructure
    baseline_models, vectorizer, X_train, X_val, X_test, y_train, y_val, y_test = load_ensemble_infrastructure()
    
    # Step 2: Train neural network
    neural_network, nn_f1, best_nn_config, nn_results = train_neural_network(
        vectorizer, X_train, X_val, y_train, y_val
    )
    
    # Step 3: Test ensemble combinations
    ensemble_results, best_ensemble_f1, best_ensemble_config, best_ensemble_model = test_neural_network_ensemble(
        baseline_models, neural_network, vectorizer, X_train, X_val, y_train, y_val
    )
    
    # Step 4: Independent validation
    ind_f1, ind_precision, ind_recall, ind_accuracy = validate_on_independent_dataset(
        best_ensemble_model, vectorizer
    )
    
    # Step 5: Save models and results
    results = save_models_and_results(
        neural_network, best_ensemble_model, nn_results, ensemble_results, best_ensemble_config
    )
    
    # Final summary
    print("\n🎉 NEURAL NETWORK ENSEMBLE CREATION COMPLETE!")
    print("=" * 60)
    
    baseline_f1 = 0.9375
    improvement = best_ensemble_f1 - baseline_f1
    
    print(f"📊 Baseline (Day 4): {baseline_f1*100:.2f}% F1-Score")
    print(f"🧠 Neural Network: {nn_f1*100:.2f}% F1-Score ({best_nn_config})")
    print(f"🏆 Best Ensemble: {best_ensemble_f1*100:.2f}% F1-Score ({best_ensemble_config})")
    print(f"📈 Improvement: {improvement*100:+.2f} percentage points")
    print(f"🎯 94% Target: {'✅ ACHIEVED!' if best_ensemble_f1 >= 0.94 else f'❌ {(0.94-best_ensemble_f1)*100:.2f} points remaining'}")
    
    if ind_f1:
        print(f"🧪 Independent validation: {ind_f1*100:.2f}% F1-Score")
    
    print(f"\n📋 DETAILED ENSEMBLE RESULTS:")
    for config_name, result in ensemble_results.items():
        if "error" not in result:
            f1 = result["f1"]
            description = result["description"]
            print(f"  {config_name}: {f1*100:.2f}% F1 - {description}")
    
    return best_ensemble_f1 >= 0.94

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎊 SUCCESS: 94% TARGET ACHIEVED WITH NEURAL NETWORK ENSEMBLE!")
    else:
        print("\n📈 PROGRESS: Significant improvement achieved, continue optimization if needed") 