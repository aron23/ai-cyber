#!/usr/bin/env python3
"""
Neural Network Ensemble Integration - 94% Target Achievement
Integrating Day 3 Wide Network (91.34% F1) with baseline models
Current best: 93.75% F1 → Target: 94%+ F1
"""

import warnings
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score
from sklearn.base import BaseEstimator, ClassifierMixin

warnings.filterwarnings('ignore')

print("🧠 NEURAL NETWORK ENSEMBLE INTEGRATION - 94% TARGET")
print("=" * 70)
print(f"📅 Started: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("🚀 Goal: Integrate NN (91.34% F1) + Baseline models → 94%+ F1")
print("🎯 Strategy: Multi-model stacking with neural network")
print("=" * 70)
print()

class NeuralNetworkWrapper(BaseEstimator, ClassifierMixin):
    """Wrapper for neural network model to work with sklearn ensemble"""
    
    def __init__(self, model_path, vectorizer):
        self.model_path = model_path
        self.vectorizer = vectorizer
        self.model = None
        self.is_fitted = False
    
    def fit(self, X, y):
        """Load and prepare neural network model"""
        if not self.is_fitted:
            self.model = joblib.load(self.model_path)
            self.is_fitted = True
        return self
    
    def predict(self, X):
        """Make predictions using neural network"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Transform text to TF-IDF features
        X_transformed = self.vectorizer.transform(X)
        
        # Neural network prediction
        predictions = self.model.predict(X_transformed)
        return predictions
    
    def predict_proba(self, X):
        """Get prediction probabilities - neural networks typically support this"""
        if not self.is_fitted:
            raise ValueError("Model must be fitted before prediction")
        
        # Transform text to TF-IDF features
        X_transformed = self.vectorizer.transform(X)
        
        # Get probabilities if available
        if hasattr(self.model, 'predict_proba'):
            return self.model.predict_proba(X_transformed)
        else:
            # Fallback: convert predictions to probabilities
            predictions = self.model.predict(X_transformed)
            proba = np.zeros((len(predictions), 2))
            proba[predictions == 0, 0] = 0.8  # 80% confidence for class 0
            proba[predictions == 0, 1] = 0.2
            proba[predictions == 1, 0] = 0.2
            proba[predictions == 1, 1] = 0.8  # 80% confidence for class 1
            return proba

def load_models_and_data():
    """Load all models including neural network"""
    models_dir = Path("models")
    data_dir = Path("data")
    
    print("📦 LOADING MODELS AND DATA")
    print("-" * 40)
    
    # Load vectorizer
    vectorizer = joblib.load(models_dir / "tfidf_vectorizer_v1.0.0.joblib")
    print(f"✅ TF-IDF Vectorizer: {len(vectorizer.vocabulary_):,} features")
    
    # Load baseline models
    baseline_models = {
        "logistic": joblib.load(models_dir / "logistic_regression_baseline_v1.0.0.joblib"),
        "svm": joblib.load(models_dir / "svm_baseline_v1.0.0.joblib"),
        "naive_bayes": joblib.load(models_dir / "naive_bayes_baseline_v1.0.0.joblib"),
        "random_forest": joblib.load(models_dir / "random_forest_baseline_v1.0.0.joblib")
    }
    
    # Load neural network (wide_network - best performer at 91.34% F1)
    nn_path = models_dir / "day3_neural_networks_cpu" / "wide_network_model.joblib"
    neural_network = NeuralNetworkWrapper(nn_path, vectorizer)
    
    print(f"✅ Baseline models: {len(baseline_models)}")
    print("✅ Neural Network: wide_network (91.34% F1)")
    
    # Combine all models
    all_models = {**baseline_models, "neural_network": neural_network}
    
    # Load data
    df = pd.read_csv(data_dir / "processed" / "full_processed.csv")
    X = df['original_message']
    y = df['label_encoded']
    
    # Split data (same as Day 4)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    print(f"✅ Data: {len(X_train)} train, {len(X_val)} val, {len(X_test)} test")
    print()
    
    return all_models, vectorizer, X_train, X_val, y_train, y_val

def validate_individual_models(models, vectorizer, X_train, X_val, y_train, y_val):
    """Validate individual model performance including neural network"""
    print("🧪 VALIDATING INDIVIDUAL MODEL PERFORMANCE")
    print("-" * 50)
    
    # Transform data
    X_train_vec = vectorizer.transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    performances = {}
    
    for name, model in models.items():
        print(f"\n🎯 {name.upper()}")
        
        try:
            # Fit model
            if name == "neural_network":
                # Neural network uses raw text, fit with X_train
                model.fit(X_train, y_train)
                y_pred = model.predict(X_val)
            else:
                # Baseline models use vectorized features
                model.fit(X_train_vec, y_train)
                y_pred = model.predict(X_val_vec)
            
            # Calculate metrics
            f1 = f1_score(y_val, y_pred, average='binary', pos_label=1)
            precision = precision_score(y_val, y_pred, average='binary', pos_label=1)
            recall = recall_score(y_val, y_pred, average='binary', pos_label=1)
            accuracy = accuracy_score(y_val, y_pred)
            
            performances[name] = {
                "f1": f1,
                "precision": precision,
                "recall": recall,
                "accuracy": accuracy
            }
            
            print(f"  ✅ F1: {f1:.4f} ({f1*100:.2f}%)")
            print(f"  🎯 Precision: {precision:.4f}")
            print(f"  📈 Recall: {recall:.4f}")
            print(f"  ✓ Accuracy: {accuracy:.4f}")
            
        except Exception as e:
            print(f"  ❌ Failed: {e}")
            performances[name] = {"f1": 0, "error": str(e)}
    
    # Summary
    valid_models = {name: perf for name, perf in performances.items() if "error" not in perf}
    best_model = max(valid_models.keys(), key=lambda x: valid_models[x]["f1"])
    
    print(f"\n📊 INDIVIDUAL MODEL SUMMARY")
    print("-" * 30)
    for name, perf in sorted(valid_models.items(), key=lambda x: x[1]["f1"], reverse=True):
        print(f"  {name}: {perf['f1']*100:.2f}% F1")
    
    print(f"\n🏆 Best individual: {best_model} ({valid_models[best_model]['f1']*100:.2f}% F1)")
    print()
    
    return performances, valid_models

def test_neural_network_ensembles(models, valid_models, vectorizer, X_train, X_val, y_train, y_val):
    """Test ensemble combinations including neural network"""
    print("🧠 TESTING NEURAL NETWORK ENSEMBLES")
    print("-" * 50)
    
    # Transform data for baseline models
    X_train_vec = vectorizer.transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    best_f1 = 0
    best_config = None
    best_model = None
    
    # Test combinations with neural network
    test_combinations = [
        ("top2_nn", ["logistic", "neural_network"]),
        ("top3_nn", ["logistic", "svm", "neural_network"]),
        ("top4_nn", ["logistic", "svm", "naive_bayes", "neural_network"]),
        ("all5_nn", list(valid_models.keys()))
    ]
    
    for combo_name, model_names in test_combinations:
        print(f"\n🎯 {combo_name.upper()}: {model_names}")
        
        try:
            # Prepare estimators
            estimators = []
            for name in model_names:
                if name in models and name in valid_models:
                    estimators.append((name, models[name]))
            
            if len(estimators) < 2:
                print(f"  ⚠️ Not enough valid models ({len(estimators)})")
                continue
            
            # Create stacking classifier
            stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=LogisticRegression(C=10.0, random_state=42, max_iter=1000),
                cv=3,
                n_jobs=-1
            )
            
            # Fit and predict
            print(f"  🏋️ Training ensemble with {len(estimators)} models...")
            stacking.fit(X_train, y_train)  # Use raw text for compatibility with neural network
            y_pred = stacking.predict(X_val)
            
            # Calculate metrics
            f1 = f1_score(y_val, y_pred, average='binary', pos_label=1)
            precision = precision_score(y_val, y_pred, average='binary', pos_label=1)
            recall = recall_score(y_val, y_pred, average='binary', pos_label=1)
            accuracy = accuracy_score(y_val, y_pred)
            
            print(f"  ✅ F1: {f1:.4f} ({f1*100:.2f}%)")
            print(f"  🎯 Precision: {precision:.4f}")
            print(f"  📈 Recall: {recall:.4f}")
            print(f"  ✓ Accuracy: {accuracy:.4f}")
            
            # Check if best
            if f1 > best_f1:
                best_f1 = f1
                best_config = combo_name
                best_model = stacking
                print(f"  🏆 NEW BEST!")
            
        except Exception as e:
            print(f"  ❌ {combo_name} failed: {e}")
    
    return best_f1, best_config, best_model

def main():
    """Main neural network integration execution"""
    
    # Load models and data
    models, vectorizer, X_train, X_val, y_train, y_val = load_models_and_data()
    
    # Validate individual models
    performances, valid_models = validate_individual_models(
        models, vectorizer, X_train, X_val, y_train, y_val
    )
    
    # Test neural network ensembles
    best_f1, best_config, best_model = test_neural_network_ensembles(
        models, valid_models, vectorizer, X_train, X_val, y_train, y_val
    )
    
    # Results summary
    print("\n🎉 NEURAL NETWORK INTEGRATION RESULTS")
    print("=" * 50)
    
    baseline_f1 = 0.9375  # Day 4 baseline
    improvement = (best_f1 - baseline_f1) * 100
    
    print(f"📊 Baseline (Day 4): {baseline_f1*100:.2f}% F1-Score")
    print(f"🧠 Neural Network: {performances.get('neural_network', {}).get('f1', 0)*100:.2f}% F1")
    print(f"🏆 Best Ensemble: {best_f1*100:.2f}% F1 ({best_config})")
    print(f"📈 Improvement: {improvement:+.2f} percentage points")
    print(f"🎯 94% Target: {'✅ ACHIEVED!' if best_f1 >= 0.94 else f'❌ {(0.94-best_f1)*100:.2f} points remaining'}")
    
    # Save best model
    if best_model and best_f1 > baseline_f1:
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        model_path = Path("ensemble_models") / f"neural_ensemble_best_{timestamp}.joblib"
        joblib.dump(best_model, model_path)
        print(f"💾 Best neural ensemble saved: {model_path}")
    
    return best_f1 >= 0.94

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ 94% TARGET ACHIEVED WITH NEURAL NETWORKS!' if success else '⚠️ Continue with other optimizations'}") 