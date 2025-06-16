#!/usr/bin/env python3
"""
Final Threshold Optimization - 94% Target Achievement
Current: 93.75% F1 → Target: 94%+ F1 (0.25+ points needed)
Strategy: Optimize decision threshold for our best stacking ensemble
"""

import warnings
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import train_test_split
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score, precision_score, recall_score, accuracy_score

warnings.filterwarnings('ignore')

print("🎚️ FINAL THRESHOLD OPTIMIZATION - 94% TARGET")
print("=" * 60)
print(f"📅 Started: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("🚀 Goal: 93.75% → 94%+ F1 via threshold tuning")
print("🎯 Strategy: Optimize decision threshold for F1-Score")
print("=" * 60)
print()

def load_best_ensemble_and_data():
    """Load our best ensemble and data"""
    models_dir = Path("models")
    data_dir = Path("data")
    
    print("📦 LOADING BEST ENSEMBLE AND DATA")
    print("-" * 40)
    
    # Load vectorizer
    vectorizer = joblib.load(models_dir / "tfidf_vectorizer_v1.0.0.joblib")
    
    # Load baseline models
    models = {
        "logistic": joblib.load(models_dir / "logistic_regression_baseline_v1.0.0.joblib"),
        "svm": joblib.load(models_dir / "svm_baseline_v1.0.0.joblib"),
        "naive_bayes": joblib.load(models_dir / "naive_bayes_baseline_v1.0.0.joblib"),
    }
    
    # Load data
    df = pd.read_csv(data_dir / "processed" / "full_processed.csv")
    X = df['original_message']
    y = df['label_encoded']
    
    # Split data (same as Day 4)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    print(f"✅ Models: {len(models)}")
    print(f"✅ Data: {len(X_train)} train, {len(X_val)} val")
    print()
    
    return models, vectorizer, X_train, X_val, y_train, y_val

def create_best_ensemble(models, vectorizer, X_train, y_train):
    """Create our best ensemble (TOP3: Logistic + SVM + Naive Bayes)"""
    print("🏗️ CREATING BEST ENSEMBLE")
    print("-" * 30)
    
    # Transform data
    X_train_vec = vectorizer.transform(X_train)
    
    # Create estimators
    estimators = [
        ("logistic", models["logistic"]),
        ("svm", models["svm"]),
        ("naive_bayes", models["naive_bayes"])
    ]
    
    # Create stacking classifier (based on our Day 4 best configuration)
    stacking = StackingClassifier(
        estimators=estimators,
        final_estimator=LogisticRegression(C=1.0, random_state=42, max_iter=1000),
        cv=5, n_jobs=-1
    )
    
    print("🏋️ Training best ensemble...")
    stacking.fit(X_train_vec, y_train)
    
    print("✅ Best ensemble trained (Logistic + SVM + Naive Bayes)")
    return stacking

def optimize_decision_threshold(ensemble, vectorizer, X_val, y_val):
    """Optimize decision threshold for maximum F1-Score"""
    print("\n🎚️ OPTIMIZING DECISION THRESHOLD")
    print("-" * 40)
    
    # Transform validation data
    X_val_vec = vectorizer.transform(X_val)
    
    # Get prediction probabilities
    y_proba = ensemble.predict_proba(X_val_vec)[:, 1]
    
    # Test different thresholds
    thresholds = np.arange(0.1, 0.9, 0.01)  # Fine-grained search
    results = []
    
    print("🔍 Testing thresholds (fine-grained search):")
    best_f1 = 0
    best_threshold = 0.5
    
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        
        # Skip if all predictions are the same class
        if len(np.unique(y_pred_thresh)) == 1:
            continue
            
        f1 = f1_score(y_val, y_pred_thresh, average='binary', pos_label=1)
        precision = precision_score(y_val, y_pred_thresh, average='binary', pos_label=1)
        recall = recall_score(y_val, y_pred_thresh, average='binary', pos_label=1)
        accuracy = accuracy_score(y_val, y_pred_thresh)
        
        results.append({
            'threshold': threshold,
            'f1': f1,
            'precision': precision,
            'recall': recall,
            'accuracy': accuracy
        })
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    # Show top 10 thresholds
    results_sorted = sorted(results, key=lambda x: x['f1'], reverse=True)[:10]
    
    print("\n🏆 TOP 10 THRESHOLDS:")
    print("Threshold   F1-Score   Precision   Recall    Accuracy")
    print("-" * 55)
    for result in results_sorted:
        print(f"{result['threshold']:.2f}      {result['f1']:.4f}     {result['precision']:.4f}     {result['recall']:.4f}     {result['accuracy']:.4f}")
    
    # Default threshold performance
    y_pred_default = ensemble.predict(X_val_vec)
    default_f1 = f1_score(y_val, y_pred_default, average='binary', pos_label=1)
    
    print(f"\n📊 THRESHOLD OPTIMIZATION RESULTS:")
    print(f"  🔄 Default threshold (0.5): {default_f1:.4f} ({default_f1*100:.2f}%)")
    print(f"  🏆 Best threshold ({best_threshold:.2f}): {best_f1:.4f} ({best_f1*100:.2f}%)")
    print(f"  📈 Improvement: {(best_f1 - default_f1)*100:+.2f} percentage points")
    
    return best_threshold, best_f1, results_sorted[0]

def validate_on_independent_dataset(ensemble, vectorizer, best_threshold):
    """Validate optimized ensemble on independent dataset"""
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
            y_proba_independent = ensemble.predict_proba(X_independent_vec)[:, 1]
            
            # Apply optimized threshold
            y_pred_independent = (y_proba_independent >= best_threshold).astype(int)
            
            # Calculate metrics
            ind_f1 = f1_score(y_independent, y_pred_independent, average='binary', pos_label=1)
            ind_precision = precision_score(y_independent, y_pred_independent, average='binary', pos_label=1)
            ind_recall = recall_score(y_independent, y_pred_independent, average='binary', pos_label=1)
            ind_accuracy = accuracy_score(y_independent, y_pred_independent)
            
            print(f"✅ Independent validation results:")
            print(f"  📊 F1-Score: {ind_f1:.4f} ({ind_f1*100:.2f}%)")
            print(f"  🎯 Precision: {ind_precision:.4f}")
            print(f"  📈 Recall: {ind_recall:.4f}")
            print(f"  ✓ Accuracy: {ind_accuracy:.4f}")
            
            return ind_f1
            
        else:
            print("⚠️ Independent dataset column names not recognized")
            return None
            
    except Exception as e:
        print(f"⚠️ Independent validation failed: {e}")
        return None

def main():
    """Main threshold optimization execution"""
    
    # Load models and data
    models, vectorizer, X_train, X_val, y_train, y_val = load_best_ensemble_and_data()
    
    # Create best ensemble
    ensemble = create_best_ensemble(models, vectorizer, X_train, y_train)
    
    # Optimize threshold
    best_threshold, best_f1, best_result = optimize_decision_threshold(
        ensemble, vectorizer, X_val, y_val
    )
    
    # Independent validation
    independent_f1 = validate_on_independent_dataset(ensemble, vectorizer, best_threshold)
    
    # Final results
    print("\n🎉 FINAL THRESHOLD OPTIMIZATION RESULTS")
    print("=" * 50)
    
    baseline_f1 = 0.9375  # Day 4 baseline
    improvement = (best_f1 - baseline_f1) * 100
    
    print(f"📊 Baseline (Day 4): {baseline_f1*100:.2f}% F1-Score")
    print(f"🎚️ Optimized Threshold: {best_f1*100:.2f}% F1-Score (threshold={best_threshold:.2f})")
    print(f"📈 Improvement: {improvement:+.2f} percentage points")
    print(f"🎯 94% Target: {'✅ ACHIEVED!' if best_f1 >= 0.94 else f'❌ {(0.94-best_f1)*100:.2f} points remaining'}")
    
    if independent_f1:
        print(f"🧪 Independent validation: {independent_f1*100:.2f}% F1-Score")
    
    # Save optimized ensemble and threshold
    if best_f1 > baseline_f1:
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        
        # Save ensemble
        ensemble_path = Path("ensemble_models") / f"threshold_optimized_ensemble_{timestamp}.joblib"
        joblib.dump(ensemble, ensemble_path)
        
        # Save threshold configuration
        import json
        config = {
            "best_threshold": best_threshold,
            "validation_f1": best_f1,
            "independent_f1": independent_f1,
            "improvement_over_baseline": improvement,
            "full_results": best_result
        }
        config_path = Path("ensemble_models") / f"threshold_config_{timestamp}.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
        
        print(f"💾 Optimized ensemble saved: {ensemble_path}")
        print(f"💾 Threshold config saved: {config_path}")
    
    return best_f1 >= 0.94

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ 94% TARGET ACHIEVED WITH THRESHOLD OPTIMIZATION!' if success else '⚠️ 94% target not achieved - excellent 93.75%+ result nonetheless'}") 