#!/usr/bin/env python3
"""
Day 4 Optimization Sprint - FIXED VERSION - 94% Target Achievement
Current: 93.75% F1 → Target: 94%+ F1 (0.25+ percentage point improvement)
"""

import warnings
import pandas as pd
import numpy as np
import joblib
from pathlib import Path
from datetime import datetime
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC  # Use SVC instead of LinearSVC for predict_proba
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.calibration import CalibratedClassifierCV

warnings.filterwarnings('ignore')

print("🎯 DAY 4 OPTIMIZATION SPRINT - FIXED VERSION - 94% TARGET")
print("=" * 70)
print(f"📅 Started: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("🚀 Goal: 93.75% → 94%+ F1-Score (0.25+ points)")
print("🔧 Fixed: SVM predict_proba + Enhanced optimization")
print("=" * 70)
print()

def load_models_and_data():
    """Load baseline models and data with SVM fix"""
    models_dir = Path("models")
    data_dir = Path("data")
    
    # Load vectorizer
    vectorizer = joblib.load(models_dir / "tfidf_vectorizer_v1.0.0.joblib")
    
    # Load models
    models = {
        "logistic": joblib.load(models_dir / "logistic_regression_baseline_v1.0.0.joblib"),
        "svm": joblib.load(models_dir / "svm_baseline_v1.0.0.joblib"),
        "naive_bayes": joblib.load(models_dir / "naive_bayes_baseline_v1.0.0.joblib"),
        "random_forest": joblib.load(models_dir / "random_forest_baseline_v1.0.0.joblib")
    }
    
    # Check if SVM supports predict_proba, if not, calibrate it
    if not hasattr(models["svm"], 'predict_proba'):
        print("🔧 SVM doesn't support predict_proba, creating calibrated version...")
        models["svm_calibrated"] = CalibratedClassifierCV(models["svm"], cv=3)
        models["svm_for_voting"] = models["svm_calibrated"]
    else:
        models["svm_for_voting"] = models["svm"]
    
    # Load data
    df = pd.read_csv(data_dir / "processed" / "full_processed.csv")
    X = df['original_message']
    y = df['label_encoded']
    
    # Split data (same as Day 4)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    print(f"✅ Loaded: {len(models)} models, {len(X_train)} train, {len(X_val)} val samples")
    return models, vectorizer, X_train, X_val, y_train, y_val

def optimize_advanced_stacking(models, vectorizer, X_train, X_val, y_train, y_val):
    """Advanced stacking optimization with multiple meta-learners"""
    print("\n🔧 ADVANCED STACKING OPTIMIZATION")
    print("-" * 50)
    
    # Transform data
    X_train_vec = vectorizer.transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    best_f1 = 0
    best_config = None
    best_model = None
    
    # Test different model combinations
    model_combos = [
        ("top2", ["logistic", "svm"]),
        ("top3", ["logistic", "svm", "naive_bayes"]),
        ("balanced3", ["logistic", "naive_bayes", "random_forest"]),
    ]
    
    # Multiple meta-learner configurations
    meta_learners = {
        "logistic_optimized": LogisticRegression(C=10.0, random_state=42, max_iter=1000),
        "logistic_regularized": LogisticRegression(C=0.1, random_state=42, max_iter=1000),
        "random_forest_tuned": RandomForestClassifier(n_estimators=100, max_depth=5, random_state=42)
    }
    
    for combo_name, model_names in model_combos:
        print(f"\n🎯 {combo_name.upper()}: {model_names}")
        
        # Prepare estimators
        estimators = [(name, models[name]) for name in model_names]
        
        for meta_name, meta_learner in meta_learners.items():
            print(f"   🧠 Meta-learner: {meta_name}")
            
            # Create stacking classifier
            stacking = StackingClassifier(
                estimators=estimators,
                final_estimator=meta_learner,
                cv=5, n_jobs=-1
            )
            
            # Fit and evaluate
            stacking.fit(X_train_vec, y_train)
            y_pred = stacking.predict(X_val_vec)
            
            f1 = f1_score(y_val, y_pred, average='binary', pos_label=1)
            precision = precision_score(y_val, y_pred, average='binary', pos_label=1)
            recall = recall_score(y_val, y_pred, average='binary', pos_label=1)
            
            print(f"     ✅ F1: {f1:.4f} ({f1*100:.2f}%)")
            print(f"     🎯 Precision: {precision:.4f}")
            print(f"     📈 Recall: {recall:.4f}")
            
            # Check if best
            if f1 > best_f1:
                best_f1 = f1
                best_config = f"{combo_name}_{meta_name}"
                best_model = stacking
                print(f"     🏆 NEW BEST!")
    
    return best_f1, best_config, best_model

def optimize_calibrated_voting(models, vectorizer, X_train, X_val, y_train, y_val):
    """Optimize voting with calibrated SVM"""
    print("\n⚖️ CALIBRATED VOTING OPTIMIZATION")
    print("-" * 40)
    
    # Transform data
    X_train_vec = vectorizer.transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    # First, fit calibrated SVM if needed
    if "svm_calibrated" in models:
        print("🔧 Fitting calibrated SVM...")
        models["svm_calibrated"].fit(X_train_vec, y_train)
    
    # Get individual performances for weighting (use predict_proba compatible models)
    voting_models = {
        "logistic": models["logistic"],
        "svm": models["svm_for_voting"] if "svm_for_voting" in models else models["svm"],
        "naive_bayes": models["naive_bayes"],
        "random_forest": models["random_forest"]
    }
    
    performances = {}
    for name, model in voting_models.items():
        # Fit if not already fitted (for calibrated models)
        if name != "svm" or not hasattr(model, 'predict_proba'):
            if hasattr(model, 'fit'):
                model.fit(X_train_vec, y_train)
        
        y_pred = model.predict(X_val_vec)
        f1 = f1_score(y_val, y_pred, average='binary', pos_label=1)
        performances[name] = f1
        print(f"  📊 {name}: {f1*100:.2f}% F1")
    
    # Performance-based weights
    total_perf = sum(performances.values())
    weights = {name: perf/total_perf for name, perf in performances.items()}
    
    print(f"\n⚖️ Weights: {[(name, f'{w:.3f}') for name, w in weights.items()]}")
    
    best_f1 = 0
    best_config = None
    best_model = None
    
    # Test multiple combinations
    combos = [
        ("top2", ["logistic", "svm"]),
        ("top3", ["logistic", "svm", "naive_bayes"]),
        ("balanced3", ["logistic", "naive_bayes", "random_forest"]),
        ("all4", list(voting_models.keys()))
    ]
    
    for combo_name, model_names in combos:
        print(f"\n🎯 {combo_name.upper()}: {model_names}")
        
        try:
            estimators = [(name, voting_models[name]) for name in model_names]
            combo_weights = [weights[name] for name in model_names]
            
            # Weighted voting classifier
            voting = VotingClassifier(
                estimators=estimators,
                voting='soft',
                weights=combo_weights
            )
            
            voting.fit(X_train_vec, y_train)
            y_pred = voting.predict(X_val_vec)
            
            f1 = f1_score(y_val, y_pred, average='binary', pos_label=1)
            precision = precision_score(y_val, y_pred, average='binary', pos_label=1)
            recall = recall_score(y_val, y_pred, average='binary', pos_label=1)
            
            print(f"  ✅ F1: {f1:.4f} ({f1*100:.2f}%)")
            print(f"  🎯 Precision: {precision:.4f}")
            print(f"  📈 Recall: {recall:.4f}")
            
            if f1 > best_f1:
                best_f1 = f1
                best_config = combo_name
                best_model = voting
                print(f"  🏆 NEW BEST!")
                
        except Exception as e:
            print(f"  ❌ {combo_name} failed: {e}")
    
    return best_f1, best_config, best_model

def optimize_threshold_tuning(best_model, vectorizer, X_train, X_val, y_train, y_val):
    """Optimize decision threshold for best model"""
    print("\n🎚️ DECISION THRESHOLD OPTIMIZATION")
    print("-" * 40)
    
    # Transform data
    X_train_vec = vectorizer.transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    # Get prediction probabilities
    if hasattr(best_model, 'predict_proba'):
        y_proba = best_model.predict_proba(X_val_vec)[:, 1]
    elif hasattr(best_model, 'decision_function'):
        y_proba = best_model.decision_function(X_val_vec)
    else:
        print("⚠️ Model doesn't support probability/decision function")
        return None, None
    
    # Test different thresholds
    thresholds = np.arange(0.3, 0.8, 0.05)
    best_threshold = 0.5
    best_f1 = 0
    
    print("🔍 Testing thresholds:")
    for threshold in thresholds:
        y_pred_thresh = (y_proba >= threshold).astype(int)
        f1 = f1_score(y_val, y_pred_thresh, average='binary', pos_label=1)
        precision = precision_score(y_val, y_pred_thresh, average='binary', pos_label=1)
        recall = recall_score(y_val, y_pred_thresh, average='binary', pos_label=1)
        
        print(f"  Threshold {threshold:.2f}: F1={f1:.4f} ({f1*100:.2f}%), P={precision:.3f}, R={recall:.3f}")
        
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = threshold
    
    print(f"\n🏆 Best threshold: {best_threshold:.2f} → F1: {best_f1:.4f} ({best_f1*100:.2f}%)")
    return best_threshold, best_f1

def main():
    """Main optimization execution"""
    
    # Load models and data
    models, vectorizer, X_train, X_val, y_train, y_val = load_models_and_data()
    
    # Optimization 1: Advanced stacking
    stack_f1, stack_config, stack_model = optimize_advanced_stacking(
        models, vectorizer, X_train, X_val, y_train, y_val
    )
    
    # Optimization 2: Calibrated voting
    vote_f1, vote_config, vote_model = optimize_calibrated_voting(
        models, vectorizer, X_train, X_val, y_train, y_val
    )
    
    # Find best overall model
    if stack_f1 > vote_f1:
        best_f1, best_type, best_model = stack_f1, "stacking", stack_model
        best_config = stack_config
    else:
        best_f1, best_type, best_model = vote_f1, "voting", vote_model
        best_config = vote_config
    
    # Optimization 3: Threshold tuning on best model
    threshold, threshold_f1 = optimize_threshold_tuning(
        best_model, vectorizer, X_train, X_val, y_train, y_val
    )
    
    if threshold_f1 and threshold_f1 > best_f1:
        print(f"🎚️ Threshold tuning improved F1: {best_f1:.4f} → {threshold_f1:.4f}")
        best_f1 = threshold_f1
        best_type = f"{best_type}_threshold_tuned"
    
    # Final results
    print("\n🎉 OPTIMIZATION RESULTS SUMMARY")
    print("=" * 50)
    
    baseline_f1 = 0.9375  # Day 4 baseline
    improvement = (best_f1 - baseline_f1) * 100
    
    print(f"📊 Baseline (Day 4): {baseline_f1*100:.2f}% F1-Score")
    print(f"🏆 Best Stacking: {stack_f1*100:.2f}% F1 ({stack_config})")
    print(f"🗳️ Best Voting: {vote_f1*100:.2f}% F1 ({vote_config})")
    print(f"🎚️ Best with Threshold: {threshold_f1*100:.2f}% F1" if threshold_f1 else "🎚️ Threshold tuning: N/A")
    
    print(f"\n🏆 OVERALL BEST: {best_f1*100:.2f}% F1-Score ({best_type})")
    print(f"📈 Improvement: {improvement:+.2f} percentage points")
    print(f"🎯 94% Target: {'✅ ACHIEVED!' if best_f1 >= 0.94 else f'❌ {(0.94-best_f1)*100:.2f} points remaining'}")
    
    # Save best model
    if best_model:
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        model_path = Path("ensemble_models") / f"optimized_best_{timestamp}.joblib"
        joblib.dump(best_model, model_path)
        print(f"💾 Best model saved: {model_path}")
        
        # Save threshold if applicable
        if threshold:
            threshold_path = Path("ensemble_models") / f"best_threshold_{timestamp}.json"
            with open(threshold_path, 'w') as f:
                import json
                json.dump({"threshold": threshold, "f1_score": threshold_f1}, f)
            print(f"💾 Best threshold saved: {threshold_path}")
    
    return best_f1 >= 0.94

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ 94% TARGET ACHIEVED!' if success else '⚠️ Continue optimization needed'}") 