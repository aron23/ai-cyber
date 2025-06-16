#!/usr/bin/env python3
"""
Day 4 Optimization Sprint - 94% Target Achievement
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
from sklearn.metrics import f1_score, precision_score, recall_score

warnings.filterwarnings('ignore')

print("🎯 DAY 4 OPTIMIZATION SPRINT - 94% TARGET")
print("=" * 60)
print(f"📅 Started: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("🚀 Goal: 93.75% → 94%+ F1-Score (0.25+ points)")
print("=" * 60)
print()

def load_models_and_data():
    """Load baseline models and data"""
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
    
    # Load data
    df = pd.read_csv(data_dir / "processed" / "full_processed.csv")
    X = df['original_message']
    y = df['label_encoded']
    
    # Split data (same as Day 4)
    X_temp, X_test, y_temp, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.25, random_state=42, stratify=y_temp)
    
    print(f"✅ Loaded: {len(models)} models, {len(X_train)} train, {len(X_val)} val samples")
    return models, vectorizer, X_train, X_val, y_train, y_val

def optimize_stacking_hyperparameters(models, vectorizer, X_train, X_val, y_train, y_val):
    """Optimize stacking ensemble hyperparameters"""
    print("\n🔧 STACKING HYPERPARAMETER OPTIMIZATION")
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
        ("all4", list(models.keys()))
    ]
    
    # Hyperparameter grids
    lr_params = {
        'final_estimator__C': [0.1, 1.0, 10.0],
        'final_estimator__penalty': ['l2'],
        'final_estimator__solver': ['lbfgs'],
        'final_estimator__max_iter': [1000]
    }
    
    for combo_name, model_names in model_combos:
        print(f"\n🎯 {combo_name.upper()}: {model_names}")
        
        # Prepare estimators
        estimators = [(name, models[name]) for name in model_names]
        
        # Create stacking classifier
        stacking = StackingClassifier(
            estimators=estimators,
            final_estimator=LogisticRegression(random_state=42),
            cv=3, n_jobs=-1
        )
        
        # Grid search
        grid = GridSearchCV(stacking, lr_params, cv=3, scoring='f1', n_jobs=-1)
        grid.fit(X_train_vec, y_train)
        
        # Evaluate on validation set
        y_pred = grid.best_estimator_.predict(X_val_vec)
        f1 = f1_score(y_val, y_pred, average='binary', pos_label=1)
        precision = precision_score(y_val, y_pred, average='binary', pos_label=1)
        recall = recall_score(y_val, y_pred, average='binary', pos_label=1)
        
        print(f"  ✅ F1: {f1:.4f} ({f1*100:.2f}%)")
        print(f"  🎯 Precision: {precision:.4f}")
        print(f"  📈 Recall: {recall:.4f}")
        print(f"  🔧 Best params: {grid.best_params_}")
        
        # Check if best
        if f1 > best_f1:
            best_f1 = f1
            best_config = combo_name
            best_model = grid.best_estimator_
            print(f"  🏆 NEW BEST!")
    
    return best_f1, best_config, best_model

def optimize_weighted_voting(models, vectorizer, X_train, X_val, y_train, y_val):
    """Optimize weighted voting ensembles"""
    print("\n⚖️ WEIGHTED VOTING OPTIMIZATION")
    print("-" * 40)
    
    # Transform data
    X_train_vec = vectorizer.transform(X_train)
    X_val_vec = vectorizer.transform(X_val)
    
    # Get individual performances for weighting
    performances = {}
    for name, model in models.items():
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
    
    # Test top model combinations with weights
    combos = [
        ("top2", ["logistic", "svm"]),
        ("top3", ["logistic", "svm", "naive_bayes"])
    ]
    
    for combo_name, model_names in combos:
        print(f"\n🎯 {combo_name.upper()}: {model_names}")
        
        estimators = [(name, models[name]) for name in model_names]
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
    
    return best_f1, best_config, best_model

def main():
    """Main optimization execution"""
    
    # Load models and data
    models, vectorizer, X_train, X_val, y_train, y_val = load_models_and_data()
    
    # Optimization 1: Stacking hyperparameters
    stack_f1, stack_config, stack_model = optimize_stacking_hyperparameters(
        models, vectorizer, X_train, X_val, y_train, y_val
    )
    
    # Optimization 2: Weighted voting
    vote_f1, vote_config, vote_model = optimize_weighted_voting(
        models, vectorizer, X_train, X_val, y_train, y_val
    )
    
    # Final results
    print("\n🎉 OPTIMIZATION RESULTS SUMMARY")
    print("=" * 50)
    
    baseline_f1 = 0.9375  # Day 4 baseline
    print(f"📊 Baseline (Day 4): {baseline_f1*100:.2f}% F1-Score")
    print(f"🏆 Best Stacking: {stack_f1*100:.2f}% F1 ({stack_config})")
    print(f"🗳️ Best Voting: {vote_f1*100:.2f}% F1 ({vote_config})")
    
    # Overall best
    if stack_f1 > vote_f1:
        best_f1, best_type, best_model = stack_f1, "stacking", stack_model
    else:
        best_f1, best_type, best_model = vote_f1, "voting", vote_model
    
    improvement = (best_f1 - baseline_f1) * 100
    
    print(f"\n🏆 OVERALL BEST: {best_f1*100:.2f}% F1-Score ({best_type})")
    print(f"📈 Improvement: +{improvement:.2f} percentage points")
    print(f"🎯 94% Target: {'✅ ACHIEVED!' if best_f1 >= 0.94 else f'❌ {(0.94-best_f1)*100:.2f} points remaining'}")
    
    # Save best model
    if best_model:
        timestamp = datetime.now().strftime("%d%m%Y_%H%M%S")
        model_path = Path("ensemble_models") / f"optimized_best_{timestamp}.joblib"
        joblib.dump(best_model, model_path)
        print(f"💾 Best model saved: {model_path}")
    
    return best_f1 >= 0.94

if __name__ == "__main__":
    success = main()
    print(f"\n{'✅ 94% TARGET ACHIEVED!' if success else '⚠️ Continue optimization needed'}") 