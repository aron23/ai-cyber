#!/usr/bin/env python3
"""
COLLAB-001 Day 3: Final Production Implementation
SMS Spam Detection Project - Strategic Excellence Phase
"""

import warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import f1_score, precision_score, recall_score
from scipy.optimize import minimize
import time
import json
from datetime import datetime

print('='*80)
print('🚀 COLLAB-001 DAY 3: FINAL PRODUCTION IMPLEMENTATION')
print('='*80)
print('📅 Day 3 Execution: 16/06/2025 08:55:00')
print('🎯 Mission: Achieve confirmed 95%+ F1-Score with ensemble methods')
print('⚡ Status: Final implementation with corrected label encoding')
print()

start_time = time.time()
baseline_f1 = 0.9467
target_f1 = 0.95

# Load data with corrected labels
print('📊 LOADING PRODUCTION DATA')
print('-' * 50)

train_sparse = sparse.load_npz('data/features/train_features.npz')
val_sparse = sparse.load_npz('data/features/val_features.npz')
test_sparse = sparse.load_npz('data/features/test_features.npz')

X_train = train_sparse.toarray()
X_val = val_sparse.toarray()
X_test = test_sparse.toarray()

train_df = pd.read_csv('data/processed/train.csv')
val_df = pd.read_csv('data/processed/validation.csv')
test_df = pd.read_csv('data/processed/test.csv')

# Correct label encoding: 'spam' = 1, 'ham' = 0
y_train = (train_df['label'] == 'spam').astype(int)
y_val = (val_df['label'] == 'spam').astype(int)
y_test = (test_df['label'] == 'spam').astype(int)

print(f'✅ Data loaded successfully:')
print(f'   Training: {X_train.shape[0]:,} samples, {X_train.shape[1]:,} features')
print(f'   Validation: {X_val.shape[0]:,} samples')
print(f'   Test: {X_test.shape[0]:,} samples')
print(f'📊 Class distribution:')
print(f'   Training: {y_train.mean()*100:.1f}% spam, {(1-y_train.mean())*100:.1f}% ham')
print(f'   Validation: {y_val.mean()*100:.1f}% spam, {(1-y_val.mean())*100:.1f}% ham')

# Train base models
print()
print('🧠 TRAINING PRODUCTION BASE MODELS')
print('-' * 50)

models = {}

# 1. Logistic Regression
print('📊 Training Logistic Regression...')
try:
    lr = LogisticRegression(class_weight='balanced', max_iter=1000, random_state=42, C=1.0)
    lr.fit(X_train, y_train)
    models['logistic'] = lr
    print('   ✅ Logistic Regression trained successfully')
except Exception as e:
    print(f'   ❌ Logistic Regression failed: {e}')

# 2. Random Forest
print('🌳 Training Random Forest...')
try:
    rf = RandomForestClassifier(
        n_estimators=100, 
        class_weight='balanced', 
        random_state=42, 
        n_jobs=-1,
        max_depth=15
    )
    rf.fit(X_train, y_train)
    models['random_forest'] = rf
    print('   ✅ Random Forest trained successfully')
except Exception as e:
    print(f'   ❌ Random Forest failed: {e}')

# 3. Neural Network
print('🧠 Training Neural Network...')
try:
    nn = MLPClassifier(
        hidden_layer_sizes=(256, 128), 
        random_state=42, 
        max_iter=200,
        early_stopping=True,
        validation_fraction=0.1
    )
    nn.fit(X_train, y_train)
    models['neural_network'] = nn
    print('   ✅ Neural Network trained successfully')
except Exception as e:
    print(f'   ❌ Neural Network failed: {e}')

print(f'\n✅ Base models trained: {len(models)} models ready')

if len(models) < 2:
    print('❌ Insufficient models for ensemble - need at least 2')
    exit(1)

# Evaluate base models on validation set
print()
print('📊 EVALUATING BASE MODELS ON VALIDATION SET')
print('-' * 50)

base_results = {}

for name, model in models.items():
    print(f'📈 Evaluating {name.title().replace("_", " ")}...')
    
    try:
        y_pred = model.predict(X_val)
        y_prob = model.predict_proba(X_val)[:, 1]
        
        f1 = f1_score(y_val, y_pred)
        precision = precision_score(y_val, y_pred)
        recall = recall_score(y_val, y_pred)
        
        base_results[name] = {
            'f1_score': f1,
            'precision': precision,
            'recall': recall,
            'probabilities': y_prob,
            'predictions': y_pred
        }
        
        improvement = ((f1 - baseline_f1) / baseline_f1) * 100
        
        print(f'   F1: {f1:.4f} ({f1*100:.2f}%)')
        print(f'   Precision: {precision:.4f}, Recall: {recall:.4f}')
        print(f'   vs Baseline: {improvement:+.2f}%')
        
    except Exception as e:
        print(f'   ❌ Evaluation failed: {e}')

print(f'\n✅ Base model evaluation complete: {len(base_results)} models evaluated')

# Advanced ensemble optimization
print()
print('⚖️  ADVANCED ENSEMBLE OPTIMIZATION')
print('-' * 50)

model_names = list(base_results.keys())
prob_matrix = np.column_stack([base_results[name]['probabilities'] for name in model_names])

print(f'🔍 Optimizing ensemble with {len(model_names)} models...')
print(f'   Models: {", ".join(model_names)}')

# Weighted voting optimization with F1-score objective and recall constraint
def objective(weights):
    weights = np.array(weights)
    weights = weights / weights.sum()  # Normalize
    
    ensemble_prob = np.dot(prob_matrix, weights)
    ensemble_pred = (ensemble_prob >= 0.5).astype(int)
    
    f1 = f1_score(y_val, ensemble_pred)
    recall = recall_score(y_val, ensemble_pred)
    
    # Penalty for low recall
    if recall < 0.88:
        return -(f1 * 0.7)  # Heavy penalty for recall violation
    
    return -f1  # Minimize negative F1

# Initialize with F1-score based weights
f1_scores = [base_results[name]['f1_score'] for name in model_names]
initial_weights = np.array(f1_scores) / sum(f1_scores)

print(f'📊 Initial weights (F1-based):')
for name, weight in zip(model_names, initial_weights):
    print(f'   {name.title().replace("_", " ")}: {weight:.3f}')

# Optimize weights
print('🚀 Running weight optimization...')
try:
    result = minimize(
        objective, 
        initial_weights, 
        method='SLSQP',
        bounds=[(0.05, 0.8) for _ in model_names],  # Prevent extreme weights
        constraints={'type': 'eq', 'fun': lambda w: w.sum() - 1}
    )
    optimal_weights = result.x / result.x.sum()
    print('   ✅ Weight optimization successful')
except Exception as e:
    print(f'   ⚠️ Optimization failed: {e}, using F1-based weights')
    optimal_weights = initial_weights

# Apply optimal weights to create ensemble
ensemble_prob = np.dot(prob_matrix, optimal_weights)
ensemble_pred = (ensemble_prob >= 0.5).astype(int)

# Calculate ensemble performance
ensemble_f1 = f1_score(y_val, ensemble_pred)
ensemble_precision = precision_score(y_val, ensemble_pred)
ensemble_recall = recall_score(y_val, ensemble_pred)

improvement = ((ensemble_f1 - baseline_f1) / baseline_f1) * 100

print()
print('📊 ENSEMBLE VALIDATION RESULTS:')
print('-' * 50)
print('🏋️ Optimized weights:')
for name, weight in zip(model_names, optimal_weights):
    print(f'   {name.title().replace("_", " ")}: {weight:.3f}')

print(f'\n📈 Ensemble performance:')
print(f'   F1-Score: {ensemble_f1:.4f} ({ensemble_f1*100:.2f}%)')
print(f'   Precision: {ensemble_precision:.4f} ({ensemble_precision*100:.2f}%)')
print(f'   Recall: {ensemble_recall:.4f} ({ensemble_recall*100:.2f}%)')
print(f'   Improvement: {improvement:+.2f}% over baseline')

# Target achievement analysis
validation_target_achieved = ensemble_f1 >= target_f1
validation_recall_ok = ensemble_recall >= 0.88

print(f'\n🎯 VALIDATION TARGET ANALYSIS:')
print(f'   Primary Target (F1≥95.0%): {"✅ ACHIEVED" if validation_target_achieved else "❌ NOT ACHIEVED"}')
print(f'   Recall Constraint (≥88%): {"✅ MET" if validation_recall_ok else "❌ VIOLATED"}')

# Proceed to test set if validation successful
if validation_target_achieved and validation_recall_ok:
    print(f'   🌟 VALIDATION SUCCESSFUL - proceeding to test set evaluation!')
    
    # Test set evaluation
    print()
    print('🧪 FINAL TEST SET VALIDATION')
    print('-' * 50)
    
    print('🎯 Applying validated ensemble to test set...')
    
    # Get test predictions from each base model
    test_probs = []
    for name, model in models.items():
        if name in model_names:  # Only use models that were evaluated
            test_prob = model.predict_proba(X_test)[:, 1]
            test_probs.append(test_prob)
            print(f'   ✅ {name.title().replace("_", " ")}: test predictions generated')
    
    # Apply ensemble to test set
    test_prob_matrix = np.column_stack(test_probs)
    test_ensemble_prob = np.dot(test_prob_matrix, optimal_weights)
    test_ensemble_pred = (test_ensemble_prob >= 0.5).astype(int)
    
    # Calculate test performance
    test_f1 = f1_score(y_test, test_ensemble_pred)
    test_precision = precision_score(y_test, test_ensemble_pred)
    test_recall = recall_score(y_test, test_ensemble_pred)
    
    print()
    print('📊 FINAL TEST SET RESULTS:')
    print('-' * 50)
    print(f'   F1-Score: {test_f1:.4f} ({test_f1*100:.2f}%)')
    print(f'   Precision: {test_precision:.4f} ({test_precision*100:.2f}%)')
    print(f'   Recall: {test_recall:.4f} ({test_recall*100:.2f}%)')
    
    # Final success assessment
    final_target_achieved = test_f1 >= target_f1
    final_recall_ok = test_recall >= 0.88
    final_success = final_target_achieved and final_recall_ok
    
    print()
    print('🎯 FINAL TARGET ACHIEVEMENT:')
    print('-' * 50)
    print(f'   F1-Score ≥ 95.0%: {"✅ ACHIEVED" if final_target_achieved else "❌ NOT ACHIEVED"}')
    print(f'   Recall ≥ 88.0%: {"✅ MET" if final_recall_ok else "❌ VIOLATED"}')
    print(f'   Mission Status: {"🌟 ACCOMPLISHED" if final_success else "⚠️ PARTIAL SUCCESS"}')
    
    if final_success:
        stretch_achieved = test_f1 >= 0.955
        print(f'   Stretch Target (≥95.5%): {"🌟 ACHIEVED" if stretch_achieved else "❌ NOT ACHIEVED"}')
    
    # Save results
    timestamp = datetime.now().strftime('%d%m%Y_%H%M%S')
    
    results_summary = {
        'timestamp': timestamp,
        'phase': 'COLLAB-001 Day 3 Final',
        'duration_minutes': (time.time() - start_time) / 60,
        'baseline_f1': baseline_f1,
        'target_f1': target_f1,
        'validation_performance': {
            'f1_score': ensemble_f1,
            'precision': ensemble_precision,
            'recall': ensemble_recall
        },
        'test_performance': {
            'f1_score': test_f1,
            'precision': test_precision,
            'recall': test_recall
        },
        'ensemble_weights': dict(zip(model_names, optimal_weights.tolist())),
        'validation_success': validation_target_achieved and validation_recall_ok,
        'final_success': final_success,
        'models_used': model_names
    }
    
    results_path = f'models/collab001_day3_final_results_{timestamp}.json'
    with open(results_path, 'w') as f:
        json.dump(results_summary, f, indent=2)
    
else:
    print(f'   ⚠️ VALIDATION TARGETS NOT MET - test evaluation skipped')
    final_success = False
    results_path = None

# Final completion summary
duration = (time.time() - start_time) / 60

print()
print('='*80)
print('✅ COLLAB-001 DAY 3 COMPLETION SUMMARY')
print('='*80)
print(f'   Duration: {duration:.1f} minutes')
print(f'   Models Trained: {len(models)}')
print(f'   Ensemble Method: Advanced Weighted Voting with Optimization')
print(f'   Validation F1: {ensemble_f1:.4f} ({ensemble_f1*100:.2f}%)')

if validation_target_achieved and validation_recall_ok:
    print(f'   Test F1: {test_f1:.4f} ({test_f1*100:.2f}%)')
    print(f'   Final Status: {"🌟 MISSION ACCOMPLISHED" if final_success else "⚠️ PARTIAL SUCCESS"}')
else:
    print(f'   Final Status: ⚠️ VALIDATION INCOMPLETE')

if results_path:
    print(f'💾 Results saved: {results_path}')

print()
if final_success:
    print('🎉 COLLAB-001 PROJECT: EXCEPTIONAL SUCCESS!')
    print('🎯 95%+ F1-Score target achieved in production!')
    print('🚀 Advanced ensemble methods successfully implemented!')
    print('📋 Ready for production deployment!')
else:
    print('📊 COLLAB-001 PROJECT: SIGNIFICANT PROGRESS ACHIEVED')
    print('🔧 Strong foundation established for continued optimization')
    print('📋 Multiple deployment options available')

print('='*80) 