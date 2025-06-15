#!/usr/bin/env python3
"""
SMS Spam Detection - DS-004: Class Imbalance & Baseline Models (Fixed)
Author: Data Scientist
Date: 15/06/2025
Target: F1≥90%, Precision≥92%, Recall≥88%
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
import joblib
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Scikit-learn imports
from sklearn.model_selection import cross_val_score, StratifiedKFold
from sklearn.metrics import (
    classification_report, confusion_matrix, roc_auc_score, 
    precision_recall_curve, roc_curve, f1_score,
    precision_score, recall_score, average_precision_score
)

# Model imports
from sklearn.naive_bayes import MultinomialNB, ComplementNB
from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

# Class imbalance handling (using sklearn built-in)
from sklearn.utils.class_weight import compute_class_weight
from sklearn.utils import resample

print("=" * 80)
print("🚀 DS-004 BASELINE MODELS & CLASS IMBALANCE HANDLING (FIXED)")
print("=" * 80)
print(f"📅 Launch time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print("🎯 Target: F1≥90%, Precision≥92%, Recall≥88%")
print()

# =============================================================================
# PHASE 1: LOAD FOUNDATION ASSETS
# =============================================================================
print("📁 PHASE 1: LOADING FOUNDATION ASSETS")
print("-" * 50)

# Load processed data
print("Loading processed datasets...")
train_df = pd.read_csv('data/processed/train.csv')
val_df = pd.read_csv('data/processed/validation.csv')
test_df = pd.read_csv('data/processed/test.csv')

# Load feature matrices (1,000 optimal features)
print("Loading feature matrices...")
X_train = np.load('data/features/train_features_standard.npy')
X_val_sparse = np.load('data/features/val_features.npz', allow_pickle=True)
X_test_sparse = np.load('data/features/test_features.npz', allow_pickle=True)

# Extract feature matrices
X_val = X_val_sparse['features'].toarray() if hasattr(X_val_sparse['features'], 'toarray') else X_val_sparse['features']
X_test = X_test_sparse['features'].toarray() if hasattr(X_test_sparse['features'], 'toarray') else X_test_sparse['features']

# Target variables
y_train = train_df['label'].values
y_val = val_df['label'].values
y_test = test_df['label'].values

print(f"✅ Datasets loaded successfully:")
print(f"   Training: {X_train.shape} features, {len(y_train)} samples")
print(f"   Validation: {X_val.shape} features, {len(y_val)} samples")
print(f"   Test: {X_test.shape} features, {len(y_test)} samples")

# Analyze class imbalance
train_class_dist = pd.Series(y_train).value_counts()
print(f"\n🏷️ Class distribution analysis:")
print(f"   Ham (0): {train_class_dist[0]} ({train_class_dist[0]/len(y_train)*100:.1f}%)")
print(f"   Spam (1): {train_class_dist[1]} ({train_class_dist[1]/len(y_train)*100:.1f}%)")
print(f"   Imbalance ratio: {train_class_dist[0]/train_class_dist[1]:.1f}:1")

print("\n✅ Foundation assets loaded - Ready for baseline models!")

# =============================================================================
# PHASE 2: CLASS IMBALANCE HANDLING (WITHOUT EXTERNAL LIBRARIES)
# =============================================================================
print("\n" + "=" * 80)
print("⚖️ PHASE 2: CLASS IMBALANCE HANDLING (SKLEARN BUILT-IN)")
print("=" * 80)

# Baseline performance (no balancing)
print("📊 2.1 BASELINE ASSESSMENT (Raw Imbalanced Data)")
print("-" * 50)

# Quick baseline with Logistic Regression
lr_baseline = LogisticRegression(random_state=42, max_iter=1000)
lr_baseline.fit(X_train, y_train)
y_pred_baseline = lr_baseline.predict(X_val)

baseline_f1 = f1_score(y_val, y_pred_baseline)
baseline_precision = precision_score(y_val, y_pred_baseline)
baseline_recall = recall_score(y_val, y_pred_baseline)

print(f"Baseline Performance (Imbalanced):")
print(f"   F1-Score: {baseline_f1:.4f}")
print(f"   Precision: {baseline_precision:.4f}")
print(f"   Recall: {baseline_recall:.4f}")

# Manual oversampling (Simple duplication approach)
print("\n⚖️ 2.2 MANUAL OVERSAMPLING IMPLEMENTATION")
print("-" * 50)

# Separate majority and minority classes
df_train = pd.DataFrame(X_train)
df_train['label'] = y_train

df_majority = df_train[df_train.label == 0]
df_minority = df_train[df_train.label == 1]

print(f"Original class distribution:")
print(f"   Majority (Ham): {len(df_majority)} samples")
print(f"   Minority (Spam): {len(df_minority)} samples")

# Upsample minority class
df_minority_upsampled = resample(df_minority, 
                                replace=True,
                                n_samples=len(df_majority),
                                random_state=42)

# Combine majority class with upsampled minority class
df_upsampled = pd.concat([df_majority, df_minority_upsampled])

# Separate features and target
X_train_balanced = df_upsampled.drop('label', axis=1).values
y_train_balanced = df_upsampled['label'].values

print(f"Balanced class distribution:")
print(f"   Total samples: {len(y_train_balanced)}")
balanced_class_dist = pd.Series(y_train_balanced).value_counts()
print(f"   Ham: {balanced_class_dist[0]}, Spam: {balanced_class_dist[1]}")

# Test balanced performance
lr_balanced = LogisticRegression(random_state=42, max_iter=1000)
lr_balanced.fit(X_train_balanced, y_train_balanced)
y_pred_balanced = lr_balanced.predict(X_val)

balanced_f1 = f1_score(y_val, y_pred_balanced)
balanced_precision = precision_score(y_val, y_pred_balanced)
balanced_recall = recall_score(y_val, y_pred_balanced)

print(f"Balanced Performance:")
print(f"   F1-Score: {balanced_f1:.4f} (Δ{balanced_f1-baseline_f1:+.4f})")
print(f"   Precision: {balanced_precision:.4f} (Δ{balanced_precision-baseline_precision:+.4f})")
print(f"   Recall: {balanced_recall:.4f} (Δ{balanced_recall-baseline_recall:+.4f})")

# =============================================================================
# PHASE 3: BASELINE MODEL IMPLEMENTATION
# =============================================================================
print("\n" + "=" * 80)
print("🤖 PHASE 3: BASELINE MODEL IMPLEMENTATION")
print("=" * 80)

models = {}
results = {}

# Use the best performing training data (balanced vs class_weight)
X_train_final = X_train_balanced
y_train_final = y_train_balanced

# 3.1 Naive Bayes Models
print("📈 3.1 NAIVE BAYES MODELS")
print("-" * 50)

# Multinomial Naive Bayes
print("Training Multinomial Naive Bayes...")
mnb = MultinomialNB()
mnb.fit(X_train_final, y_train_final)
y_pred_mnb = mnb.predict(X_val)

mnb_f1 = f1_score(y_val, y_pred_mnb)
mnb_precision = precision_score(y_val, y_pred_mnb)
mnb_recall = recall_score(y_val, y_pred_mnb)

models['MultinomialNB'] = mnb
results['MultinomialNB'] = {
    'f1': mnb_f1, 'precision': mnb_precision, 'recall': mnb_recall
}

print(f"Multinomial NB Performance:")
print(f"   F1-Score: {mnb_f1:.4f}")
print(f"   Precision: {mnb_precision:.4f}")
print(f"   Recall: {mnb_recall:.4f}")

# Complement Naive Bayes
print("\nTraining Complement Naive Bayes...")
cnb = ComplementNB()
cnb.fit(X_train_final, y_train_final)
y_pred_cnb = cnb.predict(X_val)

cnb_f1 = f1_score(y_val, y_pred_cnb)
cnb_precision = precision_score(y_val, y_pred_cnb)
cnb_recall = recall_score(y_val, y_pred_cnb)

models['ComplementNB'] = cnb
results['ComplementNB'] = {
    'f1': cnb_f1, 'precision': cnb_precision, 'recall': cnb_recall
}

print(f"Complement NB Performance:")
print(f"   F1-Score: {cnb_f1:.4f}")
print(f"   Precision: {cnb_precision:.4f}")
print(f"   Recall: {cnb_recall:.4f}")

# 3.2 Support Vector Machine
print("\n🎯 3.2 SUPPORT VECTOR MACHINE")
print("-" * 50)

print("Training Linear SVM...")
svm = LinearSVC(random_state=42, max_iter=2000)
svm.fit(X_train_final, y_train_final)
y_pred_svm = svm.predict(X_val)

svm_f1 = f1_score(y_val, y_pred_svm)
svm_precision = precision_score(y_val, y_pred_svm)
svm_recall = recall_score(y_val, y_pred_svm)

models['LinearSVM'] = svm
results['LinearSVM'] = {
    'f1': svm_f1, 'precision': svm_precision, 'recall': svm_recall
}

print(f"Linear SVM Performance:")
print(f"   F1-Score: {svm_f1:.4f}")
print(f"   Precision: {svm_precision:.4f}")
print(f"   Recall: {svm_recall:.4f}")

# 3.3 Logistic Regression
print("\n📊 3.3 LOGISTIC REGRESSION")
print("-" * 50)

print("Training Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_final, y_train_final)
y_pred_lr = lr.predict(X_val)

lr_f1 = f1_score(y_val, y_pred_lr)
lr_precision = precision_score(y_val, y_pred_lr)
lr_recall = recall_score(y_val, y_pred_lr)

models['LogisticRegression'] = lr
results['LogisticRegression'] = {
    'f1': lr_f1, 'precision': lr_precision, 'recall': lr_recall
}

print(f"Logistic Regression Performance:")
print(f"   F1-Score: {lr_f1:.4f}")
print(f"   Precision: {lr_precision:.4f}")
print(f"   Recall: {lr_recall:.4f}")

# 3.4 Random Forest
print("\n🌳 3.4 RANDOM FOREST")
print("-" * 50)

print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_final, y_train_final)
y_pred_rf = rf.predict(X_val)

rf_f1 = f1_score(y_val, y_pred_rf)
rf_precision = precision_score(y_val, y_pred_rf)
rf_recall = recall_score(y_val, y_pred_rf)

models['RandomForest'] = rf
results['RandomForest'] = {
    'f1': rf_f1, 'precision': rf_precision, 'recall': rf_recall
}

print(f"Random Forest Performance:")
print(f"   F1-Score: {rf_f1:.4f}")
print(f"   Precision: {rf_precision:.4f}")
print(f"   Recall: {rf_recall:.4f}")

# Also test with class_weight='balanced' models
print("\n💰 3.5 CLASS-WEIGHTED MODELS")
print("-" * 50)

# Test class-weighted versions
models_weighted = {}
results_weighted = {}

# Logistic Regression with class weights
lr_weighted = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_weighted.fit(X_train, y_train)
y_pred_lr_weighted = lr_weighted.predict(X_val)

lr_weighted_f1 = f1_score(y_val, y_pred_lr_weighted)
lr_weighted_precision = precision_score(y_val, y_pred_lr_weighted)
lr_weighted_recall = recall_score(y_val, y_pred_lr_weighted)

models_weighted['LogisticRegression_Weighted'] = lr_weighted
results_weighted['LogisticRegression_Weighted'] = {
    'f1': lr_weighted_f1, 'precision': lr_weighted_precision, 'recall': lr_weighted_recall
}

print(f"Logistic Regression (Class Weighted) Performance:")
print(f"   F1-Score: {lr_weighted_f1:.4f}")
print(f"   Precision: {lr_weighted_precision:.4f}")
print(f"   Recall: {lr_weighted_recall:.4f}")

# Random Forest with class weights
rf_weighted = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf_weighted.fit(X_train, y_train)
y_pred_rf_weighted = rf_weighted.predict(X_val)

rf_weighted_f1 = f1_score(y_val, y_pred_rf_weighted)
rf_weighted_precision = precision_score(y_val, y_pred_rf_weighted)
rf_weighted_recall = recall_score(y_val, y_pred_rf_weighted)

models_weighted['RandomForest_Weighted'] = rf_weighted
results_weighted['RandomForest_Weighted'] = {
    'f1': rf_weighted_f1, 'precision': rf_weighted_precision, 'recall': rf_weighted_recall
}

print(f"Random Forest (Class Weighted) Performance:")
print(f"   F1-Score: {rf_weighted_f1:.4f}")
print(f"   Precision: {rf_weighted_precision:.4f}")
print(f"   Recall: {rf_weighted_recall:.4f}")

# =============================================================================
# PHASE 4: RESULTS ANALYSIS & MODEL SELECTION
# =============================================================================
print("\n" + "=" * 80)
print("📋 PHASE 4: RESULTS ANALYSIS & MODEL SELECTION")
print("=" * 80)

# Combine all results
all_results = {**results, **results_weighted}
all_models = {**models, **models_weighted}

# Create results summary
results_df = pd.DataFrame(all_results).T
results_df = results_df.round(4)
results_df = results_df.sort_values('f1', ascending=False)

print("📊 BASELINE MODELS PERFORMANCE SUMMARY")
print("-" * 50)
print(results_df.to_string())

# Identify best model
best_model_name = results_df['f1'].idxmax()
best_model = all_models[best_model_name]
best_f1 = results_df.loc[best_model_name, 'f1']
best_precision = results_df.loc[best_model_name, 'precision']
best_recall = results_df.loc[best_model_name, 'recall']

print(f"\n🏆 BEST PERFORMING MODEL: {best_model_name}")
print(f"   F1-Score: {best_f1:.4f}")
print(f"   Precision: {best_precision:.4f}")
print(f"   Recall: {best_recall:.4f}")

# Check target achievement
target_f1 = 0.90
target_precision = 0.92
target_recall = 0.88

print(f"\n🎯 TARGET ACHIEVEMENT ANALYSIS")
print("-" * 50)
print(f"F1-Score Target (≥{target_f1:.2f}): {'✅ ACHIEVED' if best_f1 >= target_f1 else '❌ NOT MET'} ({best_f1:.4f})")
print(f"Precision Target (≥{target_precision:.2f}): {'✅ ACHIEVED' if best_precision >= target_precision else '❌ NOT MET'} ({best_precision:.4f})")
print(f"Recall Target (≥{target_recall:.2f}): {'✅ ACHIEVED' if best_recall >= target_recall else '❌ NOT MET'} ({best_recall:.4f})")

# Final test set evaluation
print(f"\n🔬 FINAL TEST SET EVALUATION")
print("-" * 50)
y_pred_test = best_model.predict(X_test)
test_f1 = f1_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)

print(f"Test Set Performance ({best_model_name}):")
print(f"   F1-Score: {test_f1:.4f}")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall: {test_recall:.4f}")

# Model serialization
print(f"\n📦 MODEL SERIALIZATION")
print("-" * 50)
print("Saving best model for production deployment...")
joblib.dump(best_model, f'models/best_baseline_model_{best_model_name.lower()}.pkl')
print(f"✅ Best model saved: models/best_baseline_model_{best_model_name.lower()}.pkl")

# Save results summary
results_df.to_csv('models/baseline_models_results.csv')
print("✅ Results summary saved: models/baseline_models_results.csv")

print("\n" + "=" * 80)
print("🎉 DS-004 BASELINE MODELS IMPLEMENTATION COMPLETE!")
print("=" * 80)
print(f"📅 Completion time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print(f"🏆 Best Model: {best_model_name} (F1: {best_f1:.4f})")
print(f"🔬 Test Set F1: {test_f1:.4f}")
print(f"🎯 Target Status: {'✅ ALL TARGETS MET' if (best_f1 >= target_f1 and best_precision >= target_precision and best_recall >= target_recall) else '⚠️ PARTIAL TARGET ACHIEVEMENT'}")
print("🚀 Ready for DS-005 Advanced Models!")
print("=" * 80) 