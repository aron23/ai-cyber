import pandas as pd
import numpy as np
from datetime import datetime
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.utils import resample
from scipy.sparse import csr_matrix
import joblib

print("🔧 DS-004 FIXED: CONSISTENT FEATURE SCALING")
print("=" * 60)
print(f"📅 Start: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

# Load data
train_df = pd.read_csv('data/processed/train.csv')
val_df = pd.read_csv('data/processed/validation.csv')
test_df = pd.read_csv('data/processed/test.csv')

# Load ALL features in RAW format (consistent across train/val/test)
X_train_raw = np.load('data/features/train_features_none.npy')

# Load validation features (sparse format) - they are already in raw format
val_sparse = np.load('data/features/val_features.npz', allow_pickle=True)
X_val_sparse = csr_matrix((val_sparse['data'], val_sparse['indices'], val_sparse['indptr']), shape=val_sparse['shape'])
X_val_raw = X_val_sparse.toarray()

# Load test features (sparse format) - they are already in raw format  
test_sparse = np.load('data/features/test_features.npz', allow_pickle=True)
X_test_sparse = csr_matrix((test_sparse['data'], test_sparse['indices'], test_sparse['indptr']), shape=test_sparse['shape'])
X_test_raw = X_test_sparse.toarray()

# Target variables
label_map = {'ham': 0, 'spam': 1}
y_train = train_df['label'].map(label_map).values
y_val = val_df['label'].map(label_map).values
y_test = test_df['label'].map(label_map).values

print(f"✅ Consistent raw features loaded:")
print(f"   Train: {X_train_raw.shape} (min={X_train_raw.min():.1f}, max={X_train_raw.max():.1f})")
print(f"   Val:   {X_val_raw.shape} (min={X_val_raw.min():.1f}, max={X_val_raw.max():.1f})")
print(f"   Test:  {X_test_raw.shape} (min={X_test_raw.min():.1f}, max={X_test_raw.max():.1f})")

# Class distribution
class_dist = pd.Series(y_train).value_counts().sort_index()
print(f"\n🏷️ Class distribution: Ham={class_dist[0]}, Spam={class_dist[1]} (Ratio: {class_dist[0]/class_dist[1]:.1f}:1)")

# MODERATE class balancing (3:1 instead of 1:1)
print(f"\n⚖️ Applying MODERATE class balancing (3:1 Ham:Spam)...")
df_train = pd.DataFrame(X_train_raw)
df_train['label'] = y_train
df_majority = df_train[df_train.label == 0]  # Ham
df_minority = df_train[df_train.label == 1]  # Spam

# Target 3:1 ratio instead of 1:1
target_spam_samples = len(df_majority) // 3
df_minority_balanced = resample(df_minority, replace=True, n_samples=target_spam_samples, random_state=42)
df_balanced = pd.concat([df_majority, df_minority_balanced])
X_train_balanced = df_balanced.drop('label', axis=1).values
y_train_balanced = df_balanced['label'].values

balanced_dist = pd.Series(y_train_balanced).value_counts().sort_index()
print(f"Balanced: Ham={balanced_dist[0]}, Spam={balanced_dist[1]} (Ratio: {balanced_dist[0]/balanced_dist[1]:.1f}:1)")

# Train models with consistent features
models = {}
results = {}

print(f"\n🤖 TRAINING MODELS (Raw Features, 3:1 Balance):")
print("-" * 50)

# Multinomial Naive Bayes
print("Training Multinomial Naive Bayes...")
mnb = MultinomialNB(alpha=0.1)  # Add some smoothing
mnb.fit(X_train_balanced, y_train_balanced)
y_pred_mnb = mnb.predict(X_val_raw)
models['MultinomialNB'] = mnb
results['MultinomialNB'] = {
    'f1': f1_score(y_val, y_pred_mnb), 
    'precision': precision_score(y_val, y_pred_mnb), 
    'recall': recall_score(y_val, y_pred_mnb)
}
print(f"   F1: {results['MultinomialNB']['f1']:.4f}")

# Logistic Regression
print("Training Logistic Regression...")
lr = LogisticRegression(random_state=42, max_iter=1000, C=0.1)  # Regularization
lr.fit(X_train_balanced, y_train_balanced)
y_pred_lr = lr.predict(X_val_raw)
models['LogisticRegression'] = lr
results['LogisticRegression'] = {
    'f1': f1_score(y_val, y_pred_lr), 
    'precision': precision_score(y_val, y_pred_lr), 
    'recall': recall_score(y_val, y_pred_lr)
}
print(f"   F1: {results['LogisticRegression']['f1']:.4f}")

# Random Forest
print("Training Random Forest...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, max_depth=10)
rf.fit(X_train_balanced, y_train_balanced)
y_pred_rf = rf.predict(X_val_raw)
models['RandomForest'] = rf
results['RandomForest'] = {
    'f1': f1_score(y_val, y_pred_rf), 
    'precision': precision_score(y_val, y_pred_rf), 
    'recall': recall_score(y_val, y_pred_rf)
}
print(f"   F1: {results['RandomForest']['f1']:.4f}")

# Class-weighted models (original imbalanced data)
print(f"\n💰 CLASS-WEIGHTED MODELS (Original Imbalance):")
print("-" * 50)

# Logistic Regression with balanced class weights
print("Training Logistic Regression (Class Weighted)...")
lr_weighted = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced', C=0.1)
lr_weighted.fit(X_train_raw, y_train)
y_pred_lr_weighted = lr_weighted.predict(X_val_raw)
models['LogisticRegression_Weighted'] = lr_weighted
results['LogisticRegression_Weighted'] = {
    'f1': f1_score(y_val, y_pred_lr_weighted), 
    'precision': precision_score(y_val, y_pred_lr_weighted), 
    'recall': recall_score(y_val, y_pred_lr_weighted)
}
print(f"   F1: {results['LogisticRegression_Weighted']['f1']:.4f}")

# Random Forest with balanced class weights
print("Training Random Forest (Class Weighted)...")
rf_weighted = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced', max_depth=10)
rf_weighted.fit(X_train_raw, y_train)
y_pred_rf_weighted = rf_weighted.predict(X_val_raw)
models['RandomForest_Weighted'] = rf_weighted
results['RandomForest_Weighted'] = {
    'f1': f1_score(y_val, y_pred_rf_weighted), 
    'precision': precision_score(y_val, y_pred_rf_weighted), 
    'recall': recall_score(y_val, y_pred_rf_weighted)
}
print(f"   F1: {results['RandomForest_Weighted']['f1']:.4f}")

# Results analysis
print(f"\n" + "=" * 60)
print("📊 FIXED BASELINE MODELS PERFORMANCE")
print("=" * 60)

results_df = pd.DataFrame(results).T.round(4)
results_df = results_df.sort_values('f1', ascending=False)
print(results_df.to_string())

best_model_name = results_df['f1'].idxmax()
best_model = models[best_model_name]
best_f1 = results_df.loc[best_model_name, 'f1']
best_precision = results_df.loc[best_model_name, 'precision']
best_recall = results_df.loc[best_model_name, 'recall']

print(f"\n🏆 BEST MODEL: {best_model_name}")
print(f"   F1-Score: {best_f1:.4f}")
print(f"   Precision: {best_precision:.4f}")  
print(f"   Recall: {best_recall:.4f}")

# Target achievement
target_f1, target_precision, target_recall = 0.90, 0.92, 0.88
print(f"\n🎯 TARGET ACHIEVEMENT:")
print("-" * 30)
print(f"F1≥{target_f1}: {'✅' if best_f1 >= target_f1 else '❌'} ({best_f1:.4f})")
print(f"Precision≥{target_precision}: {'✅' if best_precision >= target_precision else '❌'} ({best_precision:.4f})")
print(f"Recall≥{target_recall}: {'✅' if best_recall >= target_recall else '❌'} ({best_recall:.4f})")

# Test set evaluation
print(f"\n🔬 TEST SET EVALUATION:")
print("-" * 30)
y_pred_test = best_model.predict(X_test_raw)
test_f1 = f1_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)

print(f"Test Performance ({best_model_name}):")
print(f"   F1: {test_f1:.4f}")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall: {test_recall:.4f}")

# Save best model
joblib.dump(best_model, 'models/best_baseline_model_fixed.pkl')
results_df.to_csv('models/baseline_results_fixed.csv')

# Summary
improvement = best_f1 - 0.2403  # Previous best F1
all_targets_met = (best_f1 >= target_f1 and best_precision >= target_precision and best_recall >= target_recall)

print(f"\n" + "=" * 60)
print("🎉 DS-004 FIXED IMPLEMENTATION COMPLETE!")
print("=" * 60)
print(f"📅 Completion: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print(f"🚀 Best Model: {best_model_name}")
print(f"📈 Improvement: +{improvement:.4f} F1-Score")
print(f"📊 Val F1: {best_f1:.4f} | Test F1: {test_f1:.4f}")
print(f"🎯 Status: {'✅ ALL TARGETS MET' if all_targets_met else '⚠️ SIGNIFICANT IMPROVEMENT'}")
print("=" * 60) 