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

print("🚀 DS-004 BASELINE MODELS LAUNCH!")
print(f"📅 Start: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")

# Load data
train_df = pd.read_csv('data/processed/train.csv')
val_df = pd.read_csv('data/processed/validation.csv')
test_df = pd.read_csv('data/processed/test.csv')

# Load training features (both raw and standardized)
X_train_raw = np.load('data/features/train_features_none.npy')  # For Naive Bayes
X_train_std = np.load('data/features/train_features_standard.npy')  # For other models

# Load validation features (sparse format)
val_sparse = np.load('data/features/val_features.npz', allow_pickle=True)
X_val_sparse = csr_matrix((val_sparse['data'], val_sparse['indices'], val_sparse['indptr']), shape=val_sparse['shape'])
X_val = X_val_sparse.toarray()

# Load test features (sparse format)
test_sparse = np.load('data/features/test_features.npz', allow_pickle=True)
X_test_sparse = csr_matrix((test_sparse['data'], test_sparse['indices'], test_sparse['indptr']), shape=test_sparse['shape'])
X_test = X_test_sparse.toarray()

# Target variables - convert text labels to numeric
# Map 'ham' -> 0, 'spam' -> 1
label_map = {'ham': 0, 'spam': 1}
y_train = train_df['label'].map(label_map).values
y_val = val_df['label'].map(label_map).values
y_test = test_df['label'].map(label_map).values

print(f"✅ Data loaded: train {X_train_std.shape}, val {X_val.shape}, test {X_test.shape}")

# Class imbalance analysis
class_dist = pd.Series(y_train).value_counts().sort_index()
print(f"🏷️ Class distribution:")
print(f"   Ham (0): {class_dist[0]} samples")
print(f"   Spam (1): {class_dist[1]} samples")
print(f"   Imbalance ratio: {class_dist[0]/class_dist[1]:.1f}:1 (Ham:Spam)")

# Manual oversampling for class balance - create balanced sets for both feature types
print("\n⚖️ Creating balanced training sets...")

# For raw features (Naive Bayes)
df_train_raw = pd.DataFrame(X_train_raw)
df_train_raw['label'] = y_train
df_majority_raw = df_train_raw[df_train_raw.label == 0]  # Ham
df_minority_raw = df_train_raw[df_train_raw.label == 1]  # Spam
df_minority_upsampled_raw = resample(df_minority_raw, replace=True, n_samples=len(df_majority_raw), random_state=42)
df_upsampled_raw = pd.concat([df_majority_raw, df_minority_upsampled_raw])
X_train_raw_balanced = df_upsampled_raw.drop('label', axis=1).values
y_train_raw_balanced = df_upsampled_raw['label'].values

# For standardized features (other models)
df_train_std = pd.DataFrame(X_train_std)
df_train_std['label'] = y_train
df_majority_std = df_train_std[df_train_std.label == 0]  # Ham
df_minority_std = df_train_std[df_train_std.label == 1]  # Spam
df_minority_upsampled_std = resample(df_minority_std, replace=True, n_samples=len(df_majority_std), random_state=42)
df_upsampled_std = pd.concat([df_majority_std, df_minority_upsampled_std])
X_train_std_balanced = df_upsampled_std.drop('label', axis=1).values
y_train_std_balanced = df_upsampled_std['label'].values

print(f"Raw balanced set: {len(y_train_raw_balanced)} samples")
print(f"Standardized balanced set: {len(y_train_std_balanced)} samples")

# Initialize model storage
models = {}
results = {}

print("\n🤖 TRAINING BASELINE MODELS:")
print("-" * 50)

# Multinomial Naive Bayes (using raw features)
print("Training Multinomial Naive Bayes (raw features)...")
mnb = MultinomialNB()
mnb.fit(X_train_raw_balanced, y_train_raw_balanced)
y_pred_mnb = mnb.predict(X_val)
models['MultinomialNB'] = mnb
results['MultinomialNB'] = {
    'f1': f1_score(y_val, y_pred_mnb), 
    'precision': precision_score(y_val, y_pred_mnb), 
    'recall': recall_score(y_val, y_pred_mnb)
}
print(f"   F1: {results['MultinomialNB']['f1']:.4f}")

# Logistic Regression (using standardized features)
print("Training Logistic Regression (standardized features)...")
lr = LogisticRegression(random_state=42, max_iter=1000)
lr.fit(X_train_std_balanced, y_train_std_balanced)
y_pred_lr = lr.predict(X_val)
models['LogisticRegression'] = lr
results['LogisticRegression'] = {
    'f1': f1_score(y_val, y_pred_lr), 
    'precision': precision_score(y_val, y_pred_lr), 
    'recall': recall_score(y_val, y_pred_lr)
}
print(f"   F1: {results['LogisticRegression']['f1']:.4f}")

# Random Forest (using standardized features)
print("Training Random Forest (standardized features)...")
rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
rf.fit(X_train_std_balanced, y_train_std_balanced)
y_pred_rf = rf.predict(X_val)
models['RandomForest'] = rf
results['RandomForest'] = {
    'f1': f1_score(y_val, y_pred_rf), 
    'precision': precision_score(y_val, y_pred_rf), 
    'recall': recall_score(y_val, y_pred_rf)
}
print(f"   F1: {results['RandomForest']['f1']:.4f}")

# Class-weighted models (using original imbalanced data)
print("\n💰 CLASS-WEIGHTED MODELS:")
print("-" * 50)

# Logistic Regression with class weights
print("Training Logistic Regression (Class Weighted)...")
lr_weighted = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
lr_weighted.fit(X_train_std, y_train)
y_pred_lr_weighted = lr_weighted.predict(X_val)
models['LogisticRegression_Weighted'] = lr_weighted
results['LogisticRegression_Weighted'] = {
    'f1': f1_score(y_val, y_pred_lr_weighted), 
    'precision': precision_score(y_val, y_pred_lr_weighted), 
    'recall': recall_score(y_val, y_pred_lr_weighted)
}
print(f"   F1: {results['LogisticRegression_Weighted']['f1']:.4f}")

# Random Forest with class weights
print("Training Random Forest (Class Weighted)...")
rf_weighted = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1, class_weight='balanced')
rf_weighted.fit(X_train_std, y_train)
y_pred_rf_weighted = rf_weighted.predict(X_val)
models['RandomForest_Weighted'] = rf_weighted
results['RandomForest_Weighted'] = {
    'f1': f1_score(y_val, y_pred_rf_weighted), 
    'precision': precision_score(y_val, y_pred_rf_weighted), 
    'recall': recall_score(y_val, y_pred_rf_weighted)
}
print(f"   F1: {results['RandomForest_Weighted']['f1']:.4f}")

# Results analysis
print("\n" + "=" * 80)
print("📊 BASELINE MODELS PERFORMANCE SUMMARY")
print("=" * 80)

results_df = pd.DataFrame(results).T.round(4)
results_df = results_df.sort_values('f1', ascending=False)
print(results_df.to_string())

best_model_name = results_df['f1'].idxmax()
best_model = models[best_model_name]
best_f1 = results_df.loc[best_model_name, 'f1']
best_precision = results_df.loc[best_model_name, 'precision']
best_recall = results_df.loc[best_model_name, 'recall']

print(f"\n🏆 BEST PERFORMING MODEL: {best_model_name}")
print(f"   F1-Score: {best_f1:.4f}")
print(f"   Precision: {best_precision:.4f}")
print(f"   Recall: {best_recall:.4f}")

# Target achievement analysis
target_f1, target_precision, target_recall = 0.90, 0.92, 0.88
print(f"\n🎯 TARGET ACHIEVEMENT ANALYSIS:")
print("-" * 50)
print(f"F1-Score Target (≥{target_f1:.2f}): {'✅ ACHIEVED' if best_f1 >= target_f1 else '❌ NOT MET'} ({best_f1:.4f})")
print(f"Precision Target (≥{target_precision:.2f}): {'✅ ACHIEVED' if best_precision >= target_precision else '❌ NOT MET'} ({best_precision:.4f})")
print(f"Recall Target (≥{target_recall:.2f}): {'✅ ACHIEVED' if best_recall >= target_recall else '❌ NOT MET'} ({best_recall:.4f})")

# Final test set evaluation
print(f"\n🔬 FINAL TEST SET EVALUATION:")
print("-" * 50)
y_pred_test = best_model.predict(X_test)
test_f1 = f1_score(y_test, y_pred_test)
test_precision = precision_score(y_test, y_pred_test)
test_recall = recall_score(y_test, y_pred_test)

print(f"Test Set Performance ({best_model_name}):")
print(f"   F1-Score: {test_f1:.4f}")
print(f"   Precision: {test_precision:.4f}")
print(f"   Recall: {test_recall:.4f}")

# Save models and results
print(f"\n📦 MODEL SERIALIZATION:")
print("-" * 50)
joblib.dump(best_model, 'models/best_baseline_model.pkl')
results_df.to_csv('models/baseline_results.csv')
print("✅ Best model saved: models/best_baseline_model.pkl")
print("✅ Results saved: models/baseline_results.csv")

# Final summary
all_targets_met = (best_f1 >= target_f1 and best_precision >= target_precision and best_recall >= target_recall)
print(f"\n" + "=" * 80)
print("🎉 DS-004 BASELINE MODELS IMPLEMENTATION COMPLETE!")
print("=" * 80)
print(f"📅 Completion time: {datetime.now().strftime('%d/%m/%Y %H:%M:%S')}")
print(f"🏆 Best Model: {best_model_name}")
print(f"📊 Validation F1: {best_f1:.4f} | Test F1: {test_f1:.4f}")
print(f"🎯 Target Status: {'✅ ALL TARGETS MET' if all_targets_met else '⚠️ PARTIAL ACHIEVEMENT'}")
print("🚀 Ready for DS-005 Advanced Models!")
print("=" * 80) 