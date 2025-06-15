import numpy as np
import pandas as pd
from scipy.sparse import csr_matrix

print("🔍 DS-004 PERFORMANCE DIAGNOSTIC")
print("=" * 50)

# Load validation and test data
val_df = pd.read_csv('data/processed/validation.csv')
test_df = pd.read_csv('data/processed/test.csv')

# Load feature matrices
X_train_raw = np.load('data/features/train_features_none.npy')
X_train_std = np.load('data/features/train_features_standard.npy')

val_sparse = np.load('data/features/val_features.npz', allow_pickle=True)
X_val_sparse = csr_matrix((val_sparse['data'], val_sparse['indices'], val_sparse['indptr']), shape=val_sparse['shape'])
X_val = X_val_sparse.toarray()

test_sparse = np.load('data/features/test_features.npz', allow_pickle=True)
X_test_sparse = csr_matrix((test_sparse['data'], test_sparse['indices'], test_sparse['indptr']), shape=test_sparse['shape'])
X_test = X_test_sparse.toarray()

print("📊 FEATURE SCALING ANALYSIS:")
print("-" * 30)
print(f"Train Raw:  min={X_train_raw.min():.3f}, max={X_train_raw.max():.3f}, mean={X_train_raw.mean():.3f}")
print(f"Train Std:  min={X_train_std.min():.3f}, max={X_train_std.max():.3f}, mean={X_train_std.mean():.3f}")
print(f"Val Features: min={X_val.min():.3f}, max={X_val.max():.3f}, mean={X_val.mean():.3f}")
print(f"Test Features: min={X_test.min():.3f}, max={X_test.max():.3f}, mean={X_test.mean():.3f}")

# Check class distribution
label_map = {'ham': 0, 'spam': 1}
y_train = pd.read_csv('data/processed/train.csv')['label'].map(label_map).values
y_val = val_df['label'].map(label_map).values
y_test = test_df['label'].map(label_map).values

print(f"\n🏷️ CLASS DISTRIBUTION ANALYSIS:")
print("-" * 30)

for name, y in [("Train", y_train), ("Val", y_val), ("Test", y_test)]:
    class_dist = pd.Series(y).value_counts().sort_index()
    print(f"{name}: Ham={class_dist[0]} ({class_dist[0]/len(y)*100:.1f}%), Spam={class_dist[1]} ({class_dist[1]/len(y)*100:.1f}%), Ratio={class_dist[0]/class_dist[1]:.1f}:1")

print(f"\n🔍 LIKELY ISSUES:")
print("-" * 30)

# Check if val/test features match standardized training
val_like_std = abs(X_val.mean()) < 0.1 and abs(X_val.std() - 1.0) < 0.2
val_like_raw = X_val.min() >= 0 and X_val.max() > 10

print(f"Val features look like standardized: {val_like_std}")
print(f"Val features look like raw: {val_like_raw}")

if not val_like_std and not val_like_raw:
    print("⚠️ WARNING: Validation features don't match either raw or standardized training features!")
    
# Check if extreme class imbalance is causing issues
val_spam_ratio = pd.Series(y_val).value_counts()[1] / len(y_val)
if val_spam_ratio < 0.05:
    print(f"⚠️ WARNING: Very low spam ratio in validation set ({val_spam_ratio*100:.1f}%)")

print(f"\n💡 RECOMMENDATIONS:")
print("-" * 30)
print("1. Use consistent feature scaling across train/val/test")
print("2. Try less aggressive class balancing (3:1 instead of 1:1)")
print("3. Verify validation features match training preprocessing")
print("4. Test with reduced feature set (top 200 features)") 