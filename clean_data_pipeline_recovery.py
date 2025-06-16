#!/usr/bin/env python3
"""
Clean Data Pipeline Recovery - Day 1 Critical Implementation
Eliminate data leakage and implement legitimate NLP processing
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import LabelEncoder
from scipy.sparse import save_npz
import os
import re
import hashlib

print('=== CLEAN DATA PIPELINE RECOVERY - DAY 1 ===')
print('Deadline: 16/06/2025 18:00 (CRITICAL)')
print('Objective: Zero data leakage + Legitimate NLP')
print()

# Step 1: Load original SMS data
print('STEP 1: Loading Original SMS Data')
try:
    # Read the original SMS collection file
    with open('data/SMSSPamCollection', 'r', encoding='latin-1') as f:
        lines = f.readlines()
    
    print(f'✓ Loaded {len(lines)} raw SMS messages')
    
    # Parse the tab-separated format
    data = []
    for line in lines:
        line = line.strip()
        if line:  # Skip empty lines
            parts = line.split('\t', 1)  # Split on first tab only
            if len(parts) == 2:
                label, message = parts
                data.append({'label': label.strip(), 'message': message.strip()})
            else:
                print(f'Warning: Skipping malformed line: {line[:50]}...')
    
    df_original = pd.DataFrame(data)
    print(f'✓ Parsed {len(df_original)} valid messages')
    print(f'✓ Labels: {df_original["label"].value_counts().to_dict()}')
    
except Exception as e:
    print(f'❌ Error loading original data: {e}')
    exit(1)

# Step 2: Remove exact duplicates (legitimate preprocessing)
print('\nSTEP 2: Removing Exact Duplicates')
original_count = len(df_original)
df_original = df_original.drop_duplicates(subset=['message'], keep='first')
final_count = len(df_original)
duplicates_removed = original_count - final_count

print(f'✓ Removed {duplicates_removed} exact duplicates')
print(f'✓ Final dataset: {final_count} unique messages')

# Step 3: Verify message quality
print('\nSTEP 3: Message Quality Verification')
avg_length = df_original['message'].str.len().mean()
avg_words = df_original['message'].str.split().str.len().mean()
print(f'✓ Average message length: {avg_length:.1f} characters')
print(f'✓ Average words per message: {avg_words:.1f} words')

if avg_length < 30:
    print('⚠️  Warning: Messages seem short, but proceeding with original data')

# Step 4: Create clean train/val/test splits with ZERO overlap
print('\nSTEP 4: Creating Clean Splits (Zero Overlap)')

# Create unique message hashes to detect any potential overlap
df_original['message_hash'] = df_original['message'].apply(
    lambda x: hashlib.md5(x.encode()).hexdigest()
)

# Stratified split: 60% train, 20% val, 20% test
X = df_original['message'].values
y = df_original['label'].values

# First split: 60% train, 40% temp
X_train, X_temp, y_train, y_temp = train_test_split(
    X, y, test_size=0.4, random_state=42, stratify=y
)

# Second split: 20% val, 20% test from temp
X_val, X_test, y_val, y_test = train_test_split(
    X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp
)

print(f'✓ Train: {len(X_train)} samples ({len(X_train)/len(X)*100:.1f}%)')
print(f'✓ Val: {len(X_val)} samples ({len(X_val)/len(X)*100:.1f}%)')
print(f'✓ Test: {len(X_test)} samples ({len(X_test)/len(X)*100:.1f}%)')

# Step 5: CRITICAL - Verify zero overlap
print('\nSTEP 5: CRITICAL - Zero Overlap Verification')
train_set = set(X_train)
val_set = set(X_val)
test_set = set(X_test)

overlap_train_val = len(train_set.intersection(val_set))
overlap_train_test = len(train_set.intersection(test_set))
overlap_val_test = len(val_set.intersection(test_set))

print(f'Train-Val overlap: {overlap_train_val} (MUST be 0)')
print(f'Train-Test overlap: {overlap_train_test} (MUST be 0)')
print(f'Val-Test overlap: {overlap_val_test} (MUST be 0)')

if overlap_train_val == 0 and overlap_train_test == 0 and overlap_val_test == 0:
    print('✅ SUCCESS: Zero data leakage confirmed!')
else:
    print('🚨 CRITICAL ERROR: Data leakage still exists!')
    exit(1)

# Step 6: Check class distribution consistency
print('\nSTEP 6: Class Distribution Verification')
def get_spam_rate(labels):
    return (labels == 'spam').mean()

train_spam_rate = get_spam_rate(y_train)
val_spam_rate = get_spam_rate(y_val)
test_spam_rate = get_spam_rate(y_test)

print(f'Train spam rate: {train_spam_rate:.3f} ({train_spam_rate*100:.1f}%)')
print(f'Val spam rate: {val_spam_rate:.3f} ({val_spam_rate*100:.1f}%)')
print(f'Test spam rate: {test_spam_rate:.3f} ({test_spam_rate*100:.1f}%)')

max_diff = max(abs(train_spam_rate - val_spam_rate), 
               abs(train_spam_rate - test_spam_rate),
               abs(val_spam_rate - test_spam_rate))

if max_diff <= 0.05:
    print(f'✅ Class distributions consistent (max diff: {max_diff:.3f})')
else:
    print(f'⚠️  Class distribution variance: {max_diff:.3f}')

# Step 7: Implement proper NLP - TF-IDF on actual text
print('\nSTEP 7: Legitimate NLP Processing (TF-IDF)')

# Basic text preprocessing
def preprocess_text(text):
    # Basic cleaning while preserving spam indicators
    text = re.sub(r'http[s]?://\S+', ' URL ', text)  # Replace URLs
    text = re.sub(r'\b\d{10,}\b', ' PHONE ', text)   # Replace phone numbers
    text = re.sub(r'[£$]\d+', ' MONEY ', text)       # Replace money amounts
    text = re.sub(r'\s+', ' ', text)                 # Normalize whitespace
    return text.strip().lower()

# Preprocess all text
X_train_processed = [preprocess_text(text) for text in X_train]
X_val_processed = [preprocess_text(text) for text in X_val]
X_test_processed = [preprocess_text(text) for text in X_test]

# Create TF-IDF vectorizer (proper NLP, not pre-engineered features)
tfidf = TfidfVectorizer(
    max_features=5000,          # Reasonable for 5K samples
    ngram_range=(1, 2),         # Unigrams and bigrams
    min_df=2,                   # Ignore very rare terms
    max_df=0.95,                # Ignore very common terms
    stop_words='english',       # Remove common stop words
    sublinear_tf=True           # Helps with high-frequency terms
)

# Fit TF-IDF on training data only (no data leakage)
print('✓ Fitting TF-IDF on training data...')
X_train_tfidf = tfidf.fit_transform(X_train_processed)

# Transform validation and test (no fitting on them!)
print('✓ Transforming validation and test data...')
X_val_tfidf = tfidf.transform(X_val_processed)
X_test_tfidf = tfidf.transform(X_test_processed)

print(f'✓ Feature matrix shape: {X_train_tfidf.shape}')
print(f'✓ Vocabulary size: {len(tfidf.vocabulary_)}')
print(f'✓ Sparsity: {1 - X_train_tfidf.nnz / np.prod(X_train_tfidf.shape):.3f}')

# Step 8: Encode labels
print('\nSTEP 8: Label Encoding')
label_encoder = LabelEncoder()
y_train_encoded = label_encoder.fit_transform(y_train)
y_val_encoded = label_encoder.transform(y_val)
y_test_encoded = label_encoder.transform(y_test)

print(f'✓ Label mapping: {dict(zip(label_encoder.classes_, label_encoder.transform(label_encoder.classes_)))}')
print(f'✓ Labels encoded: ham=0, spam=1')

# Step 9: Save clean data
print('\nSTEP 9: Saving Clean Data')
os.makedirs('data/clean', exist_ok=True)
os.makedirs('data/clean/features', exist_ok=True)

# Save feature matrices
save_npz('data/clean/features/train_features.npz', X_train_tfidf)
save_npz('data/clean/features/val_features.npz', X_val_tfidf)
save_npz('data/clean/features/test_features.npz', X_test_tfidf)

# Save processed text and labels
train_df = pd.DataFrame({
    'message': X_train,
    'message_processed': X_train_processed,
    'label': y_train,
    'label_encoded': y_train_encoded
})

val_df = pd.DataFrame({
    'message': X_val,
    'message_processed': X_val_processed,
    'label': y_val,
    'label_encoded': y_val_encoded
})

test_df = pd.DataFrame({
    'message': X_test,
    'message_processed': X_test_processed,
    'label': y_test,
    'label_encoded': y_test_encoded
})

train_df.to_csv('data/clean/train_clean.csv', index=False)
val_df.to_csv('data/clean/val_clean.csv', index=False)
test_df.to_csv('data/clean/test_clean.csv', index=False)

print('✅ Clean data saved successfully!')

# Step 10: Final verification
print('\nSTEP 10: Final Clean Pipeline Verification')
print('=' * 50)
print('DATA LEAKAGE CHECK: ✅ ZERO OVERLAP CONFIRMED')
print('NLP PROCESSING: ✅ LEGITIMATE TF-IDF ON ACTUAL TEXT')
print('FEATURE QUALITY: ✅ PROPER TEXT VECTORIZATION')
print('SPLITS: ✅ STRATIFIED 60/20/20 DISTRIBUTION')
print('=' * 50)

print(f'\n🎯 DAY 1 SUCCESS: Clean pipeline ready for legitimate research!')
print(f'Next: Day 2 baseline models (expected F1: 75-80%)')
print(f'Timeline: On track for honest results by June 20, 2025')

# Save pipeline documentation
with open('data/clean/pipeline_documentation.txt', 'w') as f:
    f.write("CLEAN DATA PIPELINE - RESEARCH INTEGRITY RECOVERY\n")
    f.write("=" * 50 + "\n")
    f.write(f"Date: 16/06/2025\n")
    f.write(f"Total samples: {len(df_original)}\n")
    f.write(f"Train: {len(X_train)} samples\n")
    f.write(f"Val: {len(X_val)} samples\n")
    f.write(f"Test: {len(X_test)} samples\n")
    f.write(f"Features: {X_train_tfidf.shape[1]} TF-IDF features\n")
    f.write(f"Data leakage: ZERO (verified)\n")
    f.write(f"NLP method: TF-IDF on original message text\n")
    f.write(f"Expected performance: 75-80% F1-Score (realistic)\n")

print('\n✅ DAY 1 COMPLETE: Research integrity restored!') 