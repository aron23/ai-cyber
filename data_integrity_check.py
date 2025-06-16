#!/usr/bin/env python3
"""
Data Integrity Check - Investigate potential shortcuts in training
"""
import pandas as pd
import numpy as np
from scipy import sparse
import os

print('=== DATA INTEGRITY & COMPLEXITY CHECK ===')

# Load and examine the actual dataset
try:
    train_df = pd.read_csv('data/processed/train.csv')
    val_df = pd.read_csv('data/processed/validation.csv') 
    test_df = pd.read_csv('data/processed/test.csv')
    
    print(f'✓ Data files loaded successfully')
    print(f'Train samples: {len(train_df)}')
    print(f'Val samples: {len(val_df)}')
    print(f'Test samples: {len(test_df)}')
    
    # Display first few rows to understand structure
    print(f'\nTrain data structure:')
    print(train_df.head())
    print(f'\nColumns: {list(train_df.columns)}')
    
except Exception as e:
    print(f'❌ Error loading data: {e}')
    exit(1)

# Check for data leakage - any overlap between sets?
print(f'\n=== DATA LEAKAGE CHECK ===')
try:
    # Get text column (might be 'text', 'message', or first column)
    text_col = 'text' if 'text' in train_df.columns else train_df.columns[0]
    label_col = 'label' if 'label' in train_df.columns else train_df.columns[1]
    
    train_texts = set(train_df[text_col].astype(str))
    val_texts = set(val_df[text_col].astype(str))
    test_texts = set(test_df[text_col].astype(str))
    
    overlap_train_val = len(train_texts.intersection(val_texts))
    overlap_train_test = len(train_texts.intersection(test_texts))
    overlap_val_test = len(val_texts.intersection(test_texts))
    
    print(f'Train-Val overlap: {overlap_train_val} samples')
    print(f'Train-Test overlap: {overlap_train_test} samples')
    print(f'Val-Test overlap: {overlap_val_test} samples')
    
    if overlap_train_val > 0 or overlap_train_test > 0:
        print('🚨 WARNING: Data leakage detected!')
    else:
        print('✓ No data leakage detected')
        
except Exception as e:
    print(f'❌ Error checking leakage: {e}')

# Check class distribution consistency
print(f'\n=== CLASS DISTRIBUTION CHECK ===')
try:
    train_spam_rate = (train_df[label_col] == 'spam').mean()
    val_spam_rate = (val_df[label_col] == 'spam').mean()
    test_spam_rate = (test_df[label_col] == 'spam').mean()
    
    print(f'Train spam rate: {train_spam_rate:.3f} ({train_spam_rate*100:.1f}%)')
    print(f'Val spam rate: {val_spam_rate:.3f} ({val_spam_rate*100:.1f}%)')
    print(f'Test spam rate: {test_spam_rate:.3f} ({test_spam_rate*100:.1f}%)')
    
    # Check if distributions are similar (within 5%)
    max_diff = max(abs(train_spam_rate - val_spam_rate), 
                   abs(train_spam_rate - test_spam_rate),
                   abs(val_spam_rate - test_spam_rate))
    
    if max_diff > 0.05:
        print(f'⚠️  Warning: Class distribution varies by {max_diff:.3f}')
    else:
        print(f'✓ Class distributions are consistent (max diff: {max_diff:.3f})')
        
except Exception as e:
    print(f'❌ Error checking distributions: {e}')

# Check text complexity
print(f'\n=== TEXT COMPLEXITY ANALYSIS ===')
try:
    # Sample some texts to check complexity
    sample_texts = train_df[text_col].head(10)
    
    print('Sample messages:')
    for i, text in enumerate(sample_texts):
        print(f'{i+1}. ({len(str(text))} chars): {str(text)[:100]}...')
    
    # Basic complexity metrics
    avg_length = train_df[text_col].astype(str).str.len().mean()
    avg_words = train_df[text_col].astype(str).str.split().str.len().mean()
    
    print(f'\nText Statistics:')
    print(f'Average message length: {avg_length:.1f} characters')
    print(f'Average words per message: {avg_words:.1f} words')
    
    # Check if messages are too simple/uniform
    unique_lengths = train_df[text_col].astype(str).str.len().nunique()
    total_messages = len(train_df)
    length_diversity = unique_lengths / total_messages
    
    print(f'Length diversity: {length_diversity:.3f} (higher = more diverse)')
    
    if avg_length < 20:
        print('⚠️  Warning: Messages are very short - might be oversimplified')
    if length_diversity < 0.1:
        print('⚠️  Warning: Low text diversity - might be artificial dataset')
        
except Exception as e:
    print(f'❌ Error analyzing text complexity: {e}')

# Check feature complexity
print(f'\n=== FEATURE COMPLEXITY CHECK ===')
try:
    # Load feature matrix to check sparsity and complexity
    train_features = sparse.load_npz('data/features/train_features.npz')
    
    print(f'Feature matrix shape: {train_features.shape}')
    print(f'Feature matrix sparsity: {1 - train_features.nnz / np.prod(train_features.shape):.3f}')
    print(f'Non-zero elements: {train_features.nnz:,}')
    
    # Check if features are too sparse (might indicate oversimplified)
    sparsity = 1 - train_features.nnz / np.prod(train_features.shape)
    if sparsity > 0.99:
        print('⚠️  Warning: Features are extremely sparse - might be oversimplified')
    elif sparsity > 0.95:
        print('✓ Normal sparsity for text features')
    else:
        print('ℹ️  Features are relatively dense')
    
except Exception as e:
    print(f'❌ Error checking features: {e}')

# Check for potential dataset issues
print(f'\n=== DATASET REALISM CHECK ===')
try:
    # Look for signs of artificial/toy dataset
    total_samples = len(train_df) + len(val_df) + len(test_df)
    print(f'Total dataset size: {total_samples:,} samples')
    
    if total_samples < 1000:
        print('⚠️  Warning: Very small dataset - results may not generalize')
    elif total_samples < 5000:
        print('ℹ️  Small dataset - explains fast training but check generalization')
    else:
        print('✓ Reasonable dataset size')
    
    # Check if this is a standard benchmark dataset
    if total_samples in [5572, 5574]:  # Common SMS spam dataset sizes
        print('ℹ️  This appears to be the SMS Spam Collection dataset')
        print('   - Well-known benchmark dataset')
        print('   - Relatively simple spam detection problem')
        print('   - Fast training is expected for this dataset')
    
except Exception as e:
    print(f'❌ Error in realism check: {e}')

print(f'\n=== CONCLUSION ===')
print('Check completed. Review warnings above for potential shortcuts.') 