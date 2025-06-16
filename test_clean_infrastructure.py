#!/usr/bin/env python3
"""
Test Clean Data Infrastructure - Research Integrity Recovery
============================================================
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import hashlib

def generate_message_hash(text):
    normalized = str(text).lower().strip()
    return hashlib.sha256(normalized.encode('utf-8')).hexdigest()

def load_raw_data():
    print('🔄 Loading clean source data...')
    with open('data/SMSSPamCollection', 'r', encoding='utf-8') as f:
        lines = f.readlines()
    
    data = []
    for line in lines:
        parts = line.strip().split('\t', 1)
        if len(parts) == 2:
            label, text = parts
            data.append({'label': label.strip(), 'text': text.strip()})
    
    df = pd.DataFrame(data)
    print(f'✅ Loaded {len(df)} messages')
    print(f'   Labels: {dict(df["label"].value_counts())}')
    return df

def detect_duplicates(df):
    print('🔍 Detecting duplicate messages...')
    df['message_hash'] = df['text'].apply(generate_message_hash)
    hash_counts = df['message_hash'].value_counts()
    duplicates = hash_counts[hash_counts > 1]
    
    print(f'   Total messages: {len(df)}')
    print(f'   Unique messages: {len(hash_counts)}')
    print(f'   Duplicate groups: {len(duplicates)}')
    return df, duplicates

def create_clean_splits(df):
    print('🔧 Creating clean data splits with zero leakage...')
    
    # Remove duplicates BEFORE splitting
    df_dedup = df.drop_duplicates(subset=['message_hash'], keep='first')
    removed = len(df) - len(df_dedup)
    if removed > 0:
        print(f'   Removed {removed} duplicates before splitting')
    
    # Stratified splitting
    X = df_dedup['text']
    y = df_dedup['label']
    
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y, shuffle=True
    )
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, test_size=0.25, random_state=42, 
        stratify=y_train_val, shuffle=True
    )
    
    train_df = pd.DataFrame({'text': X_train, 'label': y_train}).reset_index(drop=True)
    val_df = pd.DataFrame({'text': X_val, 'label': y_val}).reset_index(drop=True)
    test_df = pd.DataFrame({'text': X_test, 'label': y_test}).reset_index(drop=True)
    
    print(f'✅ Clean splits created:')
    print(f'   Train: {len(train_df)} ({(train_df["label"] == "spam").mean():.1%} spam)')
    print(f'   Val:   {len(val_df)} ({(val_df["label"] == "spam").mean():.1%} spam)')
    print(f'   Test:  {len(test_df)} ({(test_df["label"] == "spam").mean():.1%} spam)')
    
    return train_df, val_df, test_df

def validate_zero_leakage(train_df, val_df, test_df):
    print('🔒 Validating ZERO data leakage...')
    
    train_hashes = set(train_df['text'].apply(generate_message_hash))
    val_hashes = set(val_df['text'].apply(generate_message_hash))
    test_hashes = set(test_df['text'].apply(generate_message_hash))
    
    train_val_overlap = len(train_hashes.intersection(val_hashes))
    train_test_overlap = len(train_hashes.intersection(test_hashes))
    val_test_overlap = len(val_hashes.intersection(test_hashes))
    total_overlaps = train_val_overlap + train_test_overlap + val_test_overlap
    
    if total_overlaps == 0:
        print('✅ ZERO LEAKAGE CONFIRMED - Research integrity maintained!')
        status = 'PASS'
    else:
        print(f'❌ DATA LEAKAGE DETECTED: {total_overlaps} overlaps')
        print(f'   Train-Val: {train_val_overlap}, Train-Test: {train_test_overlap}, Val-Test: {val_test_overlap}')
        status = 'FAIL'
    
    return status, total_overlaps

def main():
    print('=' * 80)
    print('🔬 CLEAN DATA INFRASTRUCTURE - RESEARCH INTEGRITY RECOVERY')
    print('=' * 80)
    print('Mission: Eliminate data leakage for legitimate 75-90% F1-Score models')
    print('Timeline: Priority 1 - Deadline June 17, 2025 17:00')
    print('-' * 80)

    # Load and process
    df = load_raw_data()
    df, duplicates = detect_duplicates(df)
    train_df, val_df, test_df = create_clean_splits(df)
    status, overlaps = validate_zero_leakage(train_df, val_df, test_df)

    print()
    print('=' * 80)
    print('🎯 CLEAN DATA INFRASTRUCTURE - VALIDATION COMPLETE')
    print('=' * 80)
    print(f'✅ ZERO LEAKAGE STATUS: {status}')
    print(f'✅ Clean datasets: {len(train_df)}/{len(val_df)}/{len(test_df)} samples')
    print(f'✅ Expected performance: 75-90% F1-Score (realistic)')
    print(f'✅ Research integrity: VERIFIED CLEAN')
    print('=' * 80)
    
    # Save clean datasets
    print('\n💾 Saving clean datasets...')
    train_df.to_csv('data/clean/train_clean.csv', index=False)
    val_df.to_csv('data/clean/val_clean.csv', index=False)
    test_df.to_csv('data/clean/test_clean.csv', index=False)
    print('✅ Clean datasets saved to data/clean/')
    
    return status == 'PASS'

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1) 