"""
Unified data loading utilities for fair comparison between experiments
"""
import pandas as pd
import numpy as np
import glob
from sklearn.preprocessing import LabelEncoder

def load_parquet_data_respecting_splits(base_path, dataset_type='5channel'):
    """
    Load parquet files keeping original train/val/test splits separate
    
    Args:
        base_path: Path to dataset directory
        dataset_type: Either '5channel' or 'multichannel_hilbert'
    
    Returns:
        X_train, y_train, X_val, y_val, X_test, y_test, label_encoder
    """
    data_splits = {'train': [], 'val': [], 'test': []}
    label_splits = {'train': [], 'val': [], 'test': []}
    
    # Get all class directories
    class_dirs = [d for d in glob.glob(f"{base_path}*/") if not any(s in d for s in ['train', 'val', 'test'])]
    class_names = sorted([d.split('/')[-2] for d in class_dirs])  # Sort for consistency
    print(f"Found {len(class_names)} classes: {class_names}")
    
    for class_dir in class_dirs:
        class_name = class_dir.split('/')[-2]
        print(f"Loading {class_name}...")
        
        for split in ['train', 'val', 'test']:
            split_path = f"{class_dir}{split}/"
            parquet_files = sorted(glob.glob(f"{split_path}*.parquet"))  # Sort for consistency
            
            split_count = 0
            for file_path in parquet_files:
                try:
                    df = pd.read_parquet(file_path)
                    
                    if 'image_data' in df.columns:
                        for idx, row in df.iterrows():
                            image_data = np.array(row['image_data'], dtype=np.float32)
                            data_splits[split].append(image_data)
                            label_splits[split].append(class_name)
                            split_count += 1
                        
                except Exception as e:
                    print(f"   Error loading {file_path}: {e}")
            
            print(f"   Loaded {split_count} {split} samples")
    
    # Convert to numpy arrays for each split
    X_train = np.array(data_splits['train'], dtype=np.float32)
    y_train = np.array(label_splits['train'])
    X_val = np.array(data_splits['val'], dtype=np.float32)
    y_val = np.array(label_splits['val'])
    X_test = np.array(data_splits['test'], dtype=np.float32)
    y_test = np.array(label_splits['test'])
    
    # Create label encoder using all labels
    label_encoder = LabelEncoder()
    all_labels = np.concatenate([y_train, y_val, y_test])
    label_encoder.fit(all_labels)
    
    # Encode labels
    y_train_encoded = label_encoder.transform(y_train)
    y_val_encoded = label_encoder.transform(y_val)
    y_test_encoded = label_encoder.transform(y_test)
    
    print(f"\n✓ Loaded splits:")
    print(f"   Train: {len(X_train)} samples")
    print(f"   Val: {len(X_val)} samples")
    print(f"   Test: {len(X_test)} samples")
    print(f"   Total: {len(X_train) + len(X_val) + len(X_test)} samples")
    
    # Data quality checks
    print(f"\n✓ Data quality:")
    print(f"   Features per sample: {X_train.shape[1]}")
    print(f"   Value range: [{X_train.min():.3f}, {X_train.max():.3f}]")
    print(f"   Classes: {len(label_encoder.classes_)}")
    
    return X_train, y_train_encoded, X_val, y_val_encoded, X_test, y_test_encoded, label_encoder


def load_mixed_data_for_compatibility(base_path):
    """
    Load all data mixed together (for backward compatibility)
    This approach has data leakage but matches original notebook behavior
    """
    all_image_data = []
    all_labels = []
    splits = ['train', 'val', 'test']
    
    # Get all class directories
    class_dirs = [d for d in glob.glob(f"{base_path}*/") if not any(s in d for s in splits)]
    class_names = sorted([d.split('/')[-2] for d in class_dirs])
    print(f"Found classes: {class_names}")
    
    for class_dir in class_dirs:
        class_name = class_dir.split('/')[-2]
        print(f"Loading {class_name}...")
        
        for split in splits:
            split_path = f"{class_dir}{split}/"
            parquet_files = sorted(glob.glob(f"{split_path}*.parquet"))
            
            for file_path in parquet_files:
                try:
                    df = pd.read_parquet(file_path)
                    
                    if 'image_data' in df.columns:
                        for idx, row in df.iterrows():
                            image_data = np.array(row['image_data'], dtype=np.float32)
                            all_image_data.append(image_data)
                            all_labels.append(class_name)
                        
                        print(f"   Loaded {len(df)} samples from {file_path.split('/')[-1]}")
                        
                except Exception as e:
                    print(f"   Error loading {file_path}: {e}")
    
    if not all_image_data:
        raise ValueError("No image data was loaded successfully!")
    
    # Convert to numpy arrays
    X = np.array(all_image_data, dtype=np.float32)
    y = np.array(all_labels)
    
    print(f"\n✓ Total samples loaded: {len(X)}")
    print(f"✓ Image data shape: {X.shape}")
    print(f"✓ Unique labels: {np.unique(y)}")
    
    return X, y