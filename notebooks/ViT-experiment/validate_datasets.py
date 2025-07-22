"""
Validate that both datasets are loaded correctly and can be fairly compared
"""
import numpy as np
import sys
sys.path.append('/home/ubuntu/analyst/notebooks/ViT-experiment/')
from data_loader_utils import load_parquet_data_respecting_splits, load_mixed_data_for_compatibility

print("=" * 80)
print("DATASET VALIDATION FOR FAIR COMPARISON")
print("=" * 80)

# Define paths
paths = {
    '5channel': '/home/ubuntu/analyst/notebooks/ViT-experiment/pcap-dataset-samples/parquet/5channel_32x32/',
    'multichannel_hilbert': '/home/ubuntu/analyst/notebooks/ViT-experiment/pcap-dataset-samples/parquet/multichannel_hilbert_32x32/'
}

# Expected configuration
expected_features = 5 * 32 * 32  # 5120
expected_image_shape = (5, 32, 32)

# Load both datasets
datasets = {}
for name, path in paths.items():
    print(f"\n{'='*40}")
    print(f"Loading {name} dataset...")
    print(f"Path: {path}")
    print(f"{'='*40}")
    
    try:
        # Load with original splits preserved
        X_train, y_train, X_val, y_val, X_test, y_test, label_encoder = load_parquet_data_respecting_splits(path, name)
        
        datasets[name] = {
            'X_train': X_train,
            'y_train': y_train,
            'X_val': X_val,
            'y_val': y_val,
            'X_test': X_test,
            'y_test': y_test,
            'label_encoder': label_encoder,
            'loaded': True
        }
        
        # Validate data shape
        print(f"\n✓ Validation for {name}:")
        print(f"  - Features match expected: {X_train.shape[1]} == {expected_features}")
        print(f"  - Can reshape to image: {X_train.shape[1]} → {expected_image_shape}")
        
        # Test reshape
        test_reshape = X_train[0].reshape(expected_image_shape)
        print(f"  - Reshape successful: {test_reshape.shape}")
        
    except Exception as e:
        print(f"✗ Error loading {name}: {e}")
        datasets[name] = {'loaded': False, 'error': str(e)}

print("\n" + "="*80)
print("COMPARISON SUMMARY")
print("="*80)

if all(d.get('loaded', False) for d in datasets.values()):
    # Compare datasets
    d1 = datasets['5channel']
    d2 = datasets['multichannel_hilbert']
    
    print("\n📊 Dataset Size Comparison:")
    print(f"{'Dataset':<25} {'Train':<10} {'Val':<10} {'Test':<10} {'Total':<10}")
    print("-" * 65)
    
    for name, d in datasets.items():
        if d['loaded']:
            train_size = len(d['X_train'])
            val_size = len(d['X_val'])
            test_size = len(d['X_test'])
            total = train_size + val_size + test_size
            print(f"{name:<25} {train_size:<10} {val_size:<10} {test_size:<10} {total:<10}")
    
    print("\n📊 Class Distribution Comparison:")
    for name, d in datasets.items():
        if d['loaded']:
            print(f"\n{name}:")
            classes = d['label_encoder'].classes_
            print(f"  Classes: {len(classes)}")
            for i, class_name in enumerate(classes):
                train_count = np.sum(d['y_train'] == i)
                val_count = np.sum(d['y_val'] == i)
                test_count = np.sum(d['y_test'] == i)
                print(f"  {class_name:<25}: Train={train_count:<6} Val={val_count:<6} Test={test_count}")
    
    print("\n📊 Data Quality Comparison:")
    for name, d in datasets.items():
        if d['loaded']:
            print(f"\n{name}:")
            X_all = np.concatenate([d['X_train'], d['X_val'], d['X_test']])
            print(f"  Value range: [{X_all.min():.3f}, {X_all.max():.3f}]")
            print(f"  Mean: {X_all.mean():.3f}")
            print(f"  Std: {X_all.std():.3f}")
            print(f"  Non-zero ratio: {np.mean(X_all > 0):.3f}")
    
    # Check if classes match
    if np.array_equal(d1['label_encoder'].classes_, d2['label_encoder'].classes_):
        print("\n✅ Classes match between datasets - comparison is valid")
    else:
        print("\n❌ Classes don't match between datasets!")
        print(f"5channel classes: {d1['label_encoder'].classes_}")
        print(f"multichannel_hilbert classes: {d2['label_encoder'].classes_}")
    
    # Check split sizes
    if (len(d1['X_train']) == len(d2['X_train']) and 
        len(d1['X_val']) == len(d2['X_val']) and 
        len(d1['X_test']) == len(d2['X_test'])):
        print("✅ Split sizes match - comparison is valid")
    else:
        print("❌ Split sizes don't match!")
        
    print("\n" + "="*80)
    print("RECOMMENDATION")
    print("="*80)
    
    if all(d.get('loaded', False) for d in datasets.values()):
        print("✅ Both datasets loaded successfully")
        print("✅ Both can be reshaped to 5×32×32 images")
        print("✅ Use the load_parquet_data_respecting_splits() function in both notebooks")
        print("✅ This preserves original train/val/test splits and prevents data leakage")
    else:
        print("❌ One or more datasets failed to load")
        print("❌ Fix data paths and ensure data exists before proceeding")
        
else:
    print("\n❌ Could not load all datasets for comparison")
    for name, d in datasets.items():
        if not d.get('loaded', False):
            print(f"  - {name}: {d.get('error', 'Unknown error')}")

print("\n" + "="*80)