"""
PayloadByteDataLoader: Efficient data loading for Payload-Byte datasets.
Supports chunked reading, parallel processing, and train/val/test splitting.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union, Iterator
import multiprocessing as mp
from concurrent.futures import ThreadPoolExecutor, as_completed
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
from src.utils.logger import setup_logger

logger = setup_logger('data_loader')


class PayloadByteDataLoader:
    """
    Efficient data loader for Payload-Byte network intrusion detection datasets.
    
    Features:
    - Memory-efficient chunked reading
    - Multi-threaded parallel loading
    - Automatic train/val/test splitting
    - PyTorch Dataset integration
    """
    
    def __init__(self, 
                 data_path: Union[str, Path],
                 batch_size: int = 32,
                 chunk_size: int = 10000,
                 n_workers: int = 4,
                 cache_dir: Optional[Path] = None):
        """
        Initialize the data loader.
        
        Args:
            data_path: Path to data directory or specific CSV file
            batch_size: Batch size for data loading
            chunk_size: Number of rows to read at once
            n_workers: Number of parallel workers
            cache_dir: Directory for caching processed data
        """
        self.data_path = Path(data_path)
        self.batch_size = batch_size
        self.chunk_size = chunk_size
        self.n_workers = n_workers
        self.cache_dir = Path(cache_dir) if cache_dir else None
        
        # Data attributes
        self.file_list = []
        self.header_features = ['src_ip', 'dst_ip', 'src_port', 'dst_port']
        self.byte_features = [f'byte_{i}' for i in range(1500)]
        self.all_features = self.header_features + self.byte_features
        
        # Data splits
        self.train_data = None
        self.val_data = None
        self.test_data = None
        
        self._discover_files()
    
    def _discover_files(self):
        """Discover all CSV files in the data path."""
        if self.data_path.is_file():
            self.file_list = [self.data_path]
        else:
            self.file_list = list(self.data_path.glob('*.csv'))
        
        if not self.file_list:
            raise ValueError(f"No CSV files found in {self.data_path}")
        
        logger.info(f"Found {len(self.file_list)} data files")
    
    def load_raw_packets(self, 
                        files: Optional[List[Path]] = None,
                        nrows: Optional[int] = None) -> pd.DataFrame:
        """
        Load raw packet data from CSV files.
        
        Args:
            files: List of files to load (None for all)
            nrows: Maximum number of rows to load
            
        Returns:
            DataFrame containing packet data
        """
        if files is None:
            files = self.file_list
        
        logger.info(f"Loading data from {len(files)} files...")
        
        # Load data in parallel
        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = []
            for file_path in files:
                future = executor.submit(self._load_single_file, file_path, nrows)
                futures.append(future)
            
            # Collect results
            dataframes = []
            for future in as_completed(futures):
                try:
                    df = future.result()
                    if df is not None:
                        dataframes.append(df)
                except Exception as e:
                    logger.error(f"Error loading file: {e}")
        
        if not dataframes:
            raise ValueError("No data could be loaded")
        
        # Concatenate all dataframes
        combined_df = pd.concat(dataframes, ignore_index=True)
        logger.info(f"Loaded {len(combined_df):,} total packets")
        
        return combined_df
    
    def _load_single_file(self, 
                         file_path: Path, 
                         nrows: Optional[int] = None) -> Optional[pd.DataFrame]:
        """Load a single CSV file."""
        try:
            if nrows:
                df = pd.read_csv(file_path, nrows=nrows)
            else:
                # Use chunked reading for memory efficiency
                chunks = []
                for chunk in pd.read_csv(file_path, chunksize=self.chunk_size):
                    chunks.append(chunk)
                df = pd.concat(chunks, ignore_index=True)
            
            logger.debug(f"Loaded {len(df)} rows from {file_path.name}")
            return df
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return None
    
    def create_train_val_test_splits(self,
                                   data: Optional[pd.DataFrame] = None,
                                   train_ratio: float = 0.7,
                                   val_ratio: float = 0.15,
                                   test_ratio: float = 0.15,
                                   stratify: bool = True,
                                   random_state: int = 42) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Create train/validation/test splits.
        
        Args:
            data: DataFrame to split (None to load all data)
            train_ratio: Proportion for training set
            val_ratio: Proportion for validation set
            test_ratio: Proportion for test set
            stratify: Whether to stratify by label
            random_state: Random seed for reproducibility
            
        Returns:
            Tuple of (train_df, val_df, test_df)
        """
        if data is None:
            data = self.load_raw_packets()
        
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, \
            "Ratios must sum to 1.0"
        
        logger.info(f"Creating splits: train={train_ratio}, val={val_ratio}, test={test_ratio}")
        
        # Get features and labels
        X = data[self.all_features]
        y = data['label'] if 'label' in data.columns else None
        
        # First split: train+val vs test
        X_temp, X_test, y_temp, y_test = train_test_split(
            X, y,
            test_size=test_ratio,
            stratify=y if stratify and y is not None else None,
            random_state=random_state
        )
        
        # Second split: train vs val
        val_size = val_ratio / (train_ratio + val_ratio)
        X_train, X_val, y_train, y_val = train_test_split(
            X_temp, y_temp,
            test_size=val_size,
            stratify=y_temp if stratify and y_temp is not None else None,
            random_state=random_state
        )
        
        # Reconstruct DataFrames
        train_df = X_train.copy()
        val_df = X_val.copy()
        test_df = X_test.copy()
        
        if y is not None:
            train_df['label'] = y_train
            val_df['label'] = y_val
            test_df['label'] = y_test
        
        self.train_data = train_df
        self.val_data = val_df
        self.test_data = test_df
        
        logger.info(f"Split sizes - Train: {len(train_df):,}, Val: {len(val_df):,}, Test: {len(test_df):,}")
        
        return train_df, val_df, test_df
    
    def get_batch_generator(self, 
                          dataset: str = 'train',
                          shuffle: bool = True,
                          drop_last: bool = False) -> Iterator[pd.DataFrame]:
        """
        Get a batch generator for the specified dataset.
        
        Args:
            dataset: One of 'train', 'val', or 'test'
            shuffle: Whether to shuffle data
            drop_last: Whether to drop incomplete last batch
            
        Yields:
            DataFrame batches
        """
        # Get the appropriate dataset
        if dataset == 'train':
            data = self.train_data
        elif dataset == 'val':
            data = self.val_data
        elif dataset == 'test':
            data = self.test_data
        else:
            raise ValueError(f"Unknown dataset: {dataset}")
        
        if data is None:
            raise ValueError(f"No {dataset} data available. Run create_train_val_test_splits() first.")
        
        # Create index array
        indices = np.arange(len(data))
        
        if shuffle:
            np.random.shuffle(indices)
        
        # Generate batches
        n_batches = len(data) // self.batch_size
        if not drop_last and len(data) % self.batch_size != 0:
            n_batches += 1
        
        for i in range(n_batches):
            start_idx = i * self.batch_size
            end_idx = min((i + 1) * self.batch_size, len(data))
            
            batch_indices = indices[start_idx:end_idx]
            batch_data = data.iloc[batch_indices]
            
            yield batch_data
    
    def get_feature_names(self) -> Dict[str, List[str]]:
        """Get organized feature names."""
        return {
            'header': self.header_features,
            'bytes': self.byte_features,
            'all': self.all_features
        }
    
    def get_data_stats(self) -> Dict:
        """Get statistics about loaded data."""
        stats = {
            'n_files': len(self.file_list),
            'n_features': len(self.all_features),
            'n_header_features': len(self.header_features),
            'n_byte_features': len(self.byte_features)
        }
        
        if self.train_data is not None:
            stats['n_train'] = len(self.train_data)
            stats['n_val'] = len(self.val_data) if self.val_data is not None else 0
            stats['n_test'] = len(self.test_data) if self.test_data is not None else 0
            
            # Label distribution
            if 'label' in self.train_data.columns:
                stats['train_labels'] = self.train_data['label'].value_counts().to_dict()
        
        return stats


class PayloadByteDataset(Dataset):
    """PyTorch Dataset wrapper for Payload-Byte data."""
    
    def __init__(self, 
                 dataframe: pd.DataFrame,
                 transform=None,
                 target_transform=None):
        """
        Initialize PyTorch dataset.
        
        Args:
            dataframe: Pandas DataFrame with packet data
            transform: Optional transform for features
            target_transform: Optional transform for labels
        """
        self.data = dataframe
        self.transform = transform
        self.target_transform = target_transform
        
        # Separate features and labels
        feature_cols = [col for col in dataframe.columns if col != 'label']
        self.features = dataframe[feature_cols].values.astype(np.float32)
        
        if 'label' in dataframe.columns:
            self.labels = dataframe['label'].values.astype(np.int64)
        else:
            self.labels = None
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        # Get features
        x = self.features[idx]
        
        if self.transform:
            x = self.transform(x)
        
        # Get label if available
        if self.labels is not None:
            y = self.labels[idx]
            if self.target_transform:
                y = self.target_transform(y)
            return x, y
        else:
            return x


def create_pytorch_dataloaders(data_loader: PayloadByteDataLoader,
                             batch_size: int = 32,
                             num_workers: int = 4,
                             pin_memory: bool = True) -> Dict[str, DataLoader]:
    """
    Create PyTorch DataLoaders from PayloadByteDataLoader.
    
    Args:
        data_loader: Initialized PayloadByteDataLoader with splits
        batch_size: Batch size for DataLoaders
        num_workers: Number of worker processes
        pin_memory: Whether to pin memory for GPU transfer
        
    Returns:
        Dictionary with 'train', 'val', and 'test' DataLoaders
    """
    dataloaders = {}
    
    # Create datasets
    if data_loader.train_data is not None:
        train_dataset = PayloadByteDataset(data_loader.train_data)
        dataloaders['train'] = DataLoader(
            train_dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    if data_loader.val_data is not None:
        val_dataset = PayloadByteDataset(data_loader.val_data)
        dataloaders['val'] = DataLoader(
            val_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    if data_loader.test_data is not None:
        test_dataset = PayloadByteDataset(data_loader.test_data)
        dataloaders['test'] = DataLoader(
            test_dataset,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=pin_memory
        )
    
    return dataloaders


if __name__ == "__main__":
    # Example usage
    data_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'payload_byte'
    
    # Initialize loader
    loader = PayloadByteDataLoader(data_path, batch_size=64)
    
    # Load and split data
    train_df, val_df, test_df = loader.create_train_val_test_splits()
    
    # Print statistics
    stats = loader.get_data_stats()
    print("Data Loader Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Test batch generator
    print("\nTesting batch generator:")
    for i, batch in enumerate(loader.get_batch_generator('train')):
        print(f"  Batch {i}: shape={batch.shape}")
        if i >= 2:  # Just show first 3 batches
            break