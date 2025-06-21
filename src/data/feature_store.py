"""
Feature Store Module

Provides a centralized storage and retrieval system for engineered features
with versioning, metadata tracking, and efficient access patterns.
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Union, Tuple, Any
import json
import hashlib
from datetime import datetime
import logging
from dataclasses import dataclass, asdict
import pickle


logger = logging.getLogger(__name__)


@dataclass
class FeatureMetadata:
    """Metadata for stored features"""
    name: str
    version: str
    shape: Tuple[int, ...]
    dtype: str
    created_at: str
    description: str
    preprocessing_params: Dict[str, Any]
    hash: str
    

class FeatureStore:
    """
    Centralized feature storage system using HDF5 for efficient
    storage and retrieval of large feature matrices.
    """
    
    def __init__(self, store_path: Union[str, Path], chunk_size: int = 1000):
        """
        Initialize the feature store.
        
        Args:
            store_path: Path to the HDF5 store file
            chunk_size: Chunk size for HDF5 datasets
        """
        self.store_path = Path(store_path)
        self.store_path.parent.mkdir(parents=True, exist_ok=True)
        self.chunk_size = chunk_size
        self._metadata_cache = {}
        
    def store_features(self,
                      features: Union[np.ndarray, pd.DataFrame],
                      name: str,
                      version: Optional[str] = None,
                      description: str = "",
                      preprocessing_params: Optional[Dict[str, Any]] = None,
                      compression: str = 'gzip',
                      compression_level: int = 4) -> str:
        """
        Store features with metadata and versioning.
        
        Args:
            features: Feature matrix to store
            name: Name of the feature set
            version: Version string (auto-generated if None)
            description: Description of the features
            preprocessing_params: Parameters used for preprocessing
            compression: HDF5 compression algorithm
            compression_level: Compression level (1-9)
            
        Returns:
            Feature set identifier (name:version)
        """
        # Convert DataFrame to numpy if needed
        if isinstance(features, pd.DataFrame):
            feature_names = features.columns.tolist()
            features = features.values
        else:
            feature_names = None
            
        # Auto-generate version if not provided
        if version is None:
            version = self._generate_version(features)
            
        # Create metadata
        metadata = FeatureMetadata(
            name=name,
            version=version,
            shape=features.shape,
            dtype=str(features.dtype),
            created_at=datetime.utcnow().isoformat(),
            description=description,
            preprocessing_params=preprocessing_params or {},
            hash=self._compute_hash(features)
        )
        
        # Store in HDF5
        feature_key = f"{name}/{version}"
        
        with h5py.File(self.store_path, 'a') as store:
            # Create group if it doesn't exist
            if name not in store:
                store.create_group(name)
                
            # Store features
            if version in store[name]:
                logger.warning(f"Overwriting existing features: {feature_key}")
                del store[name][version]
                
            dataset = store[name].create_dataset(
                version,
                data=features,
                chunks=(min(self.chunk_size, features.shape[0]),) + features.shape[1:],
                compression=compression,
                compression_opts=compression_level
            )
            
            # Store metadata as attributes
            dataset.attrs['metadata'] = json.dumps(asdict(metadata))
            
            # Store feature names if available
            if feature_names:
                dataset.attrs['feature_names'] = json.dumps(feature_names)
                
        # Update cache
        self._metadata_cache[feature_key] = metadata
        
        logger.info(f"Stored features: {feature_key}, shape: {features.shape}")
        return feature_key
    
    def load_features(self,
                     name: str,
                     version: Optional[str] = None,
                     feature_names: Optional[List[str]] = None,
                     indices: Optional[Union[slice, List[int]]] = None) -> Tuple[np.ndarray, FeatureMetadata]:
        """
        Load features from the store.
        
        Args:
            name: Name of the feature set
            version: Version to load (latest if None)
            feature_names: Specific features to load (all if None)
            indices: Row indices to load (all if None)
            
        Returns:
            Tuple of (features, metadata)
        """
        with h5py.File(self.store_path, 'r') as store:
            if name not in store:
                raise KeyError(f"Feature set '{name}' not found")
                
            # Get version
            if version is None:
                # Get latest version
                versions = list(store[name].keys())
                if not versions:
                    raise KeyError(f"No versions found for '{name}'")
                version = sorted(versions)[-1]
                
            if version not in store[name]:
                raise KeyError(f"Version '{version}' not found for '{name}'")
                
            # Load dataset
            dataset = store[name][version]
            
            # Load metadata
            metadata_json = dataset.attrs['metadata']
            metadata = FeatureMetadata(**json.loads(metadata_json))
            
            # Load feature names if available
            stored_feature_names = None
            if 'feature_names' in dataset.attrs:
                stored_feature_names = json.loads(dataset.attrs['feature_names'])
                
            # Load data
            if indices is None:
                data = dataset[:]
            else:
                data = dataset[indices]
                
            # Filter by feature names if requested
            if feature_names and stored_feature_names:
                feature_indices = [stored_feature_names.index(name) 
                                 for name in feature_names 
                                 if name in stored_feature_names]
                data = data[:, feature_indices]
                
        logger.info(f"Loaded features: {name}:{version}, shape: {data.shape}")
        return data, metadata
    
    def get_feature_statistics(self,
                             name: Optional[str] = None,
                             version: Optional[str] = None) -> Dict[str, Any]:
        """
        Get statistics about stored features.
        
        Args:
            name: Feature set name (all if None)
            version: Specific version (all if None)
            
        Returns:
            Dictionary with feature statistics
        """
        stats = {
            'total_size_mb': 0,
            'feature_sets': {},
            'total_features': 0,
            'total_samples': 0
        }
        
        with h5py.File(self.store_path, 'r') as store:
            # Filter by name if provided
            names = [name] if name else list(store.keys())
            
            for feature_name in names:
                if feature_name not in store:
                    continue
                    
                stats['feature_sets'][feature_name] = {
                    'versions': {},
                    'total_size_mb': 0
                }
                
                # Filter by version if provided
                versions = [version] if version else list(store[feature_name].keys())
                
                for ver in versions:
                    if ver not in store[feature_name]:
                        continue
                        
                    dataset = store[feature_name][ver]
                    metadata = FeatureMetadata(**json.loads(dataset.attrs['metadata']))
                    
                    size_mb = dataset.size * dataset.dtype.itemsize / (1024 * 1024)
                    
                    stats['feature_sets'][feature_name]['versions'][ver] = {
                        'shape': metadata.shape,
                        'dtype': metadata.dtype,
                        'created_at': metadata.created_at,
                        'size_mb': size_mb,
                        'description': metadata.description
                    }
                    
                    stats['feature_sets'][feature_name]['total_size_mb'] += size_mb
                    stats['total_size_mb'] += size_mb
                    stats['total_features'] += metadata.shape[1] if len(metadata.shape) > 1 else 1
                    stats['total_samples'] += metadata.shape[0]
                    
        return stats
    
    def list_features(self) -> Dict[str, List[str]]:
        """
        List all available feature sets and versions.
        
        Returns:
            Dictionary mapping feature names to list of versions
        """
        features = {}
        
        with h5py.File(self.store_path, 'r') as store:
            for name in store.keys():
                features[name] = list(store[name].keys())
                
        return features
    
    def delete_features(self, name: str, version: Optional[str] = None):
        """
        Delete features from the store.
        
        Args:
            name: Feature set name
            version: Specific version to delete (all versions if None)
        """
        with h5py.File(self.store_path, 'a') as store:
            if name not in store:
                raise KeyError(f"Feature set '{name}' not found")
                
            if version:
                if version in store[name]:
                    del store[name][version]
                    logger.info(f"Deleted features: {name}:{version}")
                else:
                    raise KeyError(f"Version '{version}' not found for '{name}'")
            else:
                del store[name]
                logger.info(f"Deleted all versions of feature set: {name}")
                
        # Clear cache
        if version:
            self._metadata_cache.pop(f"{name}/{version}", None)
        else:
            keys_to_remove = [k for k in self._metadata_cache if k.startswith(f"{name}/")]
            for key in keys_to_remove:
                self._metadata_cache.pop(key, None)
    
    def get_metadata(self, name: str, version: Optional[str] = None) -> FeatureMetadata:
        """
        Get metadata for a feature set.
        
        Args:
            name: Feature set name
            version: Version (latest if None)
            
        Returns:
            Feature metadata
        """
        # Check cache first
        if version:
            cache_key = f"{name}/{version}"
            if cache_key in self._metadata_cache:
                return self._metadata_cache[cache_key]
                
        # Load from store
        _, metadata = self.load_features(name, version)
        return metadata
    
    def _generate_version(self, features: np.ndarray) -> str:
        """Generate version string based on timestamp and data hash"""
        timestamp = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
        data_hash = self._compute_hash(features)[:8]
        return f"v_{timestamp}_{data_hash}"
    
    def _compute_hash(self, features: np.ndarray) -> str:
        """Compute hash of feature data"""
        # Use first and last 1000 rows for efficiency
        if features.shape[0] > 2000:
            sample = np.vstack([features[:1000], features[-1000:]])
        else:
            sample = features
            
        return hashlib.sha256(sample.tobytes()).hexdigest()
    
    def create_feature_pipeline(self, pipeline_config: Dict[str, Any]) -> 'FeaturePipeline':
        """
        Create a feature extraction pipeline.
        
        Args:
            pipeline_config: Configuration for the pipeline
            
        Returns:
            FeaturePipeline instance
        """
        return FeaturePipeline(self, pipeline_config)


class FeaturePipeline:
    """Pipeline for feature extraction and storage"""
    
    def __init__(self, feature_store: FeatureStore, config: Dict[str, Any]):
        """
        Initialize feature pipeline.
        
        Args:
            feature_store: FeatureStore instance
            config: Pipeline configuration
        """
        self.feature_store = feature_store
        self.config = config
        self.extractors = []
        
    def add_extractor(self, extractor_func, name: str, params: Optional[Dict] = None):
        """
        Add a feature extractor to the pipeline.
        
        Args:
            extractor_func: Function that takes data and returns features
            name: Name of the extractor
            params: Parameters for the extractor
        """
        self.extractors.append({
            'func': extractor_func,
            'name': name,
            'params': params or {}
        })
        
    def extract_and_store(self,
                         data: Union[np.ndarray, pd.DataFrame],
                         feature_set_name: str,
                         version: Optional[str] = None) -> str:
        """
        Extract features using all extractors and store them.
        
        Args:
            data: Input data
            feature_set_name: Name for the feature set
            version: Version string
            
        Returns:
            Feature set identifier
        """
        all_features = []
        feature_names = []
        
        for extractor in self.extractors:
            logger.info(f"Running extractor: {extractor['name']}")
            
            # Extract features
            features = extractor['func'](data, **extractor['params'])
            
            # Handle different return types
            if isinstance(features, pd.DataFrame):
                all_features.append(features.values)
                feature_names.extend([f"{extractor['name']}_{col}" for col in features.columns])
            else:
                all_features.append(features)
                n_features = features.shape[1] if len(features.shape) > 1 else 1
                feature_names.extend([f"{extractor['name']}_{i}" for i in range(n_features)])
                
        # Concatenate all features
        combined_features = np.hstack(all_features)
        
        # Create DataFrame for better handling
        feature_df = pd.DataFrame(combined_features, columns=feature_names)
        
        # Store features
        return self.feature_store.store_features(
            feature_df,
            name=feature_set_name,
            version=version,
            description=f"Features extracted using {len(self.extractors)} extractors",
            preprocessing_params={'extractors': [e['name'] for e in self.extractors]}
        )


def create_statistical_features(data: np.ndarray, **kwargs) -> pd.DataFrame:
    """
    Extract statistical features from packet data.
    
    Args:
        data: Packet byte data
        **kwargs: Additional parameters
        
    Returns:
        DataFrame with statistical features
    """
    features = {}
    
    # Basic statistics
    features['mean'] = np.mean(data, axis=1)
    features['std'] = np.std(data, axis=1)
    features['min'] = np.min(data, axis=1)
    features['max'] = np.max(data, axis=1)
    features['median'] = np.median(data, axis=1)
    
    # Percentiles
    features['p25'] = np.percentile(data, 25, axis=1)
    features['p75'] = np.percentile(data, 75, axis=1)
    
    # Entropy
    def entropy(x):
        _, counts = np.unique(x, return_counts=True)
        probs = counts / len(x)
        return -np.sum(probs * np.log2(probs + 1e-10))
    
    features['entropy'] = np.apply_along_axis(entropy, 1, data)
    
    return pd.DataFrame(features)


def create_frequency_features(data: np.ndarray, n_components: int = 10, **kwargs) -> pd.DataFrame:
    """
    Extract frequency domain features using FFT.
    
    Args:
        data: Packet byte data
        n_components: Number of frequency components to extract
        **kwargs: Additional parameters
        
    Returns:
        DataFrame with frequency features
    """
    # Apply FFT
    fft_data = np.fft.fft(data, axis=1)
    
    # Get magnitude spectrum
    magnitude = np.abs(fft_data)
    
    # Extract top frequency components
    features = {}
    for i in range(n_components):
        features[f'freq_component_{i}'] = magnitude[:, i]
        
    # Add spectral statistics
    features['spectral_centroid'] = np.sum(magnitude * np.arange(magnitude.shape[1]), axis=1) / np.sum(magnitude, axis=1)
    features['spectral_energy'] = np.sum(magnitude ** 2, axis=1)
    
    return pd.DataFrame(features)


def create_ngram_features(data: np.ndarray, n: int = 2, top_k: int = 20, **kwargs) -> pd.DataFrame:
    """
    Extract n-gram features from byte sequences.
    
    Args:
        data: Packet byte data
        n: N-gram size
        top_k: Number of top n-grams to use as features
        **kwargs: Additional parameters
        
    Returns:
        DataFrame with n-gram features
    """
    from collections import Counter
    
    # Count n-grams across all samples
    all_ngrams = Counter()
    
    for row in data:
        # Convert to string of bytes
        byte_str = ''.join([chr(b) for b in row])
        
        # Extract n-grams
        for i in range(len(byte_str) - n + 1):
            ngram = byte_str[i:i+n]
            all_ngrams[ngram] += 1
            
    # Get top k n-grams
    top_ngrams = [ngram for ngram, _ in all_ngrams.most_common(top_k)]
    
    # Create feature matrix
    features = {}
    for ngram in top_ngrams:
        features[f'ngram_{ngram}'] = []
        
    for row in data:
        byte_str = ''.join([chr(b) for b in row])
        row_ngrams = Counter()
        
        for i in range(len(byte_str) - n + 1):
            ngram = byte_str[i:i+n]
            if ngram in top_ngrams:
                row_ngrams[ngram] += 1
                
        for ngram in top_ngrams:
            features[f'ngram_{ngram}'].append(row_ngrams.get(ngram, 0))
            
    return pd.DataFrame(features)