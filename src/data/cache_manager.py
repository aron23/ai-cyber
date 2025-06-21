"""
Cache manager for efficient data storage and retrieval.
Supports HDF5, Parquet, and Pickle formats with compression.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
import hashlib
import json
import time
from datetime import datetime, timedelta
import h5py
import pickle
import gzip
from src.utils.logger import setup_logger
from src.utils.config_manager import get_config_manager

logger = setup_logger('cache_manager')


class CacheManager:
    """Manages cached data for the pipeline."""
    
    def __init__(self, cache_dir: Optional[Union[str, Path]] = None):
        """
        Initialize cache manager.
        
        Args:
            cache_dir: Directory for cache storage
        """
        config_mgr = get_config_manager()
        
        if cache_dir is None:
            cache_dir = config_mgr.get('data_config', 'data.cache_path', 'data/cache')
        
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Cache configuration
        cache_config = config_mgr.get('data_config', 'pipeline.cache', {})
        self.enabled = cache_config.get('enabled', True)
        self.format = cache_config.get('format', 'hdf5')
        self.compression = cache_config.get('compression', 'gzip')
        self.ttl_hours = cache_config.get('ttl_hours', 24)
        
        # Metadata file
        self.metadata_file = self.cache_dir / 'cache_metadata.json'
        self.metadata = self._load_metadata()
        
        logger.info(f"Cache manager initialized: {self.cache_dir}")
    
    def _load_metadata(self) -> Dict:
        """Load cache metadata."""
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_metadata(self):
        """Save cache metadata."""
        with open(self.metadata_file, 'w') as f:
            json.dump(self.metadata, f, indent=2)
    
    def _generate_cache_key(self, 
                          data_path: Union[str, Path],
                          params: Optional[Dict] = None) -> str:
        """
        Generate unique cache key based on data path and parameters.
        
        Args:
            data_path: Path to source data
            params: Additional parameters affecting the data
            
        Returns:
            Cache key string
        """
        # Create string representation of inputs
        key_parts = [str(data_path)]
        
        if params:
            # Sort params for consistent hashing
            sorted_params = json.dumps(params, sort_keys=True)
            key_parts.append(sorted_params)
        
        # Generate hash
        key_string = '|'.join(key_parts)
        cache_key = hashlib.md5(key_string.encode()).hexdigest()
        
        return cache_key
    
    def _get_cache_path(self, cache_key: str) -> Path:
        """Get file path for cache key."""
        extension = {
            'hdf5': '.h5',
            'parquet': '.parquet',
            'pickle': '.pkl'
        }.get(self.format, '.cache')
        
        if self.compression == 'gzip' and self.format == 'pickle':
            extension += '.gz'
        
        return self.cache_dir / f"{cache_key}{extension}"
    
    def is_cached(self, 
                  data_path: Union[str, Path],
                  params: Optional[Dict] = None) -> bool:
        """
        Check if data is cached and still valid.
        
        Args:
            data_path: Path to source data
            params: Additional parameters
            
        Returns:
            True if valid cache exists
        """
        if not self.enabled:
            return False
        
        cache_key = self._generate_cache_key(data_path, params)
        
        # Check metadata
        if cache_key not in self.metadata:
            return False
        
        # Check file exists
        cache_path = self._get_cache_path(cache_key)
        if not cache_path.exists():
            return False
        
        # Check TTL
        cached_time = datetime.fromisoformat(self.metadata[cache_key]['timestamp'])
        age_hours = (datetime.now() - cached_time).total_seconds() / 3600
        
        if age_hours > self.ttl_hours:
            logger.info(f"Cache expired for {cache_key} (age: {age_hours:.1f} hours)")
            return False
        
        return True
    
    def load_from_cache(self,
                       data_path: Union[str, Path],
                       params: Optional[Dict] = None) -> Optional[pd.DataFrame]:
        """
        Load data from cache.
        
        Args:
            data_path: Path to source data
            params: Additional parameters
            
        Returns:
            Cached DataFrame or None if not cached
        """
        if not self.is_cached(data_path, params):
            return None
        
        cache_key = self._generate_cache_key(data_path, params)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            start_time = time.time()
            
            if self.format == 'hdf5':
                df = pd.read_hdf(cache_path, key='data')
            elif self.format == 'parquet':
                df = pd.read_parquet(cache_path)
            elif self.format == 'pickle':
                if self.compression == 'gzip':
                    with gzip.open(cache_path, 'rb') as f:
                        df = pickle.load(f)
                else:
                    with open(cache_path, 'rb') as f:
                        df = pickle.load(f)
            else:
                raise ValueError(f"Unknown cache format: {self.format}")
            
            load_time = time.time() - start_time
            logger.info(f"Loaded from cache: {cache_key} ({len(df)} rows in {load_time:.2f}s)")
            
            # Update access time
            self.metadata[cache_key]['last_accessed'] = datetime.now().isoformat()
            self.metadata[cache_key]['access_count'] += 1
            self._save_metadata()
            
            return df
            
        except Exception as e:
            logger.error(f"Failed to load cache {cache_key}: {e}")
            # Remove invalid cache entry
            self.invalidate_cache(data_path, params)
            return None
    
    def save_to_cache(self,
                     data: pd.DataFrame,
                     data_path: Union[str, Path],
                     params: Optional[Dict] = None):
        """
        Save data to cache.
        
        Args:
            data: DataFrame to cache
            data_path: Path to source data
            params: Additional parameters
        """
        if not self.enabled:
            return
        
        cache_key = self._generate_cache_key(data_path, params)
        cache_path = self._get_cache_path(cache_key)
        
        try:
            start_time = time.time()
            
            if self.format == 'hdf5':
                data.to_hdf(cache_path, key='data', mode='w', complevel=9)
            elif self.format == 'parquet':
                data.to_parquet(cache_path, compression=self.compression)
            elif self.format == 'pickle':
                if self.compression == 'gzip':
                    with gzip.open(cache_path, 'wb') as f:
                        pickle.dump(data, f)
                else:
                    with open(cache_path, 'wb') as f:
                        pickle.dump(data, f)
            else:
                raise ValueError(f"Unknown cache format: {self.format}")
            
            save_time = time.time() - start_time
            file_size_mb = cache_path.stat().st_size / (1024 * 1024)
            
            # Update metadata
            self.metadata[cache_key] = {
                'data_path': str(data_path),
                'params': params,
                'timestamp': datetime.now().isoformat(),
                'last_accessed': datetime.now().isoformat(),
                'access_count': 0,
                'rows': len(data),
                'columns': len(data.columns),
                'file_size_mb': round(file_size_mb, 2),
                'format': self.format,
                'compression': self.compression
            }
            self._save_metadata()
            
            logger.info(f"Saved to cache: {cache_key} ({len(data)} rows, "
                       f"{file_size_mb:.2f} MB in {save_time:.2f}s)")
            
        except Exception as e:
            logger.error(f"Failed to save cache {cache_key}: {e}")
            # Clean up partial file
            if cache_path.exists():
                cache_path.unlink()
    
    def invalidate_cache(self,
                        data_path: Optional[Union[str, Path]] = None,
                        params: Optional[Dict] = None):
        """
        Invalidate cache entries.
        
        Args:
            data_path: Path to source data (None to clear all)
            params: Additional parameters
        """
        if data_path is None:
            # Clear all cache
            logger.info("Clearing all cache")
            for cache_file in self.cache_dir.glob('*'):
                if cache_file.name != 'cache_metadata.json':
                    cache_file.unlink()
            self.metadata = {}
            self._save_metadata()
        else:
            # Clear specific entry
            cache_key = self._generate_cache_key(data_path, params)
            cache_path = self._get_cache_path(cache_key)
            
            if cache_path.exists():
                cache_path.unlink()
            
            if cache_key in self.metadata:
                del self.metadata[cache_key]
                self._save_metadata()
            
            logger.info(f"Invalidated cache: {cache_key}")
    
    def cleanup_expired(self):
        """Remove expired cache entries."""
        expired_keys = []
        
        for cache_key, info in self.metadata.items():
            cached_time = datetime.fromisoformat(info['timestamp'])
            age_hours = (datetime.now() - cached_time).total_seconds() / 3600
            
            if age_hours > self.ttl_hours:
                expired_keys.append(cache_key)
        
        for cache_key in expired_keys:
            cache_path = self._get_cache_path(cache_key)
            if cache_path.exists():
                cache_path.unlink()
            del self.metadata[cache_key]
        
        if expired_keys:
            self._save_metadata()
            logger.info(f"Cleaned up {len(expired_keys)} expired cache entries")
    
    def get_cache_stats(self) -> Dict:
        """Get cache statistics."""
        stats = {
            'total_entries': len(self.metadata),
            'total_size_mb': 0,
            'format': self.format,
            'compression': self.compression,
            'ttl_hours': self.ttl_hours
        }
        
        # Calculate total size and access statistics
        total_accesses = 0
        oldest_entry = None
        newest_entry = None
        
        for info in self.metadata.values():
            stats['total_size_mb'] += info.get('file_size_mb', 0)
            total_accesses += info.get('access_count', 0)
            
            timestamp = datetime.fromisoformat(info['timestamp'])
            if oldest_entry is None or timestamp < oldest_entry:
                oldest_entry = timestamp
            if newest_entry is None or timestamp > newest_entry:
                newest_entry = timestamp
        
        stats['total_accesses'] = total_accesses
        stats['avg_accesses_per_entry'] = total_accesses / len(self.metadata) if self.metadata else 0
        
        if oldest_entry:
            stats['oldest_entry_age_hours'] = (datetime.now() - oldest_entry).total_seconds() / 3600
        if newest_entry:
            stats['newest_entry_age_hours'] = (datetime.now() - newest_entry).total_seconds() / 3600
        
        return stats
    
    def __repr__(self):
        stats = self.get_cache_stats()
        return (f"CacheManager(entries={stats['total_entries']}, "
                f"size={stats['total_size_mb']:.1f}MB, "
                f"format={self.format})")


# Cached data decorator
def cached_data(cache_manager: Optional[CacheManager] = None):
    """
    Decorator for caching function results.
    
    Args:
        cache_manager: CacheManager instance (creates new if None)
    """
    def decorator(func):
        def wrapper(data_path: Union[str, Path], *args, **kwargs):
            # Get or create cache manager
            cm = cache_manager or CacheManager()
            
            # Try to load from cache
            params = {'args': args, 'kwargs': kwargs}
            cached_result = cm.load_from_cache(data_path, params)
            
            if cached_result is not None:
                return cached_result
            
            # Call original function
            result = func(data_path, *args, **kwargs)
            
            # Save to cache if result is DataFrame
            if isinstance(result, pd.DataFrame):
                cm.save_to_cache(result, data_path, params)
            
            return result
        
        return wrapper
    return decorator


if __name__ == "__main__":
    # Example usage
    cache_mgr = CacheManager()
    
    # Show cache statistics
    stats = cache_mgr.get_cache_stats()
    print("Cache Statistics:")
    for key, value in stats.items():
        print(f"  {key}: {value}")
    
    # Clean up expired entries
    cache_mgr.cleanup_expired()