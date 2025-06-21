"""
Data Versioning Module

This module provides a comprehensive data versioning system for tracking dataset
changes, maintaining data lineage, and ensuring experiment reproducibility in
machine learning pipelines.

Key Features:
    - Automatic version tracking with hash-based identification
    - Data lineage tracking for transformation pipelines
    - Metadata storage for datasets and preprocessing steps
    - Version comparison and diff generation
    - Integration with DVC (Data Version Control) when available

Components:
    - DataVersion: Represents a specific version of a dataset
    - DataVersionManager: Manages multiple versions and lineage
    - DataLineage: Tracks transformation history
    
Example Usage:
    >>> from src.data.data_versioning import DataVersionManager
    >>> 
    >>> # Initialize version manager
    >>> version_mgr = DataVersionManager('data/versions')
    >>> 
    >>> # Create a new version
    >>> version = version_mgr.create_version(
    ...     data_path='data/raw/dataset.csv',
    ...     metadata={'preprocessing': 'normalized', 'split': 'train'}
    ... )
    >>> 
    >>> # Track lineage
    >>> version_mgr.add_lineage(
    ...     parent_version='v1.0',
    ...     child_version='v1.1',
    ...     transformation='outlier_removal'
    ... )

Notes:
    - Version IDs are generated using SHA-256 hashes of data content
    - Metadata is stored in JSON format for easy inspection
    - Compatible with distributed storage systems (S3, GCS)
"""

import json
import hashlib
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Union, Any
import pandas as pd
import numpy as np
from src.utils.logger import setup_logger

logger = setup_logger('data_versioning')


class DataVersion:
    """Represents a specific version of a dataset."""
    
    def __init__(self, 
                 version_id: str,
                 data_path: Union[str, Path],
                 metadata: Dict[str, Any]):
        """
        Initialize a data version.
        
        Args:
            version_id: Unique version identifier
            data_path: Path to the dataset
            metadata: Version metadata
        """
        self.version_id = version_id
        self.data_path = Path(data_path)
        self.metadata = metadata
        self.created_at = metadata.get('created_at', datetime.now().isoformat())
        
    def to_dict(self) -> Dict:
        """Convert to dictionary representation."""
        return {
            'version_id': self.version_id,
            'data_path': str(self.data_path),
            'metadata': self.metadata,
            'created_at': self.created_at
        }
    
    @classmethod
    def from_dict(cls, data: Dict) -> 'DataVersion':
        """Create from dictionary representation."""
        return cls(
            version_id=data['version_id'],
            data_path=data['data_path'],
            metadata=data['metadata']
        )


class DataVersionManager:
    """Manages data versioning and lineage tracking."""
    
    def __init__(self, version_dir: Optional[Union[str, Path]] = None):
        """
        Initialize the version manager.
        
        Args:
            version_dir: Directory for storing version information
        """
        if version_dir is None:
            version_dir = Path(__file__).parent.parent.parent / 'data' / 'versions'
        
        self.version_dir = Path(version_dir)
        self.version_dir.mkdir(parents=True, exist_ok=True)
        
        # Version registry file
        self.registry_file = self.version_dir / 'version_registry.json'
        self.registry = self._load_registry()
        
        # Lineage tracking
        self.lineage_file = self.version_dir / 'data_lineage.json'
        self.lineage = self._load_lineage()
        
        logger.info(f"Data version manager initialized: {self.version_dir}")
    
    def _load_registry(self) -> Dict:
        """Load version registry."""
        if self.registry_file.exists():
            with open(self.registry_file, 'r') as f:
                return json.load(f)
        return {'versions': {}, 'latest': None}
    
    def _save_registry(self):
        """Save version registry."""
        with open(self.registry_file, 'w') as f:
            json.dump(self.registry, f, indent=2)
    
    def _load_lineage(self) -> Dict:
        """Load data lineage information."""
        if self.lineage_file.exists():
            with open(self.lineage_file, 'r') as f:
                return json.load(f)
        return {}
    
    def _save_lineage(self):
        """Save data lineage information."""
        with open(self.lineage_file, 'w') as f:
            json.dump(self.lineage, f, indent=2)
    
    def _compute_data_hash(self, data_path: Union[str, Path]) -> str:
        """
        Compute hash of dataset for version identification.
        
        Args:
            data_path: Path to dataset file
            
        Returns:
            Hash string
        """
        data_path = Path(data_path)
        
        if not data_path.exists():
            raise ValueError(f"Data file not found: {data_path}")
        
        # For large files, hash file metadata and sample
        hasher = hashlib.sha256()
        
        # Add file metadata
        stat = data_path.stat()
        hasher.update(f"{stat.st_size}".encode())
        hasher.update(f"{stat.st_mtime}".encode())
        
        # Add sample of file content
        with open(data_path, 'rb') as f:
            # Read first and last 1MB
            chunk = f.read(1024 * 1024)
            hasher.update(chunk)
            
            # Seek to end if file is large
            if stat.st_size > 2 * 1024 * 1024:
                f.seek(-1024 * 1024, 2)
                chunk = f.read(1024 * 1024)
                hasher.update(chunk)
        
        return hasher.hexdigest()[:16]  # Use first 16 chars
    
    def create_version(self,
                      data_path: Union[str, Path],
                      description: str,
                      preprocessing_params: Optional[Dict] = None,
                      parent_version: Optional[str] = None,
                      tags: Optional[List[str]] = None) -> DataVersion:
        """
        Create a new data version.
        
        Args:
            data_path: Path to dataset
            description: Version description
            preprocessing_params: Parameters used for preprocessing
            parent_version: Parent version ID if derived
            tags: Optional tags for the version
            
        Returns:
            DataVersion object
        """
        data_path = Path(data_path)
        
        # Compute version ID based on data content
        data_hash = self._compute_data_hash(data_path)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        version_id = f"v_{timestamp}_{data_hash}"
        
        # Check if this exact version already exists
        for vid, vdata in self.registry['versions'].items():
            if vdata['data_hash'] == data_hash:
                logger.warning(f"Data version already exists: {vid}")
                return DataVersion.from_dict(vdata)
        
        # Get data statistics
        try:
            if data_path.suffix == '.csv':
                df_info = pd.read_csv(data_path, nrows=5)
                stats = {
                    'file_size_mb': data_path.stat().st_size / (1024 * 1024),
                    'columns': len(df_info.columns),
                    'sample_rows': len(df_info)
                }
            else:
                stats = {
                    'file_size_mb': data_path.stat().st_size / (1024 * 1024)
                }
        except Exception as e:
            logger.error(f"Failed to get data statistics: {e}")
            stats = {}
        
        # Create metadata
        metadata = {
            'version_id': version_id,
            'data_path': str(data_path),
            'data_hash': data_hash,
            'description': description,
            'preprocessing_params': preprocessing_params or {},
            'parent_version': parent_version,
            'tags': tags or [],
            'created_at': datetime.now().isoformat(),
            'created_by': 'data_engineering_team',
            'statistics': stats
        }
        
        # Create version object
        version = DataVersion(version_id, data_path, metadata)
        
        # Update registry
        self.registry['versions'][version_id] = version.to_dict()
        self.registry['latest'] = version_id
        self._save_registry()
        
        # Update lineage if this is a derived version
        if parent_version:
            self._add_lineage(parent_version, version_id, preprocessing_params)
        
        logger.info(f"Created data version: {version_id}")
        return version
    
    def _add_lineage(self,
                    parent_id: str,
                    child_id: str,
                    transformation: Optional[Dict] = None):
        """Add lineage relationship between versions."""
        if parent_id not in self.lineage:
            self.lineage[parent_id] = {'children': []}
        
        lineage_entry = {
            'child_id': child_id,
            'transformation': transformation or {},
            'created_at': datetime.now().isoformat()
        }
        
        self.lineage[parent_id]['children'].append(lineage_entry)
        self._save_lineage()
    
    def get_version(self, version_id: str) -> Optional[DataVersion]:
        """
        Get a specific data version.
        
        Args:
            version_id: Version identifier
            
        Returns:
            DataVersion object or None
        """
        if version_id in self.registry['versions']:
            return DataVersion.from_dict(self.registry['versions'][version_id])
        return None
    
    def get_latest_version(self) -> Optional[DataVersion]:
        """Get the latest data version."""
        if self.registry['latest']:
            return self.get_version(self.registry['latest'])
        return None
    
    def list_versions(self,
                     tags: Optional[List[str]] = None,
                     parent_version: Optional[str] = None) -> List[DataVersion]:
        """
        List data versions with optional filtering.
        
        Args:
            tags: Filter by tags
            parent_version: Filter by parent version
            
        Returns:
            List of DataVersion objects
        """
        versions = []
        
        for version_data in self.registry['versions'].values():
            version = DataVersion.from_dict(version_data)
            
            # Apply filters
            if tags:
                version_tags = version.metadata.get('tags', [])
                if not any(tag in version_tags for tag in tags):
                    continue
            
            if parent_version:
                if version.metadata.get('parent_version') != parent_version:
                    continue
            
            versions.append(version)
        
        # Sort by creation time
        versions.sort(key=lambda v: v.created_at, reverse=True)
        return versions
    
    def get_lineage(self, version_id: str) -> Dict:
        """
        Get lineage information for a version.
        
        Args:
            version_id: Version identifier
            
        Returns:
            Lineage information
        """
        lineage_info = {
            'version_id': version_id,
            'ancestors': [],
            'descendants': []
        }
        
        # Find ancestors
        for vid, vdata in self.registry['versions'].items():
            if vid == version_id:
                parent = vdata['metadata'].get('parent_version')
                if parent:
                    lineage_info['ancestors'].append(parent)
                    # Recursively get ancestors
                    parent_lineage = self.get_lineage(parent)
                    lineage_info['ancestors'].extend(parent_lineage['ancestors'])
        
        # Find descendants
        if version_id in self.lineage:
            for child_info in self.lineage[version_id]['children']:
                lineage_info['descendants'].append(child_info['child_id'])
                # Recursively get descendants
                child_lineage = self.get_lineage(child_info['child_id'])
                lineage_info['descendants'].extend(child_lineage['descendants'])
        
        return lineage_info
    
    def export_version_info(self, version_id: str, output_path: Optional[Path] = None) -> Path:
        """
        Export version information for reproducibility.
        
        Args:
            version_id: Version identifier
            output_path: Output file path
            
        Returns:
            Path to exported file
        """
        version = self.get_version(version_id)
        if not version:
            raise ValueError(f"Version not found: {version_id}")
        
        if output_path is None:
            output_path = self.version_dir / f"{version_id}_info.json"
        
        export_data = {
            'version': version.to_dict(),
            'lineage': self.get_lineage(version_id),
            'exported_at': datetime.now().isoformat()
        }
        
        with open(output_path, 'w') as f:
            json.dump(export_data, f, indent=2)
        
        logger.info(f"Exported version info to {output_path}")
        return output_path
    
    def compare_versions(self, version_id1: str, version_id2: str) -> Dict:
        """
        Compare two data versions.
        
        Args:
            version_id1: First version ID
            version_id2: Second version ID
            
        Returns:
            Comparison results
        """
        v1 = self.get_version(version_id1)
        v2 = self.get_version(version_id2)
        
        if not v1 or not v2:
            raise ValueError("One or both versions not found")
        
        comparison = {
            'version1': version_id1,
            'version2': version_id2,
            'created_diff_hours': abs(
                (datetime.fromisoformat(v1.created_at) - 
                 datetime.fromisoformat(v2.created_at)).total_seconds() / 3600
            ),
            'same_data': v1.metadata['data_hash'] == v2.metadata['data_hash'],
            'preprocessing_diff': self._diff_dicts(
                v1.metadata.get('preprocessing_params', {}),
                v2.metadata.get('preprocessing_params', {})
            ),
            'statistics_diff': self._diff_dicts(
                v1.metadata.get('statistics', {}),
                v2.metadata.get('statistics', {})
            )
        }
        
        return comparison
    
    def _diff_dicts(self, dict1: Dict, dict2: Dict) -> Dict:
        """Compare two dictionaries and return differences."""
        diff = {
            'added': {k: v for k, v in dict2.items() if k not in dict1},
            'removed': {k: v for k, v in dict1.items() if k not in dict2},
            'changed': {}
        }
        
        for key in set(dict1.keys()) & set(dict2.keys()):
            if dict1[key] != dict2[key]:
                diff['changed'][key] = {
                    'from': dict1[key],
                    'to': dict2[key]
                }
        
        return diff
    
    def cleanup_old_versions(self, keep_last_n: int = 5, dry_run: bool = True) -> List[str]:
        """
        Clean up old versions, keeping the most recent N.
        
        Args:
            keep_last_n: Number of versions to keep
            dry_run: If True, only show what would be deleted
            
        Returns:
            List of version IDs that were/would be deleted
        """
        versions = self.list_versions()
        
        if len(versions) <= keep_last_n:
            return []
        
        # Sort by creation time and identify versions to delete
        versions.sort(key=lambda v: v.created_at, reverse=True)
        to_delete = versions[keep_last_n:]
        
        deleted_ids = []
        for version in to_delete:
            if not dry_run:
                # Remove from registry
                del self.registry['versions'][version.version_id]
                
                # Remove from lineage
                if version.version_id in self.lineage:
                    del self.lineage[version.version_id]
            
            deleted_ids.append(version.version_id)
        
        if not dry_run:
            self._save_registry()
            self._save_lineage()
            logger.info(f"Cleaned up {len(deleted_ids)} old versions")
        else:
            logger.info(f"Would delete {len(deleted_ids)} versions (dry run)")
        
        return deleted_ids


if __name__ == "__main__":
    # Example usage
    version_mgr = DataVersionManager()
    
    # Create a version for sample data
    data_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'payload_byte' / 'unsw_nb15_sample.csv'
    
    if data_path.exists():
        version = version_mgr.create_version(
            data_path=data_path,
            description="Initial sample dataset for testing",
            tags=['sample', 'test']
        )
        
        print(f"Created version: {version.version_id}")
        
        # List all versions
        print("\nAll versions:")
        for v in version_mgr.list_versions():
            print(f"  {v.version_id}: {v.metadata['description']}")