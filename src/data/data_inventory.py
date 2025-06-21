"""
Data inventory tool for Payload-Byte datasets.
Analyzes and documents dataset characteristics.
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
from datetime import datetime
from src.utils.logger import setup_logger

logger = setup_logger('data_inventory')


class DataInventory:
    """Class to create and manage data inventory for Payload-Byte datasets."""
    
    def __init__(self, data_path):
        self.data_path = Path(data_path)
        self.inventory = {}
        
    def analyze_csv_file(self, filepath):
        """Analyze a single CSV file and return statistics."""
        logger.info(f"Analyzing {filepath.name}...")
        
        try:
            # Read file with chunking for memory efficiency
            df_sample = pd.read_csv(filepath, nrows=1000)
            
            # Get total rows using chunking
            total_rows = sum(1 for _ in open(filepath)) - 1  # Subtract header
            
            # Get file size
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            
            # Analyze structure
            n_columns = len(df_sample.columns)
            column_names = df_sample.columns.tolist()
            
            # Identify feature types
            header_features = [col for col in column_names if col in ['src_ip', 'dst_ip', 'src_port', 'dst_port']]
            byte_features = [col for col in column_names if col.startswith('byte_')]
            label_column = 'label' if 'label' in column_names else None
            
            # Analyze data types
            dtypes = df_sample.dtypes.to_dict()
            
            # Check for missing values
            missing_counts = df_sample.isnull().sum().to_dict()
            missing_percentage = {k: (v/len(df_sample))*100 for k, v in missing_counts.items() if v > 0}
            
            # Label distribution (if exists)
            label_distribution = {}
            if label_column:
                # Read labels in chunks for large files
                label_counts = {}
                for chunk in pd.read_csv(filepath, chunksize=10000, usecols=[label_column]):
                    for label, count in chunk[label_column].value_counts().items():
                        label_counts[label] = label_counts.get(label, 0) + count
                label_distribution = label_counts
            
            # Analyze byte value statistics (sample)
            byte_stats = {}
            if byte_features:
                byte_sample = df_sample[byte_features[:10]].values.flatten()
                byte_stats = {
                    'min': int(np.min(byte_sample)),
                    'max': int(np.max(byte_sample)),
                    'mean': float(np.mean(byte_sample)),
                    'std': float(np.std(byte_sample)),
                    'unique_values': int(len(np.unique(byte_sample)))
                }
            
            return {
                'file_name': filepath.name,
                'file_size_mb': round(file_size_mb, 2),
                'total_rows': total_rows,
                'total_columns': n_columns,
                'header_features': header_features,
                'byte_features_count': len(byte_features),
                'label_column': label_column,
                'label_distribution': label_distribution,
                'missing_values': missing_percentage,
                'byte_value_statistics': byte_stats,
                'data_types_summary': {
                    'object': sum(1 for dtype in dtypes.values() if dtype == 'object'),
                    'int64': sum(1 for dtype in dtypes.values() if dtype == 'int64'),
                    'float64': sum(1 for dtype in dtypes.values() if dtype == 'float64')
                }
            }
            
        except Exception as e:
            logger.error(f"Error analyzing {filepath}: {str(e)}")
            return None
    
    def create_inventory(self):
        """Create inventory for all CSV files in the data path."""
        csv_files = list(self.data_path.glob('*.csv'))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {self.data_path}")
            return
        
        logger.info(f"Found {len(csv_files)} CSV files to analyze")
        
        for csv_file in csv_files:
            analysis = self.analyze_csv_file(csv_file)
            if analysis:
                self.inventory[csv_file.name] = analysis
        
        # Add summary statistics
        self.inventory['summary'] = self._create_summary()
        self.inventory['created_at'] = datetime.now().isoformat()
        
        return self.inventory
    
    def _create_summary(self):
        """Create summary statistics across all files."""
        if not self.inventory:
            return {}
        
        total_size = sum(file_info['file_size_mb'] for file_info in self.inventory.values() 
                        if isinstance(file_info, dict) and 'file_size_mb' in file_info)
        total_rows = sum(file_info['total_rows'] for file_info in self.inventory.values() 
                        if isinstance(file_info, dict) and 'total_rows' in file_info)
        
        # Aggregate label distribution
        all_labels = {}
        for file_info in self.inventory.values():
            if isinstance(file_info, dict) and 'label_distribution' in file_info:
                for label, count in file_info['label_distribution'].items():
                    all_labels[label] = all_labels.get(label, 0) + count
        
        return {
            'total_files': len([k for k in self.inventory.keys() if k != 'summary' and k != 'created_at']),
            'total_size_mb': round(total_size, 2),
            'total_records': total_rows,
            'combined_label_distribution': all_labels,
            'label_balance': {
                'benign_ratio': round(all_labels.get(0, 0) / total_rows * 100, 2) if total_rows > 0 else 0,
                'attack_ratio': round(sum(v for k, v in all_labels.items() if k != 0) / total_rows * 100, 2) if total_rows > 0 else 0
            }
        }
    
    def save_inventory(self, output_path=None):
        """Save inventory to JSON file."""
        if output_path is None:
            output_path = self.data_path / 'data_inventory.json'
        
        with open(output_path, 'w') as f:
            json.dump(self.inventory, f, indent=2)
        
        logger.info(f"Data inventory saved to {output_path}")
        
    def print_summary(self):
        """Print a formatted summary of the inventory."""
        if 'summary' not in self.inventory:
            logger.warning("No inventory summary available")
            return
        
        summary = self.inventory['summary']
        
        print("\n" + "="*60)
        print("DATA INVENTORY SUMMARY")
        print("="*60)
        print(f"Total Files: {summary.get('total_files', 0)}")
        print(f"Total Size: {summary.get('total_size_mb', 0):.2f} MB")
        print(f"Total Records: {summary.get('total_records', 0):,}")
        print(f"\nLabel Distribution:")
        for label, count in summary.get('combined_label_distribution', {}).items():
            print(f"  Label {label}: {count:,} records")
        print(f"\nClass Balance:")
        print(f"  Benign: {summary.get('label_balance', {}).get('benign_ratio', 0):.2f}%")
        print(f"  Attack: {summary.get('label_balance', {}).get('attack_ratio', 0):.2f}%")
        print("="*60 + "\n")


def create_data_inventory():
    """Main function to create data inventory."""
    data_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'payload_byte'
    
    inventory = DataInventory(data_path)
    inventory.create_inventory()
    inventory.save_inventory()
    inventory.print_summary()
    
    return inventory


if __name__ == "__main__":
    create_data_inventory()