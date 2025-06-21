"""
Data validation utilities for Payload-Byte datasets.
Checks data integrity, format compliance, and identifies issues.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from typing import Dict, List, Tuple, Optional
from src.utils.logger import setup_logger

logger = setup_logger('data_validation')


class DataValidator:
    """Validates Payload-Byte dataset format and integrity."""
    
    def __init__(self):
        self.validation_results = {}
        self.expected_columns = self._get_expected_columns()
        
    def _get_expected_columns(self) -> List[str]:
        """Get expected column names for Payload-Byte format."""
        return ['src_ip', 'dst_ip', 'src_port', 'dst_port'] + \
               [f'byte_{i}' for i in range(1500)] + ['label']
    
    def validate_file(self, filepath: Path) -> Dict:
        """
        Validate a single data file.
        
        Args:
            filepath: Path to the CSV file
            
        Returns:
            Dictionary containing validation results
        """
        logger.info(f"Validating {filepath.name}...")
        
        results = {
            'file_name': filepath.name,
            'valid': True,
            'errors': [],
            'warnings': [],
            'statistics': {}
        }
        
        try:
            # Check file exists and is readable
            if not filepath.exists():
                results['valid'] = False
                results['errors'].append("File does not exist")
                return results
            
            # Check file size
            file_size_mb = filepath.stat().st_size / (1024 * 1024)
            results['statistics']['file_size_mb'] = round(file_size_mb, 2)
            
            if file_size_mb == 0:
                results['valid'] = False
                results['errors'].append("File is empty")
                return results
            
            # Read sample for validation
            df_sample = pd.read_csv(filepath, nrows=1000)
            
            # Validate column structure
            self._validate_columns(df_sample, results)
            
            # Validate data types
            self._validate_data_types(df_sample, results)
            
            # Validate value ranges
            self._validate_value_ranges(df_sample, results)
            
            # Check for missing values
            self._check_missing_values(df_sample, results)
            
            # Validate IP addresses
            self._validate_ip_addresses(df_sample, results)
            
            # Validate ports
            self._validate_ports(df_sample, results)
            
            # Check data consistency
            self._check_data_consistency(df_sample, results)
            
            # Get row count
            total_rows = sum(1 for _ in open(filepath)) - 1
            results['statistics']['total_rows'] = total_rows
            
        except Exception as e:
            results['valid'] = False
            results['errors'].append(f"Error reading file: {str(e)}")
        
        return results
    
    def _validate_columns(self, df: pd.DataFrame, results: Dict):
        """Validate column structure."""
        actual_columns = df.columns.tolist()
        
        # Check for missing columns
        missing_columns = set(self.expected_columns) - set(actual_columns)
        if missing_columns:
            results['valid'] = False
            results['errors'].append(f"Missing columns: {list(missing_columns)[:10]}...")
        
        # Check for extra columns
        extra_columns = set(actual_columns) - set(self.expected_columns)
        if extra_columns:
            results['warnings'].append(f"Extra columns found: {list(extra_columns)}")
        
        # Check column order
        if actual_columns[:4] != ['src_ip', 'dst_ip', 'src_port', 'dst_port']:
            results['warnings'].append("Header columns not in expected order")
    
    def _validate_data_types(self, df: pd.DataFrame, results: Dict):
        """Validate data types of columns."""
        # Check IP columns are strings
        for col in ['src_ip', 'dst_ip']:
            if col in df.columns and df[col].dtype != 'object':
                results['warnings'].append(f"{col} should be string type")
        
        # Check port columns are numeric
        for col in ['src_port', 'dst_port']:
            if col in df.columns and not pd.api.types.is_numeric_dtype(df[col]):
                results['errors'].append(f"{col} must be numeric")
                results['valid'] = False
        
        # Check byte columns are numeric
        byte_columns = [col for col in df.columns if col.startswith('byte_')]
        for col in byte_columns[:10]:  # Check first 10 as sample
            if not pd.api.types.is_numeric_dtype(df[col]):
                results['errors'].append(f"Byte columns must be numeric")
                results['valid'] = False
                break
        
        # Check label column
        if 'label' in df.columns and not pd.api.types.is_numeric_dtype(df['label']):
            results['errors'].append("Label column must be numeric")
            results['valid'] = False
    
    def _validate_value_ranges(self, df: pd.DataFrame, results: Dict):
        """Validate value ranges for different column types."""
        # Check byte values are in range [0, 255]
        byte_columns = [col for col in df.columns if col.startswith('byte_')]
        if byte_columns:
            byte_sample = df[byte_columns[:100]].values.flatten()
            min_val, max_val = np.min(byte_sample), np.max(byte_sample)
            
            if min_val < 0 or max_val > 255:
                results['errors'].append(f"Byte values out of range [0,255]: min={min_val}, max={max_val}")
                results['valid'] = False
            
            results['statistics']['byte_value_range'] = {'min': int(min_val), 'max': int(max_val)}
        
        # Check port ranges
        for col in ['src_port', 'dst_port']:
            if col in df.columns:
                port_values = df[col].dropna()
                if len(port_values) > 0:
                    min_port, max_port = port_values.min(), port_values.max()
                    if min_port < 0 or max_port > 65535:
                        results['errors'].append(f"{col} values out of valid range [0,65535]")
                        results['valid'] = False
        
        # Check label values
        if 'label' in df.columns:
            unique_labels = df['label'].unique()
            results['statistics']['unique_labels'] = sorted(unique_labels.tolist())
            if any(label < 0 for label in unique_labels):
                results['warnings'].append("Negative label values found")
    
    def _check_missing_values(self, df: pd.DataFrame, results: Dict):
        """Check for missing values."""
        missing_counts = df.isnull().sum()
        total_missing = missing_counts.sum()
        
        if total_missing > 0:
            missing_cols = missing_counts[missing_counts > 0].to_dict()
            results['warnings'].append(f"Missing values found in {len(missing_cols)} columns")
            results['statistics']['missing_values'] = {
                'total': int(total_missing),
                'by_column': {k: int(v) for k, v in list(missing_cols.items())[:10]}
            }
    
    def _validate_ip_addresses(self, df: pd.DataFrame, results: Dict):
        """Validate IP address format."""
        import re
        ip_pattern = re.compile(r'^(\d{1,3}\.){3}\d{1,3}$')
        
        for col in ['src_ip', 'dst_ip']:
            if col in df.columns:
                # Sample check on first 100 non-null values
                sample_ips = df[col].dropna().head(100)
                invalid_ips = []
                
                for ip in sample_ips:
                    if not ip_pattern.match(str(ip)):
                        invalid_ips.append(str(ip))
                    else:
                        # Check octets are in valid range
                        octets = str(ip).split('.')
                        if any(int(octet) > 255 for octet in octets):
                            invalid_ips.append(str(ip))
                
                if invalid_ips:
                    results['warnings'].append(f"Invalid IP addresses in {col}: {invalid_ips[:5]}")
    
    def _validate_ports(self, df: pd.DataFrame, results: Dict):
        """Validate port numbers."""
        for col in ['src_port', 'dst_port']:
            if col in df.columns:
                ports = df[col].dropna()
                if len(ports) > 0:
                    # Check for common invalid port numbers
                    if (ports == 0).any():
                        results['warnings'].append(f"Port 0 found in {col}")
                    
                    # Check distribution
                    well_known = (ports < 1024).sum()
                    registered = ((ports >= 1024) & (ports < 49152)).sum()
                    dynamic = (ports >= 49152).sum()
                    
                    results['statistics'][f'{col}_distribution'] = {
                        'well_known': int(well_known),
                        'registered': int(registered),
                        'dynamic': int(dynamic)
                    }
    
    def _check_data_consistency(self, df: pd.DataFrame, results: Dict):
        """Check for data consistency issues."""
        # Check for duplicate rows
        duplicates = df.duplicated().sum()
        if duplicates > 0:
            results['warnings'].append(f"Found {duplicates} duplicate rows in sample")
        
        # Check byte columns consistency
        byte_columns = [col for col in df.columns if col.startswith('byte_')]
        if byte_columns:
            # Check if all byte values are integers
            sample_bytes = df[byte_columns[:10]].values.flatten()
            non_integer = np.sum(sample_bytes != sample_bytes.astype(int))
            if non_integer > 0:
                results['warnings'].append("Non-integer values found in byte columns")
    
    def validate_dataset(self, data_path: Path) -> Dict:
        """
        Validate all CSV files in a directory.
        
        Args:
            data_path: Path to directory containing CSV files
            
        Returns:
            Dictionary containing validation results for all files
        """
        csv_files = list(data_path.glob('*.csv'))
        
        if not csv_files:
            logger.warning(f"No CSV files found in {data_path}")
            return {}
        
        all_results = {}
        valid_count = 0
        
        for csv_file in csv_files:
            result = self.validate_file(csv_file)
            all_results[csv_file.name] = result
            if result['valid']:
                valid_count += 1
        
        # Summary
        all_results['summary'] = {
            'total_files': len(csv_files),
            'valid_files': valid_count,
            'invalid_files': len(csv_files) - valid_count,
            'validation_rate': round((valid_count / len(csv_files)) * 100, 2)
        }
        
        return all_results
    
    def save_validation_report(self, results: Dict, output_path: Optional[Path] = None):
        """Save validation report to file."""
        import json
        from datetime import datetime
        
        if output_path is None:
            output_path = Path('data/raw/payload_byte/validation_report.json')
        
        report = {
            'validation_date': datetime.now().isoformat(),
            'results': results
        }
        
        with open(output_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Validation report saved to {output_path}")
    
    def print_summary(self, results: Dict):
        """Print validation summary."""
        summary = results.get('summary', {})
        
        print("\n" + "="*60)
        print("DATA VALIDATION SUMMARY")
        print("="*60)
        print(f"Total Files: {summary.get('total_files', 0)}")
        print(f"Valid Files: {summary.get('valid_files', 0)}")
        print(f"Invalid Files: {summary.get('invalid_files', 0)}")
        print(f"Validation Rate: {summary.get('validation_rate', 0)}%")
        
        # Print errors and warnings
        for filename, result in results.items():
            if filename != 'summary' and isinstance(result, dict):
                if result['errors'] or result['warnings']:
                    print(f"\n{filename}:")
                    if result['errors']:
                        print(f"  Errors: {len(result['errors'])}")
                        for error in result['errors'][:3]:
                            print(f"    - {error}")
                    if result['warnings']:
                        print(f"  Warnings: {len(result['warnings'])}")
                        for warning in result['warnings'][:3]:
                            print(f"    - {warning}")
        
        print("="*60 + "\n")


def validate_payload_byte_data():
    """Main function to validate Payload-Byte data."""
    data_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'payload_byte'
    
    validator = DataValidator()
    results = validator.validate_dataset(data_path)
    validator.save_validation_report(results)
    validator.print_summary(results)
    
    return results


if __name__ == "__main__":
    validate_payload_byte_data()