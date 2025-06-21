"""
Sample data generator for Payload-Byte dataset format.
Creates sample data that mimics the structure of the actual datasets.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import random
from src.utils.logger import setup_logger

logger = setup_logger('sample_data_generator')


def generate_sample_payload_byte_data(n_samples=1000, output_path=None):
    """
    Generate sample data in Payload-Byte format.
    
    Args:
        n_samples: Number of samples to generate
        output_path: Path to save the CSV file
        
    Returns:
        pd.DataFrame: Generated sample data
    """
    logger.info(f"Generating {n_samples} sample records...")
    
    # Column names based on Payload-Byte format
    columns = ['src_ip', 'dst_ip', 'src_port', 'dst_port'] + [f'byte_{i}' for i in range(1500)]
    columns.append('label')
    
    data = []
    
    # Generate sample IPs
    src_ips = [f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}" for _ in range(10)]
    dst_ips = [f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}" for _ in range(10)]
    
    # Common ports
    common_ports = [80, 443, 22, 21, 25, 3306, 3389, 8080, 8443, 53]
    
    # Attack types (0 = benign, 1-5 = different attack types)
    attack_types = ['benign', 'dos', 'probe', 'r2l', 'u2r', 'exploit']
    
    for i in range(n_samples):
        record = []
        
        # Header features
        record.append(random.choice(src_ips))
        record.append(random.choice(dst_ips))
        record.append(random.choice(common_ports + list(range(1024, 65536, 100))))
        record.append(random.choice(common_ports + list(range(1024, 65536, 100))))
        
        # Payload bytes
        # Simulate different patterns for different attack types
        attack_type = random.choice(attack_types)
        
        if attack_type == 'benign':
            # Normal traffic - more varied byte patterns
            payload = np.random.randint(0, 256, 1500)
            label = 0
        elif attack_type == 'dos':
            # DoS - repetitive patterns
            pattern = np.random.randint(0, 256, 50)
            payload = np.tile(pattern, 30)[:1500]
            label = 1
        elif attack_type == 'probe':
            # Probe - specific byte sequences
            payload = np.zeros(1500, dtype=int)
            payload[:100] = np.random.randint(200, 256, 100)
            label = 2
        elif attack_type == 'r2l':
            # Remote to local - embedded patterns
            payload = np.random.randint(0, 128, 1500)
            payload[500:600] = 255  # Suspicious pattern
            label = 3
        elif attack_type == 'u2r':
            # User to root - privilege escalation patterns
            payload = np.random.randint(0, 256, 1500)
            payload[::10] = 0  # Regular nulls
            label = 4
        else:  # exploit
            # Exploit - buffer overflow patterns
            payload = np.full(1500, 90)  # NOP sled
            payload[1000:] = np.random.randint(0, 256, 500)
            label = 5
        
        record.extend(payload.tolist())
        record.append(label)
        
        data.append(record)
        
        if (i + 1) % 100 == 0:
            logger.debug(f"Generated {i + 1}/{n_samples} samples")
    
    # Create DataFrame
    df = pd.DataFrame(data, columns=columns)
    
    # Save if path provided
    if output_path:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_path, index=False)
        logger.info(f"Sample data saved to {output_path}")
    
    return df


def create_sample_datasets():
    """Create sample datasets for both UNSW-NB15 and CIC-IDS2017 formats."""
    base_path = Path(__file__).parent.parent.parent / 'data' / 'raw' / 'payload_byte'
    
    # Generate UNSW-NB15 sample
    logger.info("Generating UNSW-NB15 sample dataset...")
    unsw_df = generate_sample_payload_byte_data(
        n_samples=2000,
        output_path=base_path / 'unsw_nb15_sample.csv'
    )
    
    # Generate CIC-IDS2017 sample
    logger.info("Generating CIC-IDS2017 sample dataset...")
    cic_df = generate_sample_payload_byte_data(
        n_samples=2000,
        output_path=base_path / 'cic_ids2017_sample.csv'
    )
    
    # Create metadata file
    metadata = {
        'unsw_nb15_sample.csv': {
            'records': len(unsw_df),
            'features': len(unsw_df.columns) - 1,
            'labels': unsw_df['label'].value_counts().to_dict()
        },
        'cic_ids2017_sample.csv': {
            'records': len(cic_df),
            'features': len(cic_df.columns) - 1,
            'labels': cic_df['label'].value_counts().to_dict()
        }
    }
    
    import json
    with open(base_path / 'sample_metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    logger.info("Sample datasets created successfully!")
    return unsw_df, cic_df


if __name__ == "__main__":
    create_sample_datasets()