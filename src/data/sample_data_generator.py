"""
Sample data generator for Payload-Byte dataset format.
Creates sample data that mimics the structure of the actual datasets.
"""

import numpy as np
import pandas as pd
from pathlib import Path
import random
from typing import Tuple, List
from src.utils.logger import setup_logger

logger = setup_logger('sample_data_generator')


class SampleDataGenerator:
    """
    Generator for creating sample network packet data for testing and demonstration.
    """
    
    def __init__(self, seed: int = 42):
        """
        Initialize the sample data generator.
        
        Args:
            seed: Random seed for reproducible results
        """
        self.seed = seed
        np.random.seed(seed)
        random.seed(seed)
        
        # Common network addresses and ports
        self.src_ips = [f"192.168.{random.randint(1, 255)}.{random.randint(1, 255)}" for _ in range(10)]
        self.dst_ips = [f"10.0.{random.randint(1, 255)}.{random.randint(1, 255)}" for _ in range(10)]
        self.common_ports = [80, 443, 22, 21, 25, 3306, 3389, 8080, 8443, 53]
        
    def generate_packet_data(self, 
                           n_samples: int = 100,
                           benign_ratio: float = 0.7,
                           packet_size_range: Tuple[int, int] = (256, 1500)) -> Tuple[List[np.ndarray], np.ndarray]:
        """
        Generate sample packet data with labels.
        
        Args:
            n_samples: Number of packets to generate
            benign_ratio: Ratio of benign packets (0.0 to 1.0)
            packet_size_range: Min and max packet sizes
            
        Returns:
            Tuple of (packet_list, labels_array)
        """
        packets = []
        labels = []
        
        min_size, max_size = packet_size_range
        n_benign = int(n_samples * benign_ratio)
        n_malicious = n_samples - n_benign
        
        # Generate benign packets
        for _ in range(n_benign):
            size = random.randint(min_size, max_size)
            packet = self._generate_benign_packet(size)
            packets.append(packet)
            labels.append(0)  # Benign
            
        # Generate malicious packets
        for _ in range(n_malicious):
            size = random.randint(min_size, max_size)
            packet = self._generate_malicious_packet(size)
            packets.append(packet)
            labels.append(1)  # Malicious
            
        # Shuffle the data
        combined = list(zip(packets, labels))
        random.shuffle(combined)
        packets, labels = zip(*combined)
        
        return list(packets), np.array(labels)
    
    def _generate_benign_packet(self, size: int) -> np.ndarray:
        """Generate a benign packet with normal patterns."""
        # Simulate HTTP/HTTPS traffic patterns
        packet = np.random.randint(0, 256, size, dtype=np.uint8)
        
        # Add some typical HTTP header patterns
        if size > 100:
            # Simulate HTTP headers at the beginning
            packet[:20] = np.random.choice([71, 69, 84, 32, 47], 20)  # "GET /" pattern
            packet[20:40] = np.random.randint(32, 127, 20)  # ASCII printable
            
        return packet
    
    def _generate_malicious_packet(self, size: int) -> np.ndarray:
        """Generate a malicious packet with suspicious patterns."""
        packet_type = random.choice(['dos', 'probe', 'exploit', 'overflow'])
        
        if packet_type == 'dos':
            # DoS attack - repetitive patterns
            pattern = np.random.randint(0, 256, 50, dtype=np.uint8)
            packet = np.tile(pattern, (size // 50) + 1)[:size]
            
        elif packet_type == 'probe':
            # Port scanning - specific sequences
            packet = np.zeros(size, dtype=np.uint8)
            packet[:100] = np.random.randint(200, 256, 100)
            
        elif packet_type == 'exploit':
            # Exploit attempt - shellcode patterns
            packet = np.random.randint(0, 256, size, dtype=np.uint8)
            # Add NOP sled pattern
            if size > 200:
                packet[100:200] = 144  # NOP instruction (0x90)
                
        else:  # overflow
            # Buffer overflow - long sequences of same value
            fill_value = random.choice([0, 65, 255])  # Common overflow values
            packet = np.full(size, fill_value, dtype=np.uint8)
            # Add some variation at the end
            packet[-50:] = np.random.randint(0, 256, 50)
            
        return packet
    
    def generate_mixed_attack_data(self, n_samples: int = 1000) -> Tuple[List[np.ndarray], np.ndarray]:
        """Generate data with multiple attack types."""
        packets = []
        labels = []
        
        attack_types = ['benign', 'dos', 'probe', 'r2l', 'u2r', 'exploit']
        samples_per_type = n_samples // len(attack_types)
        
        for i, attack_type in enumerate(attack_types):
            for _ in range(samples_per_type):
                size = random.randint(256, 1500)
                
                if attack_type == 'benign':
                    packet = self._generate_benign_packet(size)
                    label = 0
                else:
                    packet = self._generate_attack_packet(attack_type, size)
                    label = i  # Each attack type gets its own label
                    
                packets.append(packet)
                labels.append(label)
        
        # Shuffle the data
        combined = list(zip(packets, labels))
        random.shuffle(combined)
        packets, labels = zip(*combined)
        
        return list(packets), np.array(labels)
    
    def _generate_attack_packet(self, attack_type: str, size: int) -> np.ndarray:
        """Generate specific attack packet types."""
        if attack_type == 'dos':
            # DoS - repetitive patterns
            pattern = np.random.randint(0, 256, 50, dtype=np.uint8)
            packet = np.tile(pattern, (size // 50) + 1)[:size]
            
        elif attack_type == 'probe':
            # Probe - specific byte sequences
            packet = np.zeros(size, dtype=np.uint8)
            packet[:100] = np.random.randint(200, 256, 100)
            
        elif attack_type == 'r2l':
            # Remote to local - embedded patterns
            packet = np.random.randint(0, 128, size, dtype=np.uint8)
            if size > 600:
                packet[500:600] = 255  # Suspicious pattern
                
        elif attack_type == 'u2r':
            # User to root - privilege escalation patterns
            packet = np.random.randint(0, 256, size, dtype=np.uint8)
            packet[::10] = 0  # Regular nulls
            
        else:  # exploit
            # Exploit - buffer overflow patterns
            packet = np.full(size, 144, dtype=np.uint8)  # NOP sled
            if size > 500:
                packet[size//2:] = np.random.randint(0, 256, size - size//2)
                
        return packet


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