#!/usr/bin/env python3
"""
Analyze the difference between local files and remote formats
"""

import struct

def analyze_pcap_format(pcap_data, name):
    """Analyze PCAP format details"""
    print(f"\n=== {name} ===")
    print(f"File size: {len(pcap_data):,} bytes")
    
    if len(pcap_data) < 24:
        print("File too small")
        return
    
    # Check magic and endianness
    magic_bytes = pcap_data[0:4]
    print(f"Magic: {' '.join(f'{b:02x}' for b in magic_bytes)}")
    
    if magic_bytes == b'\xd4\xc3\xb2\xa1':
        endian = '<'
        print("Format: Little-endian")
    elif magic_bytes == b'\xa1\xb2\xc3\xd4':
        endian = '>'
        print("Format: Big-endian")
    else:
        print("Unknown format")
        return
    
    # Read global header
    version_major, version_minor, thiszone, sigfigs, snaplen, network = struct.unpack(
        f'{endian}HHIIII', pcap_data[4:24])
    
    print(f"Version: {version_major}.{version_minor}")
    print(f"Network: {network} ({'Linux cooked' if network == 113 else 'Ethernet' if network == 1 else 'Other'})")
    print(f"Snaplen: {snaplen}")
    
    # Analyze first 3 packets
    offset = 24
    for i in range(3):
        if offset + 16 > len(pcap_data):
            break
            
        ts_sec, ts_usec, incl_len, orig_len = struct.unpack(f'{endian}IIII', pcap_data[offset:offset+16])
        print(f"Packet {i}: incl_len={incl_len}, orig_len={orig_len}, ts={ts_sec}.{ts_usec:06d}")
        
        if offset + 16 + incl_len > len(pcap_data):
            print(f"  ERROR: Packet extends beyond file")
            break
            
        # Look at packet data
        packet_data = pcap_data[offset+16:offset+16+incl_len]
        hex_preview = ' '.join(f'{b:02x}' for b in packet_data[:16])
        print(f"  First 16 bytes: {hex_preview}")
        
        offset += 16 + incl_len

# Test local file (representative of one bucket folder)
print("Analyzing local PCAP file (representative of one bucket folder)...")
with open('/home/ubuntu/Cyber_AI/ai-cyber/raw/9.pcap', 'rb') as f:
    local_data = f.read(100000)  # First 100KB

analyze_pcap_format(local_data, "Local /raw/9.pcap")

# Test both remote formats by downloading small samples
import subprocess

print("\nDownloading samples from both remote directories...")

try:
    # Sample from 17-2-2015
    cmd1 = ['gsutil', 'cat', 'gs://ai-cyber/datasets/unsw-nb15/pcap/pcaps 17-2-2015/1.pcap']
    result1 = subprocess.run(cmd1, capture_output=True, timeout=30, input=b'')
    if result1.returncode == 0:
        remote1_data = result1.stdout[:100000]  # First 100KB
        analyze_pcap_format(remote1_data, "Remote pcaps 17-2-2015/1.pcap")
    else:
        print(f"Failed to download 17-2-2015: {result1.stderr.decode()}")

except subprocess.TimeoutExpired:
    print("17-2-2015 download timed out")
except Exception as e:
    print(f"Error with 17-2-2015: {e}")

try:
    # Sample from 22-1-2015  
    cmd2 = ['gsutil', 'cat', 'gs://ai-cyber/datasets/unsw-nb15/pcap/pcaps 22-1-2015/1.pcap']
    result2 = subprocess.run(cmd2, capture_output=True, timeout=30, input=b'')
    if result2.returncode == 0:
        remote2_data = result2.stdout[:100000]  # First 100KB
        analyze_pcap_format(remote2_data, "Remote pcaps 22-1-2015/1.pcap")
    else:
        print(f"Failed to download 22-1-2015: {result2.stderr.decode()}")

except subprocess.TimeoutExpired:
    print("22-1-2015 download timed out")
except Exception as e:
    print(f"Error with 22-1-2015: {e}")

print("\nConclusion: If the local files work but 22-1-2015 doesn't, there's likely a format difference.")