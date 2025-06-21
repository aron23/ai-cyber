#!/bin/bash

# Script to set up GPU support for ML development
set -e

echo "🚀 Setting up GPU support..."

# Colors for output
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
RED='\033[0;31m'
NC='\033[0m' # No Color

# Check if running with sudo
if [ "$EUID" -ne 0 ]; then 
    echo -e "${RED}Please run with sudo${NC}"
    exit 1
fi

# Detect GPU
echo -e "${YELLOW}Detecting GPU...${NC}"
if ! command -v nvidia-smi &> /dev/null; then
    echo -e "${RED}nvidia-smi not found. Installing NVIDIA drivers...${NC}"
    
    # Add NVIDIA package repositories
    wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2004/x86_64/cuda-keyring_1.0-1_all.deb
    dpkg -i cuda-keyring_1.0-1_all.deb
    apt-get update
    
    # Install CUDA toolkit
    apt-get install -y cuda-toolkit-11-8
    
    # Clean up
    rm cuda-keyring_1.0-1_all.deb
fi

# Display GPU info
if command -v nvidia-smi &> /dev/null; then
    echo -e "${GREEN}GPU detected:${NC}"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
else
    echo -e "${RED}No GPU detected or drivers not properly installed${NC}"
    exit 1
fi

# Install NVIDIA Container Toolkit for Docker
echo -e "${YELLOW}Installing NVIDIA Container Toolkit...${NC}"
distribution=$(. /etc/os-release;echo $ID$VERSION_ID)
curl -s -L https://nvidia.github.io/nvidia-docker/gpgkey | apt-key add -
curl -s -L https://nvidia.github.io/nvidia-docker/$distribution/nvidia-docker.list | tee /etc/apt/sources.list.d/nvidia-docker.list

apt-get update
apt-get install -y nvidia-container-toolkit
systemctl restart docker

# Update docker-compose.yml to include GPU support
echo -e "${YELLOW}Updating docker-compose.yml for GPU support...${NC}"
cat > docker-compose.gpu.yml << 'EOF'
version: '3.8'

services:
  dev-env:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility

  jupyter:
    deploy:
      resources:
        reservations:
          devices:
            - driver: nvidia
              count: all
              capabilities: [gpu]
    environment:
      - NVIDIA_VISIBLE_DEVICES=all
      - NVIDIA_DRIVER_CAPABILITIES=compute,utility
EOF

# Create GPU monitoring script
echo -e "${YELLOW}Creating GPU monitoring script...${NC}"
cat > scripts/monitor_gpu.sh << 'EOF'
#!/bin/bash

# GPU monitoring script
echo "GPU Monitoring Dashboard"
echo "========================"

while true; do
    clear
    echo "GPU Status - $(date)"
    echo "========================"
    
    # GPU utilization
    nvidia-smi --query-gpu=index,name,utilization.gpu,memory.used,memory.total,temperature.gpu --format=csv,noheader,nounits | \
    awk -F', ' '{printf "GPU %s: %s\n  Utilization: %s%%\n  Memory: %s/%s MB\n  Temperature: %s°C\n\n", $1, $2, $3, $4, $5, $6}'
    
    # Process list
    echo "GPU Processes:"
    echo "--------------"
    nvidia-smi --query-compute-apps=pid,name,used_memory --format=csv,noheader,nounits | \
    awk -F', ' '{printf "PID: %s | %s | Memory: %s MB\n", $1, $2, $3}'
    
    sleep 2
done
EOF

chmod +x scripts/monitor_gpu.sh

# Create GPU test script
echo -e "${YELLOW}Creating GPU test script...${NC}"
cat > scripts/test_gpu.py << 'EOF'
#!/usr/bin/env python3
"""Test GPU availability for ML frameworks."""

import sys

def test_torch_gpu():
    """Test PyTorch GPU availability."""
    try:
        import torch
        print("PyTorch GPU Test:")
        print(f"  PyTorch version: {torch.__version__}")
        print(f"  CUDA available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"  CUDA version: {torch.version.cuda}")
            print(f"  Number of GPUs: {torch.cuda.device_count()}")
            for i in range(torch.cuda.device_count()):
                print(f"  GPU {i}: {torch.cuda.get_device_name(i)}")
                print(f"    Memory: {torch.cuda.get_device_properties(i).total_memory / 1024**3:.2f} GB")
        print()
        return torch.cuda.is_available()
    except ImportError:
        print("PyTorch not installed")
        return False

def test_tensorflow_gpu():
    """Test TensorFlow GPU availability."""
    try:
        import tensorflow as tf
        print("TensorFlow GPU Test:")
        print(f"  TensorFlow version: {tf.__version__}")
        gpus = tf.config.list_physical_devices('GPU')
        print(f"  Number of GPUs: {len(gpus)}")
        for i, gpu in enumerate(gpus):
            print(f"  GPU {i}: {gpu.name}")
        print()
        return len(gpus) > 0
    except ImportError:
        print("TensorFlow not installed")
        return False

def test_gpu_compute():
    """Run simple GPU computation test."""
    torch_available = test_torch_gpu()
    tf_available = test_tensorflow_gpu()
    
    if torch_available:
        import torch
        print("Running PyTorch GPU computation test...")
        x = torch.rand(1000, 1000).cuda()
        y = torch.rand(1000, 1000).cuda()
        z = torch.matmul(x, y)
        print(f"  Result shape: {z.shape}")
        print("  ✓ PyTorch GPU computation successful")
        print()
    
    if tf_available:
        import tensorflow as tf
        print("Running TensorFlow GPU computation test...")
        with tf.device('/GPU:0'):
            x = tf.random.normal([1000, 1000])
            y = tf.random.normal([1000, 1000])
            z = tf.matmul(x, y)
        print(f"  Result shape: {z.shape}")
        print("  ✓ TensorFlow GPU computation successful")
        print()

if __name__ == "__main__":
    print("=" * 50)
    print("ML GPU Availability Test")
    print("=" * 50)
    print()
    
    test_gpu_compute()
    
    print("Test completed!")
EOF

chmod +x scripts/test_gpu.py

# Install GPU monitoring in Prometheus
echo -e "${YELLOW}Setting up GPU metrics for Prometheus...${NC}"
cat > monitoring/prometheus-gpu.yml << 'EOF'
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'nvidia-gpu'
    static_configs:
      - targets: ['localhost:9835']
EOF

echo -e "${GREEN}✅ GPU setup complete!${NC}"
echo ""
echo "To use GPU-enabled containers, run:"
echo "  docker-compose -f docker-compose.yml -f docker-compose.gpu.yml up -d"
echo ""
echo "To monitor GPU usage:"
echo "  ./scripts/monitor_gpu.sh"
echo ""
echo "To test GPU availability:"
echo "  docker-compose exec dev-env python scripts/test_gpu.py"