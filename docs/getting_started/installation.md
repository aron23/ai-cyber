# Installation Guide

This guide will help you set up the Vision Transformer Network Traffic Analysis project on your system.

## System Requirements

### Minimum Requirements
- **OS**: Ubuntu 20.04+ / macOS 12+ / Windows 10+
- **Python**: 3.8 or higher
- **RAM**: 16GB minimum
- **Storage**: 50GB available space
- **GPU**: NVIDIA GPU with 8GB+ VRAM (optional but recommended)

### Recommended Requirements
- **OS**: Ubuntu 22.04 LTS
- **Python**: 3.9 or 3.10
- **RAM**: 32GB or more
- **Storage**: 100GB SSD
- **GPU**: NVIDIA RTX 3080 or better with 12GB+ VRAM
- **CUDA**: 11.7 or higher

## Prerequisites

### 1. Python Installation

Verify Python installation:
```bash
python --version
# Should output: Python 3.8.x or higher
```

If Python is not installed, download from [python.org](https://www.python.org/downloads/).

### 2. CUDA Setup (Optional)

For GPU acceleration, install CUDA:

```bash
# Check if CUDA is installed
nvidia-smi

# Install CUDA (Ubuntu example)
wget https://developer.download.nvidia.com/compute/cuda/repos/ubuntu2204/x86_64/cuda-keyring_1.0-1_all.deb
sudo dpkg -i cuda-keyring_1.0-1_all.deb
sudo apt-get update
sudo apt-get -y install cuda
```

### 3. Git Installation

```bash
# Check if Git is installed
git --version

# Install Git if needed
sudo apt-get install git  # Ubuntu/Debian
brew install git          # macOS
```

## Installation Steps

### 1. Clone the Repository

```bash
git clone https://github.com/your-org/vit-network-traffic.git
cd vit-network-traffic
```

### 2. Create Virtual Environment

We recommend using a virtual environment to avoid dependency conflicts:

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Linux/macOS:
source venv/bin/activate

# On Windows:
venv\Scripts\activate

# Your prompt should now show (venv)
```

### 3. Upgrade pip

```bash
python -m pip install --upgrade pip
```

### 4. Install Dependencies

#### Basic Installation
```bash
pip install -r requirements.txt
```

#### Development Installation
```bash
pip install -r requirements-dev.txt
```

#### Documentation Tools
```bash
pip install -r docs/requirements-docs.txt
```

### 5. Install PyTorch

Install PyTorch based on your system configuration:

#### CPU Only
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

#### CUDA 11.7
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
```

#### CUDA 11.8
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

Visit [PyTorch's official site](https://pytorch.org/get-started/locally/) for other configurations.

### 6. Verify Installation

Run the verification script:

```bash
python scripts/verify_installation.py
```

Expected output:
```
✅ Python version: 3.9.16
✅ PyTorch version: 2.0.1
✅ CUDA available: True
✅ CUDA version: 11.7
✅ GPU detected: NVIDIA GeForce RTX 3080
✅ All required packages installed
✅ Installation successful!
```

## Google Cloud Setup

### 1. Install Google Cloud SDK

```bash
# Download and install
curl -O https://dl.google.com/dl/cloudsdk/channels/rapid/downloads/google-cloud-sdk-latest-linux-x86_64.tar.gz
tar -xf google-cloud-sdk-latest-linux-x86_64.tar.gz
./google-cloud-sdk/install.sh

# Initialize
gcloud init
```

### 2. Authenticate

```bash
gcloud auth login
gcloud auth application-default login
```

### 3. Set Project

```bash
gcloud config set project YOUR_PROJECT_ID
```

## Dataset Setup

### 1. Download Sample Dataset

```bash
# Download Payload-Byte dataset
python scripts/download_data.py --dataset payload-byte --output data/raw/
```

### 2. Access Cloud Datasets

```bash
# Configure GCS access
export GCS_BUCKET="gs://your-bucket-name"

# List available datasets
gsutil ls $GCS_BUCKET/datasets/
```

## Environment Variables

Create a `.env` file in the project root:

```bash
# Copy template
cp .env.template .env

# Edit with your settings
nano .env
```

Example `.env` contents:
```bash
# Google Cloud
GCP_PROJECT_ID=your-project-id
GCS_BUCKET=your-bucket-name

# Model Configuration
MODEL_PATH=models/
CHECKPOINT_DIR=checkpoints/

# Logging
LOG_LEVEL=INFO
MLFLOW_TRACKING_URI=http://localhost:5000

# GPU Settings
CUDA_VISIBLE_DEVICES=0
TF_FORCE_GPU_ALLOW_GROWTH=true
```

## Troubleshooting

### Common Issues

#### 1. CUDA Out of Memory
```bash
# Reduce batch size in config
export BATCH_SIZE=16

# Or clear GPU memory
python -c "import torch; torch.cuda.empty_cache()"
```

#### 2. ImportError: No module named 'torch'
```bash
# Reinstall PyTorch
pip uninstall torch torchvision
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu117
```

#### 3. Permission Denied
```bash
# Fix permissions
sudo chown -R $USER:$USER .
chmod +x scripts/*.sh
```

### Getting Help

If you encounter issues:

1. Check [Troubleshooting Guide](../troubleshooting.md)
2. Search [GitHub Issues](https://github.com/your-org/vit-network-traffic/issues)
3. Ask in [Discussions](https://github.com/your-org/vit-network-traffic/discussions)

## Next Steps

✅ Installation complete! Now you can:

1. Follow the [Quick Start Guide](quick_start.md) to train your first model
2. Explore [Jupyter Notebooks](../notebooks/01_data_exploration_basic.ipynb)
3. Read about [Architecture](../architecture/overview.md)

---

**Note**: Keep your environment activated whenever working with the project:
```bash
source venv/bin/activate  # Linux/macOS
venv\Scripts\activate     # Windows
```