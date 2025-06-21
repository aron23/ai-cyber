# Quick Start Guide: Vision Transformer Network Traffic Analysis

## 🚀 Getting Started in 5 Minutes

### 1. Clone the Payload-Byte Repository (Development Dataset)
```bash
git clone https://github.com/Yasir-ali-farrukh/Payload-Byte.git
cd Payload-Byte
```

### 2. Set up Your Environment
```bash
# Create virtual environment
python -m venv vit_env
source vit_env/bin/activate  # On Windows: vit_env\Scripts\activate

# Install dependencies
pip install torch torchvision transformers
pip install numpy pandas matplotlib seaborn
pip install google-cloud-storage google-cloud-aiplatform
pip install scikit-learn tqdm jupyter
```

### 3. Configure Google Cloud Access
```bash
# Authenticate with Google Cloud
gcloud auth application-default login
gcloud config set project YOUR_PROJECT_ID

# Set environment variable
export GOOGLE_APPLICATION_CREDENTIALS="path/to/your/credentials.json"
```

### 4. Quick Test: Convert Packet to Image
```python
import numpy as np
import matplotlib.pyplot as plt

def packet_to_image(packet_bytes, image_size=224, patch_size=16):
    """Convert packet bytes to 2D image representation"""
    # Pad packet to fit image size
    total_pixels = image_size * image_size
    if len(packet_bytes) < total_pixels:
        packet_bytes = np.pad(packet_bytes, (0, total_pixels - len(packet_bytes)))
    else:
        packet_bytes = packet_bytes[:total_pixels]
    
    # Reshape to 2D image
    image = packet_bytes.reshape(image_size, image_size)
    
    # Create patches
    n_patches = image_size // patch_size
    patches = image.reshape(n_patches, patch_size, n_patches, patch_size)
    patches = patches.transpose(0, 2, 1, 3).reshape(-1, patch_size, patch_size)
    
    return image, patches

# Example usage
sample_packet = np.random.randint(0, 256, size=1024, dtype=np.uint8)
image, patches = packet_to_image(sample_packet)

# Visualize
plt.figure(figsize=(10, 5))
plt.subplot(1, 2, 1)
plt.imshow(image, cmap='gray')
plt.title('Packet as Image')
plt.subplot(1, 2, 2)
plt.imshow(patches[0], cmap='gray')
plt.title('First Patch')
plt.show()
```

### 5. Load Payload-Byte Data
```python
import pandas as pd
from pathlib import Path

# Load the processed data from Payload-Byte
data_path = Path("Payload-Byte/Data/UNSW-NB15")
if data_path.exists():
    # Load CSV with labels
    df = pd.read_csv(data_path / "processed_data.csv")
    print(f"Loaded {len(df)} samples")
    print(f"Benign: {sum(df['label'] == 0)}, Malicious: {sum(df['label'] == 1)}")
```

## 📊 Key Concepts Illustrated

### Vision Transformer Architecture
```python
import torch
import torch.nn as nn

class SimpleViT(nn.Module):
    def __init__(self, image_size=224, patch_size=16, num_classes=2, dim=768):
        super().__init__()
        num_patches = (image_size // patch_size) ** 2
        patch_dim = 1 * patch_size * patch_size  # 1 channel (grayscale)
        
        self.patch_embed = nn.Linear(patch_dim, dim)
        self.pos_embed = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.transformer = nn.TransformerEncoder(
            nn.TransformerEncoderLayer(d_model=dim, nhead=12),
            num_layers=12
        )
        self.mlp_head = nn.Linear(dim, num_classes)
        
    def forward(self, x):
        # x shape: (batch, channels, height, width)
        # ... implementation ...
        return self.mlp_head(x)

# Create model
model = SimpleViT()
print(f"Model parameters: {sum(p.numel() for p in model.parameters()):,}")
```

### Few-Shot Learning Example
```python
def create_support_query_sets(data, n_way=2, k_shot=5, query_size=15):
    """Create episode for few-shot learning"""
    support_set = []
    query_set = []
    
    for class_idx in range(n_way):
        class_data = data[data['label'] == class_idx]
        indices = np.random.choice(len(class_data), k_shot + query_size, replace=False)
        
        support_set.extend(class_data.iloc[indices[:k_shot]].values)
        query_set.extend(class_data.iloc[indices[k_shot:]].values)
    
    return np.array(support_set), np.array(query_set)
```

## 🎯 First Week Goals

### Day 1-2: Environment Setup
- [ ] Set up GCP and Vertex AI access
- [ ] Clone repositories and install dependencies
- [ ] Run notebook 01 and 02 successfully
- [ ] Visualize first packet images

### Day 3-4: Data Pipeline
- [ ] Load Payload-Byte dataset
- [ ] Implement packet-to-image conversion
- [ ] Create data loaders for training
- [ ] Generate image statistics report

### Day 5: Baseline Model
- [ ] Implement simple ViT architecture
- [ ] Train on small subset (1000 samples)
- [ ] Achieve >80% accuracy on binary classification
- [ ] Save first model checkpoint

## 💡 Tips for Success

1. **Start Small**: Use Payload-Byte dataset first (smaller, faster iteration)
2. **Visualize Everything**: Always plot your packet images to understand the data
3. **Use GPU**: Even for development, GPU speeds up experimentation 10x
4. **Version Control**: Commit notebooks and code daily
5. **Document**: Write clear markdown in notebooks explaining each step

## 🔧 Troubleshooting

### Common Issues and Solutions

**Issue**: "ModuleNotFoundError: No module named 'torch'"
```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
```

**Issue**: "Google Cloud authentication failed"
```bash
gcloud auth application-default login
export GOOGLE_APPLICATION_CREDENTIALS="/path/to/key.json"
```

**Issue**: "CUDA out of memory"
```python
# Reduce batch size
batch_size = 16  # Instead of 32
# Use gradient accumulation
accumulation_steps = 4
```

## 📚 Essential Reading

1. **Vision Transformer Paper**: [An Image is Worth 16x16 Words](https://arxiv.org/abs/2010.11929)
2. **Payload-Byte Paper**: [Network Traffic Classification Tool](https://github.com/Yasir-ali-farrukh/Payload-Byte)
3. **Few-Shot Learning**: [Prototypical Networks](https://arxiv.org/abs/1703.05175)

## 🤝 Team Collaboration

### Daily Standup Template
```markdown
**Yesterday**: Completed [task]
**Today**: Working on [task]
**Blockers**: [Any issues]
**Help Needed**: [Specific assistance]
```

### Notebook Naming Convention
- `{number}_{name}_{author_initials}_{date}.ipynb`
- Example: `04_image_encoding_JD_20240115.ipynb`

## 🚦 Ready to Start?

1. Join the project Slack channel: #vit-network-traffic
2. Attend kickoff meeting: [Calendar invite]
3. Pick your first notebook from Phase 1
4. Start coding! 🎉

---

**Questions?** Contact the project lead or post in Slack. 