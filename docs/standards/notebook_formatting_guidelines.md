# Jupyter Notebook Formatting Guidelines

## Overview
This guide ensures consistency across all educational notebooks in the Vision Transformer Network Traffic Analysis project. Following these guidelines will create a cohesive learning experience.

## Notebook Naming Convention

### File Naming Structure
```
{number}_{topic}_{level}.ipynb

Examples:
01_data_exploration_basic.ipynb
07_vit_implementation_intermediate.ipynb
13_few_shot_deployment_advanced.ipynb
```

### Numbering System
- 01-03: Data and preprocessing
- 04-06: Image encoding techniques
- 07-09: Vision Transformer implementation
- 10-12: Few-shot learning
- 13-14: Optimization and deployment
- 15: Complete tutorial

## Notebook Structure Template

### 1. Header Cell (Markdown)
```markdown
# Notebook Title: Clear and Descriptive

**Project**: Vision Transformer for Network Traffic Analysis  
**Topic**: Specific topic covered  
**Level**: Basic | Intermediate | Advanced  
**Duration**: Estimated completion time  
**GPU Required**: Yes/No  

---

## 🎯 Learning Objectives

By the end of this notebook, you will be able to:
1. First specific objective
2. Second specific objective
3. Third specific objective

## 📋 Prerequisites

- List specific knowledge required
- Link to prerequisite notebooks
- Required installations

## 📚 Table of Contents

1. [Introduction](#introduction)
2. [Setup and Imports](#setup)
3. [Main Content Section 1](#section1)
4. [Main Content Section 2](#section2)
5. [Exercises](#exercises)
6. [Summary](#summary)
7. [Further Reading](#further-reading)
```

### 2. Setup Cell (Markdown + Code)
```python
# Markdown cell
"""
## 🔧 Setup and Imports

Let's start by importing necessary libraries and setting up our environment.
"""

# Code cell
# Standard library imports
import os
import sys
from pathlib import Path

# Data manipulation
import numpy as np
import pandas as pd

# Visualization
import matplotlib.pyplot as plt
import seaborn as sns

# Deep learning
import torch
import torch.nn as nn
from torchvision import transforms

# Project specific
sys.path.append('../src')
from packet_encoder import PacketImageEncoder
from vit_model import VisionTransformerClassifier

# Configuration
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

# Set random seeds for reproducibility
np.random.seed(42)
torch.manual_seed(42)

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
```

### 3. Configuration Cell
```python
# Markdown cell
"""
## ⚙️ Configuration

Define global parameters used throughout the notebook.
"""

# Code cell
# Model parameters
IMAGE_SIZE = 224
PATCH_SIZE = 16
NUM_CLASSES = 2
BATCH_SIZE = 32

# Training parameters
LEARNING_RATE = 1e-4
NUM_EPOCHS = 10
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Data parameters
MAX_PACKET_SIZE = 1500  # Standard MTU
TRAIN_SPLIT = 0.8

# Visualization parameters
FIGSIZE = (12, 8)
DPI = 100

# Display configuration
config_df = pd.DataFrame({
    'Parameter': ['Image Size', 'Patch Size', 'Batch Size', 'Learning Rate', 'Device'],
    'Value': [IMAGE_SIZE, PATCH_SIZE, BATCH_SIZE, LEARNING_RATE, DEVICE]
})
display(config_df)
```

## Content Organization

### Section Structure
Each major section should follow this pattern:

```python
# Markdown cell - Section Introduction
"""
## 📊 Section Title

Brief introduction to what this section covers and why it's important.

### Key Concepts
- Concept 1: Brief explanation
- Concept 2: Brief explanation
"""

# Markdown cell - Subsection
"""
### Implementation Details

Explain the approach before showing code.
"""

# Code cell - Implementation
# Clear implementation with comments

# Markdown cell - Visualization/Results
"""
### Visualizing the Results

Let's examine what we've created:
"""

# Code cell - Visualization
# Create informative visualizations

# Markdown cell - Interpretation
"""
### Understanding the Output

The visualization shows:
- Point 1: Explanation
- Point 2: Explanation

💡 **Key Insight**: Important takeaway
"""
```

## Code Cell Best Practices

### 1. Cell Granularity
- One concept per cell
- Keep cells focused and digestible
- Maximum 20-30 lines per cell

### 2. Progressive Complexity
```python
# Basic implementation
def encode_packet_simple(packet_bytes):
    """Simple sequential encoding."""
    return np.array(list(packet_bytes))

# Intermediate implementation  
def encode_packet_padded(packet_bytes, target_size=1024):
    """Encoding with padding."""
    encoded = np.array(list(packet_bytes))
    if len(encoded) < target_size:
        encoded = np.pad(encoded, (0, target_size - len(encoded)))
    return encoded[:target_size]

# Advanced implementation
def encode_packet_advanced(packet_bytes, method='hilbert', image_size=224):
    """Full-featured encoding with multiple methods."""
    encoder = PacketImageEncoder(method=method, image_size=image_size)
    return encoder.encode(packet_bytes)
```

### 3. Error Handling and Validation
```python
# Always include input validation in educational notebooks
def process_packet(packet_data):
    """Process packet with proper validation."""
    # Input validation
    if not isinstance(packet_data, (bytes, bytearray)):
        raise TypeError(f"Expected bytes, got {type(packet_data)}")
    
    if len(packet_data) == 0:
        raise ValueError("Packet cannot be empty")
        
    if len(packet_data) > MAX_PACKET_SIZE:
        print(f"⚠️ Warning: Packet size {len(packet_data)} exceeds MTU")
    
    # Process packet
    return encode_packet_advanced(packet_data)
```

## Visualization Standards

### 1. Consistent Styling
```python
def setup_plot_style():
    """Set consistent plot styling."""
    plt.rcParams.update({
        'figure.figsize': FIGSIZE,
        'figure.dpi': DPI,
        'font.size': 12,
        'axes.labelsize': 14,
        'axes.titlesize': 16,
        'xtick.labelsize': 12,
        'ytick.labelsize': 12,
        'legend.fontsize': 12,
        'figure.titlesize': 18
    })

setup_plot_style()
```

### 2. Informative Visualizations
```python
def plot_encoding_comparison(packet_bytes):
    """Compare different encoding methods visually."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    methods = ['sequential', 'hilbert', 'spiral']
    titles = ['Sequential Encoding', 'Hilbert Curve', 'Spiral Pattern']
    
    for ax, method, title in zip(axes, methods, titles):
        image = encode_packet_advanced(packet_bytes, method=method)
        im = ax.imshow(image, cmap='viridis', aspect='auto')
        ax.set_title(title)
        ax.set_xlabel('Pixel X')
        ax.set_ylabel('Pixel Y')
        plt.colorbar(im, ax=ax, label='Byte Value')
    
    plt.suptitle('Packet-to-Image Encoding Methods', fontsize=16)
    plt.tight_layout()
    return fig
```

### 3. Interactive Elements
```python
# Use widgets for interactive exploration
from ipywidgets import interact, IntSlider, Dropdown

@interact(
    packet_size=IntSlider(min=64, max=1500, step=64, value=512),
    encoding_method=Dropdown(options=['sequential', 'hilbert', 'spiral']),
)
def explore_encoding(packet_size, encoding_method):
    """Interactive encoding exploration."""
    # Generate sample packet
    packet = bytes(np.random.randint(0, 256, packet_size))
    
    # Encode and visualize
    image = encode_packet_advanced(packet, method=encoding_method)
    
    plt.figure(figsize=(8, 8))
    plt.imshow(image, cmap='viridis')
    plt.title(f'{encoding_method.capitalize()} Encoding ({packet_size} bytes)')
    plt.colorbar(label='Byte Value')
    plt.show()
    
    # Show statistics
    print(f"Image shape: {image.shape}")
    print(f"Unique values: {np.unique(image).size}")
    print(f"Mean byte value: {image.mean():.2f}")
```

## Exercise Integration

### Exercise Structure
```python
# Markdown cell
"""
## 🏋️ Exercises

### Exercise 1: Basic Encoding
Implement a function that encodes packet bytes using a zigzag pattern.

**Requirements:**
- Input: packet bytes
- Output: 2D numpy array
- The pattern should start from top-left corner

**Hint**: Think about how to alternate between horizontal directions.
"""

# Code cell - Exercise
# TODO: Implement zigzag encoding
def encode_zigzag(packet_bytes, image_size=64):
    """
    Encode packet bytes in a zigzag pattern.
    
    Args:
        packet_bytes: Input packet as bytes
        image_size: Size of the output image
        
    Returns:
        2D numpy array
    """
    # Your implementation here
    pass

# Test your implementation
test_packet = b"Hello, World!"
result = encode_zigzag(test_packet)
print(f"Result shape: {result.shape}")

# Solution cell (hidden by default)
"""
### 💡 Solution
<details>
<summary>Click to reveal solution</summary>

```python
def encode_zigzag(packet_bytes, image_size=64):
    image = np.zeros((image_size, image_size), dtype=np.uint8)
    packet_array = np.array(list(packet_bytes))
    
    idx = 0
    for row in range(image_size):
        if row % 2 == 0:
            # Left to right
            for col in range(image_size):
                if idx < len(packet_array):
                    image[row, col] = packet_array[idx]
                    idx += 1
        else:
            # Right to left
            for col in range(image_size - 1, -1, -1):
                if idx < len(packet_array):
                    image[row, col] = packet_array[idx]
                    idx += 1
    
    return image
```
</details>
"""
```

## Progress Tracking

### Checkpoints
```python
# Markdown cell
"""
## ✅ Progress Check

Let's verify your understanding before moving forward:

1. ✓ Environment setup complete
2. ✓ Data loaded successfully
3. ✓ Basic encoding implemented
4. ⏳ Advanced encoding techniques
5. ⏳ Model implementation

If any checks failed, please review the relevant sections.
"""

# Code cell - Automated checks
def check_progress():
    """Verify notebook progress."""
    checks = {
        'PyTorch installed': 'torch' in globals(),
        'Data loaded': 'train_data' in globals(),
        'Encoder defined': 'PacketImageEncoder' in globals(),
        'Model created': 'model' in globals()
    }
    
    for check, passed in checks.items():
        status = "✅" if passed else "❌"
        print(f"{status} {check}")
    
    return all(checks.values())

if check_progress():
    print("\n🎉 All checks passed! Ready to continue.")
else:
    print("\n⚠️ Some checks failed. Please review previous sections.")
```

## Performance Considerations

### Timing Code Execution
```python
# Use consistent timing methods
import time
from contextlib import contextmanager

@contextmanager
def timer(description="Operation"):
    """Time code execution."""
    start = time.perf_counter()
    yield
    elapsed = time.perf_counter() - start
    print(f"⏱️ {description} took {elapsed:.3f} seconds")

# Example usage
with timer("Encoding 1000 packets"):
    for _ in range(1000):
        encode_packet_advanced(sample_packet)
```

### Memory Usage
```python
# Monitor memory usage in notebooks
import psutil
import os

def print_memory_usage(stage="Current"):
    """Print current memory usage."""
    process = psutil.Process(os.getpid())
    mem_info = process.memory_info()
    print(f"💾 {stage} memory usage: {mem_info.rss / 1024 / 1024:.2f} MB")

print_memory_usage("Initial")
# ... code execution ...
print_memory_usage("After processing")
```

## Summary Section Template

```python
# Markdown cell
"""
## 📝 Summary

### What We Learned
1. **Concept 1**: Key takeaway
2. **Concept 2**: Key takeaway
3. **Concept 3**: Key takeaway

### Key Code Snippets
```python
# Most important function from this notebook
encoder = PacketImageEncoder(method='hilbert')
image = encoder.encode(packet_bytes)
```

### Performance Insights
- Sequential encoding: ~0.001s per packet
- Hilbert encoding: ~0.003s per packet
- Model inference: ~0.010s per packet

### Common Pitfalls
⚠️ **Watch out for**:
- Packet size variations
- Byte value normalization
- Memory usage with large batches
"""
```

## References and Resources

```python
# Markdown cell
"""
## 📚 Further Reading

### Papers
1. [Vision Transformer (ViT)](https://arxiv.org/abs/2010.11929) - Original ViT paper
2. [Attention Is All You Need](https://arxiv.org/abs/1706.03762) - Transformer architecture

### Related Notebooks
- [02_data_preprocessing.ipynb](02_data_preprocessing.ipynb) - Data preparation
- [08_attention_visualization.ipynb](08_attention_visualization.ipynb) - Understanding attention

### External Resources
- [PyTorch ViT Tutorial](https://pytorch.org/vision/stable/models/vit.html)
- [Hugging Face Transformers](https://huggingface.co/docs/transformers/)

### Next Steps
Ready to continue? Move on to [08_vit_training.ipynb](08_vit_training.ipynb)
"""
```

## Quality Checklist

Before finalizing a notebook:
- [ ] Clear learning objectives stated
- [ ] All code cells execute without errors
- [ ] Visualizations are informative and labeled
- [ ] Exercises included with solutions
- [ ] Memory and performance considered
- [ ] References and next steps provided
- [ ] Markdown formatting consistent
- [ ] Code follows style guidelines
- [ ] Runtime estimate accurate
- [ ] Works in clean environment