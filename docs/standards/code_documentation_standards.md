# Code Documentation Standards

## Overview
This guide defines code documentation standards for the Vision Transformer Network Traffic Analysis project, ensuring consistency across all Python modules, notebooks, and scripts.

## Python Documentation Standards

### Module-Level Documentation
Every Python file must begin with a module docstring:

```python
"""
Module: packet_image_encoder.py

This module provides utilities for converting network packet bytes into 
2D image representations suitable for Vision Transformer processing.

The module supports multiple encoding strategies:
- Sequential: Linear byte arrangement
- Hilbert: Space-filling curve for better locality
- Spiral: Radial arrangement from center outward

Example:
    from packet_image_encoder import PacketImageEncoder
    
    encoder = PacketImageEncoder(image_size=224, method='hilbert')
    image = encoder.encode(packet_bytes)
    
Attributes:
    DEFAULT_IMAGE_SIZE (int): Default image dimension (224x224)
    SUPPORTED_METHODS (list): Available encoding methods
    
Dependencies:
    - numpy >= 1.21.0
    - hilbertcurve >= 1.0.5
"""
```

### Class Documentation
Use comprehensive docstrings for all classes:

```python
class VisionTransformerClassifier:
    """
    Vision Transformer for network packet classification.
    
    This class implements a Vision Transformer architecture specifically
    designed for malware detection in network traffic. It processes packet
    images through self-attention mechanisms to capture both local and
    global patterns.
    
    Args:
        num_classes (int): Number of output classes (default: 2)
        image_size (int): Input image dimensions (default: 224)
        patch_size (int): Size of image patches (default: 16)
        dim (int): Embedding dimension (default: 768)
        depth (int): Number of transformer blocks (default: 12)
        heads (int): Number of attention heads (default: 12)
        mlp_dim (int): Dimension of MLP layer (default: 3072)
        dropout (float): Dropout rate (default: 0.1)
        
    Attributes:
        patch_embed (nn.Module): Patch embedding layer
        transformer (nn.Module): Transformer encoder stack
        classifier (nn.Module): Final classification head
        
    Example:
        >>> model = VisionTransformerClassifier(num_classes=2)
        >>> images = torch.randn(32, 3, 224, 224)
        >>> predictions = model(images)
        >>> print(predictions.shape)
        torch.Size([32, 2])
        
    Note:
        The model expects input images normalized to [0, 1] range.
        For grayscale packet images, repeat the channel dimension.
    """
    
    def __init__(self, num_classes=2, image_size=224, patch_size=16, 
                 dim=768, depth=12, heads=12, mlp_dim=3072, dropout=0.1):
        super().__init__()
        # Implementation here
```

### Function Documentation
Document all functions with detailed docstrings:

```python
def encode_packet_hilbert(packet_bytes: bytes, image_size: int = 224) -> np.ndarray:
    """
    Encode packet bytes into 2D image using Hilbert curve mapping.
    
    The Hilbert curve preserves locality, meaning bytes that are close
    in the original packet remain close in the 2D representation. This
    can help the model identify contiguous patterns in malware payloads.
    
    Args:
        packet_bytes (bytes): Raw packet payload bytes
        image_size (int, optional): Target image dimension. Must be power of 2.
            Defaults to 224.
            
    Returns:
        np.ndarray: 2D grayscale image of shape (image_size, image_size)
        
    Raises:
        ValueError: If image_size is not a power of 2
        TypeError: If packet_bytes is not bytes type
        
    Example:
        >>> packet = b'\\x00\\x01\\x02\\x03\\x04\\x05'
        >>> image = encode_packet_hilbert(packet, image_size=64)
        >>> assert image.shape == (64, 64)
        
    Performance:
        Time complexity: O(n) where n is image_size^2
        Space complexity: O(n)
        
    See Also:
        encode_packet_sequential: Linear encoding method
        encode_packet_spiral: Spiral encoding method
    """
    if not isinstance(packet_bytes, bytes):
        raise TypeError(f"Expected bytes, got {type(packet_bytes)}")
        
    if not (image_size & (image_size - 1) == 0):
        raise ValueError(f"Image size must be power of 2, got {image_size}")
        
    # Implementation here
```

### Inline Comments
Use inline comments judiciously:

```python
# Good: Explains complex logic
attention_weights = torch.softmax(scores / math.sqrt(d_k), dim=-1)  # Scale by sqrt(d_k) for stability

# Good: Clarifies non-obvious operations
packet_bytes = packet_bytes[:max_len]  # Truncate to prevent memory overflow

# Bad: States the obvious
x = x + 1  # Add 1 to x
```

### Type Hints
Always use type hints for better code clarity:

```python
from typing import List, Tuple, Optional, Union, Dict
import numpy as np
import torch

def process_packet_batch(
    packets: List[bytes],
    labels: Optional[List[int]] = None,
    augment: bool = False
) -> Tuple[torch.Tensor, Optional[torch.Tensor]]:
    """Process a batch of packets into model-ready tensors."""
    # Implementation
    
def get_encoding_stats(
    images: np.ndarray,
    method: str
) -> Dict[str, Union[float, int]]:
    """Calculate statistics for encoded images."""
    # Implementation
```

## Jupyter Notebook Documentation

### Notebook Structure
Each notebook should follow this structure:

```python
# Cell 1: Title and Overview
"""
# Notebook Title: Packet-to-Image Encoding Strategies

## Overview
This notebook explores different methods for converting network packet 
bytes into 2D images for Vision Transformer processing.

## Learning Objectives
By the end of this notebook, you will:
1. Understand three encoding methods: sequential, Hilbert, and spiral
2. Visualize the differences between encoding strategies
3. Analyze the impact on model performance
4. Choose appropriate encoding for your use case

## Prerequisites
- Basic understanding of numpy arrays
- Familiarity with image processing concepts
- Knowledge of network packet structure

## Runtime
Estimated time: 45 minutes
GPU required: No
"""

# Cell 2: Imports and Setup
"""
## 1. Environment Setup

First, let's import the necessary libraries and set up our environment.
"""
import numpy as np
import matplotlib.pyplot as plt
# ... more imports

# Cell 3: Configuration
"""
## 2. Configuration

Define constants and parameters used throughout the notebook.
"""
# Image encoding parameters
IMAGE_SIZE = 224  # Standard ViT input size
PATCH_SIZE = 16   # Size of each patch
NUM_PATCHES = (IMAGE_SIZE // PATCH_SIZE) ** 2

# Visualization settings
plt.style.use('seaborn-v0_8-darkgrid')
FIGSIZE = (12, 8)
```

### Markdown Cell Best Practices

```python
"""
### Understanding Hilbert Curve Encoding

The Hilbert curve is a space-filling curve that maps 1D data to 2D space
while preserving locality. This property is particularly useful for packet
analysis because:

1. **Locality Preservation**: Adjacent bytes remain near each other
2. **Pattern Recognition**: Contiguous malware signatures stay together
3. **Efficient Scanning**: The ViT can process related bytes in nearby patches

Here's how it works:
"""

# Follow with code demonstration
```

### Code Cell Documentation

```python
# Each code cell should have a clear purpose
"""
Generate sample packet data for visualization
"""
def generate_sample_packet(size: int, pattern: str = 'malware') -> bytes:
    """Create synthetic packet data with known patterns."""
    if pattern == 'malware':
        # Simulate common malware byte patterns
        header = b'\x4d\x5a'  # PE header
        payload = b'\x90' * 100  # NOP sled
        return header + payload + bytes(range(size - len(header) - len(payload)))
    else:
        # Benign traffic pattern
        return bytes(np.random.randint(0, 256, size, dtype=np.uint8))

# Generate examples
malware_packet = generate_sample_packet(1024, 'malware')
benign_packet = generate_sample_packet(1024, 'benign')

print(f"Malware packet preview: {malware_packet[:20].hex()}")
print(f"Benign packet preview: {benign_packet[:20].hex()}")
```

### Output Interpretation

```python
"""
### Interpreting the Results

The visualization above shows:
- **Left**: Sequential encoding spreads the NOP sled horizontally
- **Center**: Hilbert encoding creates a more compact representation
- **Right**: Spiral encoding centers important bytes

Notice how the Hilbert encoding keeps the malware signature (bright region)
more localized, which should help the attention mechanism focus on relevant
patterns.
"""
```

## API Documentation Standards

### REST API Endpoints

```python
@app.route('/api/v1/predict', methods=['POST'])
def predict():
    """
    Predict malware probability for network packet.
    
    Endpoint: POST /api/v1/predict
    
    Request Body:
        {
            "packet_bytes": "base64_encoded_packet_data",
            "encoding_method": "hilbert" | "sequential" | "spiral",
            "return_attention": boolean (optional, default: false)
        }
        
    Response:
        Success (200):
        {
            "prediction": "malware" | "benign",
            "confidence": float (0-1),
            "processing_time_ms": float,
            "attention_weights": [...] (if requested)
        }
        
        Error (400):
        {
            "error": "Invalid packet data",
            "details": "Packet size exceeds maximum allowed"
        }
        
    Example:
        curl -X POST https://api.example.com/api/v1/predict \
          -H "Content-Type: application/json" \
          -d '{"packet_bytes": "AQIDBAUGBwg=", "encoding_method": "hilbert"}'
    """
    # Implementation
```

### Configuration Documentation

```python
# config.py
"""
Configuration module for ViT packet classifier.

Environment Variables:
    MODEL_PATH: Path to trained model checkpoint
    BATCH_SIZE: Inference batch size (default: 32)
    MAX_PACKET_SIZE: Maximum packet size in bytes (default: 65535)
    LOG_LEVEL: Logging verbosity (default: INFO)
    
Config File Format (config.yaml):
    model:
        architecture: vit_base_patch16_224
        num_classes: 2
        pretrained: true
        
    preprocessing:
        image_size: 224
        encoding_method: hilbert
        normalization: true
        
    inference:
        batch_size: 32
        device: cuda
        mixed_precision: true
"""

class Config:
    """
    Configuration container for model inference.
    
    Attributes:
        model_config (dict): Model architecture parameters
        preprocess_config (dict): Preprocessing settings
        inference_config (dict): Inference runtime settings
    """
```

## Documentation Generation

### Automated Documentation
Use tools to generate API documentation:

```bash
# Generate HTML documentation
pdoc --html --output-dir docs/api src/

# Generate Markdown documentation
pydoc-markdown -p src > docs/api_reference.md
```

### Documentation Testing
Include doctests for verification:

```python
def normalize_packet_image(image: np.ndarray) -> np.ndarray:
    """
    Normalize packet image to [0, 1] range.
    
    >>> image = np.array([[0, 128, 255]])
    >>> normalized = normalize_packet_image(image)
    >>> np.allclose(normalized, [[0.0, 0.502, 1.0]], atol=0.01)
    True
    """
    return image.astype(np.float32) / 255.0
```

## Best Practices Summary

1. **Be Comprehensive**: Document all public APIs
2. **Show Examples**: Include working code examples
3. **Explain Why**: Don't just describe what, explain why
4. **Keep Updated**: Update docs with code changes
5. **Test Documentation**: Verify examples work
6. **Use Type Hints**: Improve code clarity
7. **Follow Standards**: Consistent formatting
8. **Consider Audience**: Write for your users
9. **Include Performance**: Note complexity when relevant
10. **Cross-Reference**: Link related functions/classes