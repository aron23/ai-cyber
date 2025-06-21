# Tutorial 3: Your First Packet Classification

## Overview

In this tutorial, you'll build your first packet classification model using Vision Transformers. We'll take network packet data, convert it to images, and use a pre-trained ViT model to classify packets as benign or malicious. By the end of this tutorial, you'll have a working classifier and understand the complete pipeline.

## Learning Objectives

By completing this tutorial, you will:
- Load and preprocess network packet data
- Convert packet bytes to image representations
- Fine-tune a Vision Transformer for packet classification
- Evaluate model performance
- Visualize model predictions and attention maps

## Prerequisites

Before starting this tutorial, ensure you have:
- Completed Tutorial 1: Understanding Network Packets
- Completed Tutorial 2: Why Vision Transformers for Malware?
- Basic Python programming knowledge
- Familiarity with PyTorch basics
- Environment setup as per installation guide

## Time Required

Approximately 2-3 hours

## Key Concepts

### Packet to Image Conversion
Network packets are sequences of bytes that need to be transformed into 2D images for ViT processing. We'll explore three encoding methods:
1. **Sequential encoding**: Linear arrangement of bytes
2. **Hilbert curve encoding**: Space-filling curve that preserves locality
3. **Spiral encoding**: Circular arrangement from center outward

### Transfer Learning with ViT
Instead of training from scratch, we'll leverage pre-trained Vision Transformers and fine-tune them for our specific task. This approach:
- Reduces training time significantly
- Requires less data
- Achieves better performance

## Step 1: Environment Setup

First, let's import the necessary libraries and set up our environment:

```python
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from transformers import ViTForImageClassification, ViTImageProcessor
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import seaborn as sns
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Set random seeds for reproducibility
torch.manual_seed(42)
np.random.seed(42)

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
```

## Step 2: Loading Packet Data

For this tutorial, we'll use a sample dataset of network packets. Each packet is labeled as either benign (0) or malicious (1).

```python
def load_packet_data(data_path):
    """
    Load packet data from CSV file.
    Expected format: packet_bytes (hex string), label (0 or 1)
    """
    df = pd.read_csv(data_path)
    
    # Convert hex strings to byte arrays
    packets = []
    labels = []
    
    for _, row in df.iterrows():
        # Convert hex string to bytes
        packet_bytes = bytes.fromhex(row['packet_bytes'])
        packets.append(packet_bytes)
        labels.append(row['label'])
    
    return packets, labels

# Load sample data (replace with your data path)
# For this tutorial, we'll create synthetic data
def create_sample_data(n_samples=1000):
    """Create synthetic packet data for demonstration"""
    packets = []
    labels = []
    
    for i in range(n_samples):
        # Create random packet (1024 bytes)
        if i < n_samples // 2:
            # Benign packets - more structured patterns
            packet = np.random.randint(0, 128, size=1024, dtype=np.uint8)
            labels.append(0)
        else:
            # Malicious packets - more random patterns
            packet = np.random.randint(0, 256, size=1024, dtype=np.uint8)
            labels.append(1)
        
        packets.append(bytes(packet))
    
    return packets, labels

packets, labels = create_sample_data(1000)
print(f"Loaded {len(packets)} packets")
print(f"Benign: {labels.count(0)}, Malicious: {labels.count(1)}")
```

## Step 3: Packet to Image Conversion

Now let's implement different encoding methods to convert packet bytes into images:

```python
class PacketImageEncoder:
    """Convert packet bytes to image representations"""
    
    def __init__(self, image_size=224):
        self.image_size = image_size
        self.total_pixels = image_size * image_size
    
    def encode_sequential(self, packet_bytes):
        """Sequential encoding: arrange bytes linearly"""
        # Pad or truncate to fit image size
        byte_array = np.frombuffer(packet_bytes, dtype=np.uint8)
        
        if len(byte_array) < self.total_pixels:
            # Pad with zeros
            byte_array = np.pad(byte_array, (0, self.total_pixels - len(byte_array)))
        else:
            # Truncate
            byte_array = byte_array[:self.total_pixels]
        
        # Reshape to 2D image
        image = byte_array.reshape(self.image_size, self.image_size)
        
        # Convert to RGB (repeat grayscale values)
        image_rgb = np.stack([image, image, image], axis=-1)
        
        return Image.fromarray(image_rgb.astype(np.uint8))
    
    def encode_hilbert(self, packet_bytes):
        """Hilbert curve encoding: preserves spatial locality"""
        from hilbertcurve.hilbertcurve import HilbertCurve
        
        # Create Hilbert curve
        p = int(np.log2(self.image_size))
        hilbert_curve = HilbertCurve(p, 2)
        
        byte_array = np.frombuffer(packet_bytes, dtype=np.uint8)
        
        # Create image
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        for i, byte_val in enumerate(byte_array):
            if i >= self.total_pixels:
                break
            
            # Get 2D coordinates from Hilbert curve
            coords = hilbert_curve.point_from_distance(i)
            image[coords[0], coords[1]] = byte_val
        
        # Convert to RGB
        image_rgb = np.stack([image, image, image], axis=-1)
        
        return Image.fromarray(image_rgb)
    
    def encode_spiral(self, packet_bytes):
        """Spiral encoding: arrange bytes in spiral pattern"""
        byte_array = np.frombuffer(packet_bytes, dtype=np.uint8)
        image = np.zeros((self.image_size, self.image_size), dtype=np.uint8)
        
        # Generate spiral coordinates
        x, y = self.image_size // 2, self.image_size // 2
        dx, dy = 0, -1
        
        for i, byte_val in enumerate(byte_array):
            if i >= self.total_pixels:
                break
            
            if 0 <= x < self.image_size and 0 <= y < self.image_size:
                image[y, x] = byte_val
            
            # Spiral movement logic
            if x == y or (x < 0 and x == -y) or (x > 0 and x == 1 - y):
                dx, dy = -dy, dx
            
            x, y = x + dx, y + dy
        
        # Convert to RGB
        image_rgb = np.stack([image, image, image], axis=-1)
        
        return Image.fromarray(image_rgb)

# Create encoder instance
encoder = PacketImageEncoder(image_size=224)

# Visualize different encoding methods
sample_packet = packets[0]
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

encodings = [
    ('Sequential', encoder.encode_sequential(sample_packet)),
    ('Hilbert', encoder.encode_hilbert(sample_packet)),
    ('Spiral', encoder.encode_spiral(sample_packet))
]

for i, (name, img) in enumerate(encodings):
    axes[i].imshow(img)
    axes[i].set_title(f'{name} Encoding')
    axes[i].axis('off')

plt.tight_layout()
plt.show()
```

## Step 4: Preparing Data for Training

Let's prepare our dataset for training the Vision Transformer:

```python
from torch.utils.data import Dataset, DataLoader

class PacketImageDataset(Dataset):
    """PyTorch dataset for packet images"""
    
    def __init__(self, packets, labels, encoder, processor, encoding_method='sequential'):
        self.packets = packets
        self.labels = labels
        self.encoder = encoder
        self.processor = processor
        self.encoding_method = encoding_method
    
    def __len__(self):
        return len(self.packets)
    
    def __getitem__(self, idx):
        # Convert packet to image
        packet = self.packets[idx]
        
        if self.encoding_method == 'sequential':
            image = self.encoder.encode_sequential(packet)
        elif self.encoding_method == 'hilbert':
            image = self.encoder.encode_hilbert(packet)
        else:
            image = self.encoder.encode_spiral(packet)
        
        # Process image for ViT
        inputs = self.processor(images=image, return_tensors="pt")
        
        # Remove batch dimension
        pixel_values = inputs['pixel_values'].squeeze(0)
        
        # Get label
        label = torch.tensor(self.labels[idx], dtype=torch.long)
        
        return {
            'pixel_values': pixel_values,
            'labels': label
        }

# Split data into train/validation/test sets
X_train, X_test, y_train, y_test = train_test_split(
    packets, labels, test_size=0.2, random_state=42, stratify=labels
)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

print(f"Train set: {len(X_train)} samples")
print(f"Validation set: {len(X_val)} samples")
print(f"Test set: {len(X_test)} samples")
```

## Step 5: Loading and Fine-tuning Vision Transformer

Now let's load a pre-trained ViT model and prepare it for our binary classification task:

```python
# Load pre-trained ViT model and processor
model_name = "google/vit-base-patch16-224"
processor = ViTImageProcessor.from_pretrained(model_name)
model = ViTForImageClassification.from_pretrained(
    model_name,
    num_labels=2,  # Binary classification
    ignore_mismatched_sizes=True
)

# Move model to device
model = model.to(device)

# Create datasets
train_dataset = PacketImageDataset(X_train, y_train, encoder, processor)
val_dataset = PacketImageDataset(X_val, y_val, encoder, processor)
test_dataset = PacketImageDataset(X_test, y_test, encoder, processor)

# Create data loaders
batch_size = 32
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Set up optimizer and loss function
optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
criterion = nn.CrossEntropyLoss()

# Training parameters
num_epochs = 5
best_val_accuracy = 0
```

## Step 6: Training the Model

Let's implement the training loop:

```python
def train_epoch(model, dataloader, optimizer, criterion, device):
    """Train for one epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for batch in tqdm(dataloader, desc="Training"):
        # Move data to device
        pixel_values = batch['pixel_values'].to(device)
        labels = batch['labels'].to(device)
        
        # Forward pass
        outputs = model(pixel_values)
        loss = criterion(outputs.logits, labels)
        
        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        _, predicted = torch.max(outputs.logits, 1)
        total_loss += loss.item()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy

def evaluate(model, dataloader, criterion, device):
    """Evaluate model on validation/test set"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    all_predictions = []
    all_labels = []
    
    with torch.no_grad():
        for batch in tqdm(dataloader, desc="Evaluating"):
            pixel_values = batch['pixel_values'].to(device)
            labels = batch['labels'].to(device)
            
            outputs = model(pixel_values)
            loss = criterion(outputs.logits, labels)
            
            _, predicted = torch.max(outputs.logits, 1)
            total_loss += loss.item()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)
            
            all_predictions.extend(predicted.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
    
    avg_loss = total_loss / len(dataloader)
    accuracy = correct / total
    
    return avg_loss, accuracy, all_predictions, all_labels

# Training loop
print("Starting training...")
train_losses = []
train_accuracies = []
val_losses = []
val_accuracies = []

for epoch in range(num_epochs):
    print(f"\nEpoch {epoch + 1}/{num_epochs}")
    
    # Train
    train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
    train_losses.append(train_loss)
    train_accuracies.append(train_acc)
    
    # Validate
    val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
    val_losses.append(val_loss)
    val_accuracies.append(val_acc)
    
    print(f"Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}")
    print(f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}")
    
    # Save best model
    if val_acc > best_val_accuracy:
        best_val_accuracy = val_acc
        torch.save(model.state_dict(), 'best_packet_classifier.pth')
        print("Saved best model!")
```

## Step 7: Evaluating Model Performance

Let's evaluate our trained model on the test set and visualize the results:

```python
# Load best model
model.load_state_dict(torch.load('best_packet_classifier.pth'))

# Evaluate on test set
test_loss, test_acc, predictions, true_labels = evaluate(
    model, test_loader, criterion, device
)

print(f"\nTest Set Performance:")
print(f"Test Loss: {test_loss:.4f}")
print(f"Test Accuracy: {test_acc:.4f}")

# Confusion Matrix
cm = confusion_matrix(true_labels, predictions)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Benign', 'Malicious'],
            yticklabels=['Benign', 'Malicious'])
plt.title('Confusion Matrix')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.show()

# Classification Report
print("\nClassification Report:")
print(classification_report(true_labels, predictions, 
                          target_names=['Benign', 'Malicious']))

# Plot training history
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

# Loss plot
ax1.plot(range(1, num_epochs + 1), train_losses, label='Train Loss')
ax1.plot(range(1, num_epochs + 1), val_losses, label='Val Loss')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Loss')
ax1.set_title('Training and Validation Loss')
ax1.legend()
ax1.grid(True)

# Accuracy plot
ax2.plot(range(1, num_epochs + 1), train_accuracies, label='Train Acc')
ax2.plot(range(1, num_epochs + 1), val_accuracies, label='Val Acc')
ax2.set_xlabel('Epoch')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training and Validation Accuracy')
ax2.legend()
ax2.grid(True)

plt.tight_layout()
plt.show()
```

## Step 8: Visualizing Attention Maps

One of the advantages of Vision Transformers is their interpretability through attention visualization:

```python
def visualize_attention(model, image, processor):
    """Visualize attention maps from the Vision Transformer"""
    model.eval()
    
    # Process image
    inputs = processor(images=image, return_tensors="pt")
    pixel_values = inputs['pixel_values'].to(device)
    
    # Get model outputs with attention
    with torch.no_grad():
        outputs = model(pixel_values, output_attentions=True)
    
    # Get attention weights from last layer
    attentions = outputs.attentions[-1]  # Shape: (batch, num_heads, seq_len, seq_len)
    
    # Average over heads
    attention = attentions[0].mean(dim=0).cpu().numpy()
    
    # Get attention to CLS token (first token)
    cls_attention = attention[0, 1:]  # Skip CLS token itself
    
    # Reshape to 2D (ViT uses 14x14 patches for 224x224 images)
    num_patches = int(np.sqrt(cls_attention.shape[0]))
    cls_attention = cls_attention.reshape(num_patches, num_patches)
    
    # Visualize
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
    
    # Original image
    ax1.imshow(image)
    ax1.set_title('Original Packet Image')
    ax1.axis('off')
    
    # Attention map
    im = ax2.imshow(cls_attention, cmap='hot', interpolation='nearest')
    ax2.set_title('Attention Map')
    ax2.axis('off')
    plt.colorbar(im, ax=ax2)
    
    plt.tight_layout()
    plt.show()
    
    return cls_attention

# Visualize attention for a few test samples
print("\nVisualizing Attention Maps:")
for i in range(3):
    packet = X_test[i]
    label = y_test[i]
    
    # Convert packet to image
    image = encoder.encode_sequential(packet)
    
    # Get prediction
    inputs = processor(images=image, return_tensors="pt")
    with torch.no_grad():
        outputs = model(inputs['pixel_values'].to(device))
        prediction = torch.argmax(outputs.logits, dim=1).item()
    
    print(f"\nSample {i + 1}:")
    print(f"True Label: {'Malicious' if label == 1 else 'Benign'}")
    print(f"Predicted: {'Malicious' if prediction == 1 else 'Benign'}")
    
    # Visualize attention
    visualize_attention(model, image, processor)
```

## Step 9: Analyzing Different Encoding Methods

Let's compare the performance of different encoding methods:

```python
def compare_encoding_methods(packets, labels, encoder, processor, device):
    """Compare performance of different encoding methods"""
    
    encoding_methods = ['sequential', 'hilbert', 'spiral']
    results = {}
    
    for method in encoding_methods:
        print(f"\nTraining with {method} encoding...")
        
        # Create datasets with specific encoding
        train_dataset = PacketImageDataset(X_train, y_train, encoder, processor, method)
        val_dataset = PacketImageDataset(X_val, y_val, encoder, processor, method)
        test_dataset = PacketImageDataset(X_test, y_test, encoder, processor, method)
        
        # Create data loaders
        train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create new model
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=2,
            ignore_mismatched_sizes=True
        ).to(device)
        
        optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5)
        
        # Train for fewer epochs for comparison
        best_val_acc = 0
        for epoch in range(3):
            train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
            val_loss, val_acc, _, _ = evaluate(model, val_loader, criterion, device)
            
            if val_acc > best_val_acc:
                best_val_acc = val_acc
        
        # Test performance
        test_loss, test_acc, _, _ = evaluate(model, test_loader, criterion, device)
        
        results[method] = {
            'best_val_acc': best_val_acc,
            'test_acc': test_acc
        }
        
        print(f"{method} - Val Acc: {best_val_acc:.4f}, Test Acc: {test_acc:.4f}")
    
    return results

# Compare encoding methods
encoding_results = compare_encoding_methods(packets, labels, encoder, processor, device)

# Visualize comparison
methods = list(encoding_results.keys())
val_accs = [encoding_results[m]['best_val_acc'] for m in methods]
test_accs = [encoding_results[m]['test_acc'] for m in methods]

x = np.arange(len(methods))
width = 0.35

fig, ax = plt.subplots(figsize=(10, 6))
ax.bar(x - width/2, val_accs, width, label='Validation Accuracy')
ax.bar(x + width/2, test_accs, width, label='Test Accuracy')

ax.set_xlabel('Encoding Method')
ax.set_ylabel('Accuracy')
ax.set_title('Performance Comparison of Encoding Methods')
ax.set_xticks(x)
ax.set_xticklabels(methods)
ax.legend()
ax.grid(True, axis='y')

plt.tight_layout()
plt.show()
```

## Step 10: Deploying Your Model

Here's how to save and use your trained model for inference:

```python
class PacketClassifier:
    """Wrapper class for easy packet classification"""
    
    def __init__(self, model_path, model_name="google/vit-base-patch16-224", 
                 encoding_method='sequential'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.encoder = PacketImageEncoder(image_size=224)
        self.processor = ViTImageProcessor.from_pretrained(model_name)
        self.encoding_method = encoding_method
        
        # Load model
        self.model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=2,
            ignore_mismatched_sizes=True
        )
        self.model.load_state_dict(torch.load(model_path, map_location=self.device))
        self.model.to(self.device)
        self.model.eval()
    
    def classify_packet(self, packet_bytes):
        """Classify a single packet"""
        # Convert to image
        if self.encoding_method == 'sequential':
            image = self.encoder.encode_sequential(packet_bytes)
        elif self.encoding_method == 'hilbert':
            image = self.encoder.encode_hilbert(packet_bytes)
        else:
            image = self.encoder.encode_spiral(packet_bytes)
        
        # Process image
        inputs = self.processor(images=image, return_tensors="pt")
        pixel_values = inputs['pixel_values'].to(self.device)
        
        # Get prediction
        with torch.no_grad():
            outputs = self.model(pixel_values)
            probabilities = torch.softmax(outputs.logits, dim=1)
            prediction = torch.argmax(outputs.logits, dim=1).item()
        
        return {
            'prediction': 'malicious' if prediction == 1 else 'benign',
            'confidence': probabilities[0, prediction].item(),
            'probabilities': {
                'benign': probabilities[0, 0].item(),
                'malicious': probabilities[0, 1].item()
            }
        }
    
    def classify_batch(self, packet_list):
        """Classify multiple packets"""
        results = []
        for packet in packet_list:
            result = self.classify_packet(packet)
            results.append(result)
        return results

# Example usage
classifier = PacketClassifier('best_packet_classifier.pth')

# Classify a single packet
sample_packet = X_test[0]
result = classifier.classify_packet(sample_packet)
print(f"Classification: {result['prediction']}")
print(f"Confidence: {result['confidence']:.2%}")
print(f"Probabilities: Benign={result['probabilities']['benign']:.2%}, "
      f"Malicious={result['probabilities']['malicious']:.2%}")

# Batch classification
batch_results = classifier.classify_batch(X_test[:10])
for i, result in enumerate(batch_results):
    true_label = 'malicious' if y_test[i] == 1 else 'benign'
    print(f"Packet {i}: True={true_label}, Predicted={result['prediction']}, "
          f"Confidence={result['confidence']:.2%}")
```

## Practical Exercises

### Exercise 1: Implement Custom Encoding
Create your own packet-to-image encoding method. Consider patterns that might better preserve packet structure.

```python
def encode_custom(self, packet_bytes):
    """
    Implement your custom encoding method here.
    Ideas:
    - Group bytes by protocol fields
    - Use color channels for different byte ranges
    - Apply transformations based on packet type
    """
    # Your implementation here
    pass
```

### Exercise 2: Feature Engineering
Enhance the model by incorporating packet metadata:

```python
def extract_packet_features(packet_bytes):
    """
    Extract additional features from packet bytes.
    Examples:
    - Packet size
    - Entropy
    - Byte frequency distribution
    - Protocol-specific fields
    """
    # Your implementation here
    pass
```

### Exercise 3: Model Optimization
Experiment with different hyperparameters and model configurations:
- Try different learning rates
- Adjust batch sizes
- Implement learning rate scheduling
- Try different ViT model sizes (tiny, small, base, large)

### Exercise 4: Advanced Visualization
Create more sophisticated visualizations:
- Attention rollout
- Head-wise attention analysis
- Feature map visualization
- t-SNE visualization of packet embeddings

## Troubleshooting Common Issues

### Memory Issues
If you encounter out-of-memory errors:
```python
# Reduce batch size
batch_size = 16  # or even 8

# Use gradient accumulation
accumulation_steps = 4
for i, batch in enumerate(dataloader):
    loss = compute_loss(batch)
    loss = loss / accumulation_steps
    loss.backward()
    
    if (i + 1) % accumulation_steps == 0:
        optimizer.step()
        optimizer.zero_grad()

# Use mixed precision training
from torch.cuda.amp import autocast, GradScaler
scaler = GradScaler()

with autocast():
    outputs = model(inputs)
    loss = criterion(outputs, labels)
```

### Poor Performance
If your model isn't performing well:
1. Check data quality and labels
2. Try different encoding methods
3. Increase training epochs
4. Use data augmentation
5. Adjust learning rate
6. Try different pre-trained models

### Slow Training
To speed up training:
1. Use GPU if available
2. Enable mixed precision training
3. Use larger batch sizes
4. Implement efficient data loading
5. Use compiled models (PyTorch 2.0+)

## Summary

Congratulations! You've successfully built your first packet classification model using Vision Transformers. You've learned:

✓ How to convert network packets to images
✓ How to fine-tune pre-trained ViT models
✓ How to evaluate model performance
✓ How to visualize attention mechanisms
✓ How to deploy your model for inference

This foundation prepares you for more advanced topics like few-shot learning, multi-class classification, and real-time packet analysis.

## Next Steps

1. **Tutorial 4: Interpreting Model Results** - Deep dive into understanding what your model learned
2. **Advanced Tutorial: Few-Shot Learning** - Train models with limited labeled data
3. **Research Guide: Custom Architectures** - Design specialized models for packet analysis

## Further Reading

- [Vision Transformer Paper](https://arxiv.org/abs/2010.11929)
- [An Image is Worth 16x16 Words: Transformers for Image Recognition at Scale](https://arxiv.org/abs/2010.11929)
- [Network Traffic Classification Survey](https://example.com)
- [PyTorch Vision Transformer Tutorial](https://pytorch.org/tutorials/)

## Code Repository

Find the complete code for this tutorial in our GitHub repository:
```bash
git clone https://github.com/your-org/packet-vit-tutorial
cd packet-vit-tutorial/tutorial3
```

Remember to check the repository for updates, additional examples, and community contributions!