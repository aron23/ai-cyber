# Vision Transformer Experiments for Network Traffic Classification

## Overview

This repository contains experiments applying Vision Transformers (ViT) to network traffic classification by converting packet payloads into visual representations. The research demonstrates that treating network security data as images enables computer vision techniques to identify attack patterns effectively.

## Dataset Generation

### CIC-IoT23 Dataset Processing

The `CIC-data-sampler.ipynb` notebook processes raw PCAP files from the CIC-IoT23 dataset to create multi-format image representations:

- **Source**: CIC-IoT23 PCAP files from Google Cloud Storage
- **Sampling**: 12,000 samples per attack class
- **Payload Size**: First 1,500 bytes of each packet
- **Output Formats**: Parquet files and PNG samples
- **Organization**: Data organized by label for efficient access

### Image Encoding Methods

Five distinct encoding strategies were implemented to transform network payloads into images:

1. **Grayscale 32×32**: Sequential byte mapping to single-channel images
2. **Grayscale 39×39**: Larger grayscale representation for increased spatial resolution  
3. **Grayscale 64×64**: Extended grayscale format (data preparation only)
4. **RGB Hilbert 32×32**: Hilbert curve mapping preserving byte locality in RGB space
5. **RGB Spiral 32×32**: Spiral pattern encoding for RGB channels
6. **5-Channel 32×32**: Multi-view representation with specialized channels:
   - Channel 1: Raw bytes
   - Channel 2: Header emphasis (first 64 bytes)
   - Channel 3: Byte frequency map
   - Channel 4: Local entropy
   - Channel 5: Gradient magnitude

## Experimental Setup

### Vision Transformer Architecture

All experiments utilized a consistent ViT architecture:

- **Patch Size**: 16×16 pixels
- **Patches per Image**: 4 (for 32×32 images)
- **Embedding Dimension**: 192-256
- **Attention Heads**: 3-4
- **Transformer Layers**: 6
- **Classification**: 9 attack classes

### Attack Classes

The experiments classify network traffic into 9 categories:

1. Benign_Final (normal traffic)
2. DDoS-HTTP_Flood
3. DDoS-SYN_Flood
4. DictionaryBruteForce
5. DoS-TCP_Flood
6. DoS-UDP_Flood
7. Mirai-udpplain
8. Recon-PortScan
9. SqlInjection

### Training Configuration

- **Batch Size**: 32
- **Learning Rate**: 1e-4
- **Optimizer**: AdamW with weight decay
- **Scheduler**: Cosine annealing
- **Loss Function**: Weighted cross-entropy for class imbalance
- **Train/Val/Test Split**: 70%/15%/15%

## Results Summary

### Performance Comparison

| Encoding Method | Test Accuracy | Parameters | Key Characteristics |
|-----------------|---------------|------------|---------------------|
| **5-Channel 32×32** | **93.30%** | 2.92M | Best overall performance; rich multi-dimensional features |
| **RGB Hilbert 32×32** | 92.11% | 4.97M | Spatial locality preservation via Hilbert curve |
| **Grayscale 39×39** | 91.01% | 2.71M | Higher resolution with 3×3 patch grid |
| **Grayscale 32×32** | 90.52% | 2.72M | Simple yet effective; baseline approach |
| **RGB Spiral 32×32** | 79.06% | 4.97M | Center-outward spiral pattern encoding |

All models tested on CIC-IoT23 dataset with 9 attack classes

### Key Findings

1. **Multi-Channel Superiority**: The 5-channel encoding achieved the highest accuracy (93.30% validation), demonstrating that multiple perspectives of payload data enhance classification performance.

2. **Spatial Encoding Impact**: RGB Hilbert encoding (92.11%) significantly outperformed RGB Spiral (79.06%), highlighting the importance of locality-preserving mappings. The Hilbert curve's ability to maintain spatial proximity proved more effective than the spiral pattern.

3. **Model Efficiency**: Despite having fewer parameters than RGB models, the 5-channel approach achieved superior results through information-rich encoding.

4. **Resolution Impact**: The 39×39 grayscale model (91.01%) performed slightly better than 32×32 grayscale (90.52%), suggesting marginal benefits from increased spatial resolution.

5. **Attack-Specific Performance**:
   - **Best Classified**: DoS-UDP_Flood (98.44%), DDoS-HTTP_Flood (97.72%)
   - **Most Challenging**: Recon-PortScan (86.33%), DictionaryBruteForce (87.78%)

### Confusion Analysis

Common misclassifications occurred between:
- SqlInjection ↔ DictionaryBruteForce (similar payload patterns)
- Recon-PortScan ↔ Benign_Final (reconnaissance mimics normal behavior)
- DDoS variants showed strong inter-class distinction

## Initial Proof of Concept

The original proof-of-concept notebook (`ViT_Prototype_Proof_of_Concept.ipynb`) validated the approach on the UNSW-NB15 dataset:

- **Dataset**: 79,881 samples across 10 attack types (different from CIC-IoT23)
- **Architecture**: 39×39 grayscale images with 16×16 patches
- **Result**: 76.66% test accuracy on 10-class problem
- **Significance**: Confirmed viability of treating network payloads as images

## Technical Implementation

### Data Pipeline

1. **PCAP Processing**: Extract packet payloads from network captures
2. **Byte Normalization**: Scale byte values [0-255] to [0-1]
3. **Image Formation**: Reshape according to encoding strategy
4. **Augmentation**: Minimal to preserve payload integrity

### Model Training

- Early stopping with patience=5
- Best model selection based on validation accuracy
- Comprehensive evaluation metrics including per-class precision/recall

## Future Directions

1. **Encoding Optimization**: Explore learned encoding strategies
2. **Architecture Scaling**: Test larger ViT models and different patch sizes
3. **Real-time Deployment**: Optimize for production inference
4. **Explainability**: Attention visualization for attack signature identification

## Conclusion

This research successfully demonstrates that Vision Transformers can effectively classify network traffic by treating packet payloads as images. The 5-channel encoding approach achieved 93.30% validation accuracy across 9 attack types, validating the potential of computer vision techniques in cybersecurity applications. The experiments establish a foundation for future research in visual-based network security analysis.

## Repository Structure

```
ViT-experiment/
├── CIC-data-sampler.ipynb               # Dataset generation from CIC-IoT23 PCAP files
├── ViT_Prototype_Proof_of_Concept.ipynb # Initial validation on UNSW-NB15 dataset
├── ViT_Prototype_grayscale_32x32.ipynb  # Grayscale 32×32 experiments (CIC-IoT23)
├── ViT_Prototype_grayscale_39x39.ipynb  # Grayscale 39×39 experiments (CIC-IoT23)
├── ViT_Prototype_5channel_32x32.ipynb   # 5-channel experiments (CIC-IoT23)
├── ViT_Prototype_rgb_hilbert_32x32.ipynb # RGB Hilbert curve experiments (CIC-IoT23)
├── ViT_Prototype_rgb_spiral_32x32.ipynb  # RGB spiral experiments (CIC-IoT23)
├── results_*.json                        # Detailed performance metrics
├── best_*_vit_model.pth                 # Trained model checkpoints
└── pcap-dataset-samples/                # Generated image datasets from CIC-IoT23
```

## Requirements

- PyTorch >= 1.9.0
- NumPy, Pandas, Scikit-learn
- Matplotlib, Seaborn
- Google Cloud Storage (for dataset access)
- CUDA-capable GPU (recommended) or 8-16 GB spare RAM