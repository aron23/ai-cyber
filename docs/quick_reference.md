# Quick Reference Guide

## Common Operations

### Data Loading

```python
# Load raw packet data
from src.data.data_loader import PayloadByteDataLoader

loader = PayloadByteDataLoader('data/raw/payload_byte')
df = loader.load_raw_packets(nrows=10000)

# Create train/val/test splits
train_df, val_df, test_df = loader.create_train_val_test_split(df)

# Get PyTorch dataset
train_dataset = loader.get_pytorch_dataset('train')
```

### Packet to Image Conversion

```python
# Convert packets to images
from src.data.packet_to_image import PacketImageEncoder

encoder = PacketImageEncoder(image_size=(224, 224))

# Single packet
image = encoder.encode_sequential(packet_bytes)

# Batch processing
images = encoder.encode_batch(packet_batch, method='hilbert')
```

### Configuration Management

```python
# Access configuration
from src.utils.config_manager import get_config_manager

config = get_config_manager()
batch_size = config.get('data_config', 'pipeline.batch_size')

# Override from environment
export PAYLOADBYTE_DATA_CONFIG_BATCH_SIZE=64
```

### Model Registry

```python
# Register a model
from src.models.model_registry import ModelRegistry

registry = ModelRegistry()
registry.register_model(model, 'vit_malware_v1', 
                       metrics={'accuracy': 0.95})

# Load a model
model = registry.load_model('vit_malware_v1', stage='Production')
```

### Feature Storage

```python
# Save features
from src.data.feature_store import FeatureStore

store = FeatureStore('data/features/store.h5')
store.save_features('embeddings', embeddings_array, version='1.0')

# Load features
embeddings = store.load_features('embeddings', version='1.0')
```

### Logging

```python
# Setup logger
from src.utils.logger import setup_logger

logger = setup_logger(__name__)
logger.info('Starting process')
logger.error('Error occurred', exc_info=True)
```

## Common Patterns

### Data Pipeline

```python
# Complete data pipeline example
from src.data.data_loader import PayloadByteDataLoader
from src.data.packet_to_image import PacketImageEncoder
from torch.utils.data import DataLoader

# 1. Load data
loader = PayloadByteDataLoader('data/raw')
train_df, val_df, test_df = loader.create_train_val_test_split(
    loader.load_raw_packets()
)

# 2. Create datasets
train_dataset = loader.get_pytorch_dataset('train')

# 3. Create data loader
train_loader = DataLoader(
    train_dataset,
    batch_size=32,
    shuffle=True,
    num_workers=4
)

# 4. Process batches
encoder = PacketImageEncoder()
for batch in train_loader:
    images = encoder.encode_batch(batch['bytes'])
    labels = batch['labels']
    # Train model...
```

### Model Training with Tracking

```python
# Training with MLflow tracking
import mlflow
from src.models.model_registry import ModelRegistry

registry = ModelRegistry()

with mlflow.start_run():
    # Log parameters
    mlflow.log_params({
        'batch_size': 32,
        'learning_rate': 0.001,
        'epochs': 10
    })
    
    # Train model
    for epoch in range(epochs):
        train_loss = train_epoch(model, train_loader)
        val_acc = validate(model, val_loader)
        
        # Log metrics
        mlflow.log_metrics({
            'train_loss': train_loss,
            'val_accuracy': val_acc
        }, step=epoch)
    
    # Register model
    registry.register_model(
        model, 
        'vit_malware_detector',
        metrics={'final_accuracy': val_acc}
    )
```

### Error Handling

```python
# Consistent error handling pattern
from src.utils.logger import setup_logger

logger = setup_logger(__name__)

try:
    # Risky operation
    result = process_data(data)
except FileNotFoundError as e:
    logger.error(f"Data file not found: {e}")
    # Use default or raise
except Exception as e:
    logger.error(f"Unexpected error: {e}", exc_info=True)
    raise
```

### Configuration Override

```python
# Override configuration for experiments
from src.utils.config_manager import get_config_manager

config = get_config_manager()

# Override specific values
config.set('data_config', 'pipeline.batch_size', 64)
config.set('model_config', 'learning_rate', 0.0001)

# Or use environment variables
# export PAYLOADBYTE_MODEL_CONFIG_LEARNING_RATE=0.0001
config.override_from_env()
```

## Debugging Tips

### Memory Issues

```python
# Monitor memory usage
import psutil
import gc

def log_memory_usage():
    process = psutil.Process()
    logger.info(f"Memory usage: {process.memory_info().rss / 1024 / 1024:.2f} MB")

# Force garbage collection
gc.collect()
torch.cuda.empty_cache()  # If using GPU
```

### Data Loading Performance

```python
# Profile data loading
import time

start = time.time()
data = loader.load_raw_packets(nrows=10000)
logger.info(f"Loading took {time.time() - start:.2f} seconds")

# Use parallel loading
loader = PayloadByteDataLoader(
    data_path='data',
    n_workers=8,  # Increase workers
    chunk_size=50000  # Larger chunks
)
```

### Model Debugging

```python
# Check model predictions
def debug_predictions(model, sample_batch):
    model.eval()
    with torch.no_grad():
        outputs = model(sample_batch)
        probs = torch.softmax(outputs, dim=1)
        
    logger.info(f"Output shape: {outputs.shape}")
    logger.info(f"Max probability: {probs.max():.4f}")
    logger.info(f"Predicted class: {probs.argmax(dim=1)}")
```

## Environment Setup

```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt

# Set up pre-commit hooks
pre-commit install

# Run tests
pytest tests/

# Check code quality
flake8 src/
black src/ --check
```

## Common Issues and Solutions

| Issue | Solution |
|-------|----------|
| Import errors | Ensure `src` is in PYTHONPATH or run from project root |
| Config not found | Check `configs/` directory exists and contains YAML files |
| GPU not available | Verify CUDA installation with `torch.cuda.is_available()` |
| Memory overflow | Reduce batch size or use gradient accumulation |
| Slow data loading | Increase `n_workers` in DataLoader |
| MLflow connection | Check tracking URI and ensure server is running |

## Useful Commands

```bash
# Start MLflow UI
mlflow ui --backend-store-uri sqlite:///mlflow.db

# Monitor GPU usage
nvidia-smi -l 1

# Profile code
python -m cProfile -o profile.stats script.py

# View notebook without Jupyter
jupyter nbconvert --to python notebook.ipynb

# Clean up cache
find . -type d -name __pycache__ -exec rm -r {} +
```