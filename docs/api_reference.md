# API Reference

This document provides a comprehensive API reference for all modules in the Vision Transformer malware detection project.

## Table of Contents

1. [Data Module](#data-module)
   - [PacketImageEncoder](#packetimageencoder)
   - [PayloadByteDataLoader](#payloadbytedataloader)
   - [FeatureStore](#featurestore)
   - [DataVersionManager](#dataversionmanager)
2. [Models Module](#models-module)
   - [ModelRegistry](#modelregistry)
3. [Utils Module](#utils-module)
   - [Logger](#logger)
   - [ConfigManager](#configmanager)
   - [MetricsCollector](#metricscollector)

---

## Data Module

### PacketImageEncoder

**Module**: `src.data.packet_to_image`

Converts network packet bytes into image representations for Vision Transformer models.

#### Class: `PacketImageEncoder`

```python
PacketImageEncoder(image_size=(64, 64), encoding_type='grayscale')
```

**Parameters:**
- `image_size` (tuple): Target image dimensions (height, width)
- `encoding_type` (str): Type of encoding ('grayscale' or 'rgb')

**Methods:**

##### `encode_sequential(packet_bytes)`
Encode packet bytes as grayscale image using sequential filling.

```python
image = encoder.encode_sequential(packet_bytes)
```

**Parameters:**
- `packet_bytes` (np.ndarray): Array of byte values (0-255)

**Returns:**
- `np.ndarray`: 2D array representing the encoded image

##### `encode_hilbert_curve(packet_bytes)`
Encode packet bytes using Hilbert curve to preserve locality.

```python
image = encoder.encode_hilbert_curve(packet_bytes)
```

**Parameters:**
- `packet_bytes` (np.ndarray): Array of byte values (0-255)

**Returns:**
- `np.ndarray`: 2D array with Hilbert curve mapping

##### `encode_spiral(packet_bytes)`
Encode packet bytes in a spiral pattern from center outward.

```python
image = encoder.encode_spiral(packet_bytes)
```

**Parameters:**
- `packet_bytes` (np.ndarray): Array of byte values (0-255)

**Returns:**
- `np.ndarray`: 2D array with spiral mapping

##### `encode_batch(packet_batch, method='sequential')`
Encode a batch of packets efficiently.

```python
images = encoder.encode_batch(packet_batch, method='hilbert')
```

**Parameters:**
- `packet_batch` (np.ndarray): Array of shape (batch_size, packet_length)
- `method` (str): Encoding method ('sequential', 'hilbert', 'spiral', 'block')

**Returns:**
- `np.ndarray`: Array of shape (batch_size, height, width)

#### Class: `TorchPacketEncoder`

PyTorch-based packet encoder for GPU acceleration.

```python
TorchPacketEncoder(image_size=(64, 64), encoding_type='sequential')
```

**Parameters:**
- `image_size` (tuple): Target image dimensions
- `encoding_type` (str): Encoding type ('sequential', 'hilbert', 'spiral')

**Methods:**

##### `forward(packet_batch)`
Encode a batch of packets (PyTorch forward pass).

```python
images = encoder(packet_batch)
```

**Parameters:**
- `packet_batch` (torch.Tensor): Tensor of shape (batch_size, packet_length)

**Returns:**
- `torch.Tensor`: Tensor of shape (batch_size, 1, height, width)

---

### PayloadByteDataLoader

**Module**: `src.data.data_loader`

Efficient data loader for Payload-Byte network intrusion detection datasets.

#### Class: `PayloadByteDataLoader`

```python
PayloadByteDataLoader(data_path, batch_size=32, chunk_size=10000, n_workers=4, cache_dir=None)
```

**Parameters:**
- `data_path` (str/Path): Path to data directory or CSV file
- `batch_size` (int): Batch size for data loading
- `chunk_size` (int): Number of rows to read at once
- `n_workers` (int): Number of parallel workers
- `cache_dir` (Path): Directory for caching processed data

**Methods:**

##### `load_raw_packets(files=None, nrows=None)`
Load raw packet data from CSV files.

```python
df = loader.load_raw_packets(nrows=1000)
```

**Parameters:**
- `files` (list): List of files to load (None for all)
- `nrows` (int): Maximum number of rows to load

**Returns:**
- `pd.DataFrame`: DataFrame containing packet data

##### `create_train_val_test_split(df, val_size=0.15, test_size=0.15)`
Create stratified train/validation/test splits.

```python
train_df, val_df, test_df = loader.create_train_val_test_split(df)
```

**Parameters:**
- `df` (pd.DataFrame): Input dataframe
- `val_size` (float): Validation set proportion
- `test_size` (float): Test set proportion

**Returns:**
- Tuple of (train_df, val_df, test_df)

##### `get_pytorch_dataset(split='train')`
Get PyTorch dataset for a specific split.

```python
dataset = loader.get_pytorch_dataset(split='train')
```

**Parameters:**
- `split` (str): Data split ('train', 'val', 'test')

**Returns:**
- `PayloadByteDataset`: PyTorch dataset instance

---

### FeatureStore

**Module**: `src.data.feature_store`

Centralized storage system for engineered features with versioning.

#### Class: `FeatureStore`

```python
FeatureStore(store_path, chunk_size=1000)
```

**Parameters:**
- `store_path` (str/Path): Path to HDF5 store file
- `chunk_size` (int): Chunk size for HDF5 datasets

**Methods:**

##### `save_features(name, features, version=None, metadata=None)`
Save features to the store.

```python
store.save_features('packet_embeddings', embeddings, version='1.0')
```

**Parameters:**
- `name` (str): Feature name
- `features` (np.ndarray): Feature array
- `version` (str): Version identifier
- `metadata` (dict): Additional metadata

##### `load_features(name, version=None)`
Load features from the store.

```python
features = store.load_features('packet_embeddings', version='1.0')
```

**Parameters:**
- `name` (str): Feature name
- `version` (str): Version to load (latest if None)

**Returns:**
- `np.ndarray`: Feature array

##### `list_features()`
List all available features.

```python
features_list = store.list_features()
```

**Returns:**
- `list`: List of feature metadata dictionaries

---

### DataVersionManager

**Module**: `src.data.data_versioning`

Manages data versions and tracks lineage.

#### Class: `DataVersionManager`

```python
DataVersionManager(version_dir)
```

**Parameters:**
- `version_dir` (str/Path): Directory for version storage

**Methods:**

##### `create_version(data_path, metadata=None)`
Create a new data version.

```python
version = manager.create_version('data/processed/dataset.csv', 
                                metadata={'split': 'train'})
```

**Parameters:**
- `data_path` (str/Path): Path to dataset
- `metadata` (dict): Version metadata

**Returns:**
- `DataVersion`: Created version object

##### `get_version(version_id)`
Retrieve a specific version.

```python
version = manager.get_version('v1.0')
```

**Parameters:**
- `version_id` (str): Version identifier

**Returns:**
- `DataVersion`: Version object

##### `add_lineage(parent_version, child_version, transformation)`
Track transformation lineage.

```python
manager.add_lineage('v1.0', 'v1.1', 'normalization')
```

**Parameters:**
- `parent_version` (str): Parent version ID
- `child_version` (str): Child version ID
- `transformation` (str): Transformation description

---

## Models Module

### ModelRegistry

**Module**: `src.models.model_registry`

Centralized model registry with MLflow integration.

#### Class: `ModelRegistry`

```python
ModelRegistry(tracking_uri=None, registry_uri=None, config_path='config/mlflow_config.yaml')
```

**Parameters:**
- `tracking_uri` (str): MLflow tracking server URI
- `registry_uri` (str): MLflow model registry URI
- `config_path` (str): Path to MLflow configuration

**Methods:**

##### `register_model(model, model_name, metrics=None, params=None, tags=None)`
Register a model in the registry.

```python
registry.register_model(model, 'vit_malware_detector', 
                       metrics={'accuracy': 0.95})
```

**Parameters:**
- `model`: Model object to register
- `model_name` (str): Name for the model
- `metrics` (dict): Model metrics
- `params` (dict): Model parameters
- `tags` (dict): Additional tags

**Returns:**
- `str`: Model version

##### `load_model(model_name, version=None, stage=None)`
Load a model from the registry.

```python
model = registry.load_model('vit_malware_detector', version='1')
```

**Parameters:**
- `model_name` (str): Model name
- `version` (str): Specific version
- `stage` (str): Model stage ('Production', 'Staging')

**Returns:**
- Model object

##### `transition_model_stage(model_name, version, stage)`
Transition model to a different stage.

```python
registry.transition_model_stage('vit_malware_detector', '1', 'Production')
```

**Parameters:**
- `model_name` (str): Model name
- `version` (str): Model version
- `stage` (str): Target stage

---

## Utils Module

### Logger

**Module**: `src.utils.logger`

Centralized logging configuration.

#### Functions:

##### `setup_logger(name=None)`
Set up logger with configuration.

```python
logger = setup_logger('my_module')
logger.info('Processing started')
```

**Parameters:**
- `name` (str): Logger name

**Returns:**
- `logging.Logger`: Configured logger

##### `get_logger(name)`
Get existing logger instance.

```python
logger = get_logger('my_module')
```

**Parameters:**
- `name` (str): Logger name

**Returns:**
- `logging.Logger`: Logger instance

---

### ConfigManager

**Module**: `src.utils.config_manager`

Configuration management system.

#### Class: `ConfigManager`

```python
ConfigManager(config_dir=None)
```

**Parameters:**
- `config_dir` (str/Path): Configuration directory

**Methods:**

##### `get(config_name, key=None, default=None)`
Get configuration value.

```python
batch_size = config.get('data_config', 'pipeline.batch_size', default=32)
```

**Parameters:**
- `config_name` (str): Configuration name
- `key` (str): Dot-separated path to value
- `default`: Default value if not found

**Returns:**
- Configuration value

##### `set(config_name, key, value)`
Set configuration value.

```python
config.set('data_config', 'pipeline.batch_size', 64)
```

**Parameters:**
- `config_name` (str): Configuration name
- `key` (str): Dot-separated path
- `value`: Value to set

##### `merge_configs(*config_names)`
Merge multiple configurations.

```python
merged = config.merge_configs('base_config', 'experiment_config')
```

**Parameters:**
- `*config_names`: Configuration names to merge

**Returns:**
- `dict`: Merged configuration

#### Function:

##### `get_config_manager(config_dir=None)`
Get singleton config manager instance.

```python
config = get_config_manager()
```

**Parameters:**
- `config_dir` (str/Path): Configuration directory

**Returns:**
- `ConfigManager`: Singleton instance

---

### MetricsCollector

**Module**: `src.utils.metrics_collector`

Prometheus metrics collection for ML models.

#### Class: `MetricsCollector`

```python
MetricsCollector(model_name, port=8000)
```

**Parameters:**
- `model_name` (str): Name of the model
- `port` (int): Port for metrics server

**Methods:**

##### `record_prediction(label, prediction, confidence)`
Record a model prediction.

```python
collector.record_prediction(label=1, prediction=1, confidence=0.95)
```

**Parameters:**
- `label` (int): True label
- `prediction` (int): Predicted label
- `confidence` (float): Prediction confidence

##### `record_batch_metrics(labels, predictions, confidences)`
Record metrics for a batch.

```python
collector.record_batch_metrics(labels, predictions, confidences)
```

**Parameters:**
- `labels` (array): True labels
- `predictions` (array): Predicted labels
- `confidences` (array): Prediction confidences

##### `start_server()`
Start the Prometheus metrics server.

```python
collector.start_server()
```

---

## Common Patterns and Best Practices

### Error Handling

All modules use consistent error handling:

```python
try:
    result = module.operation()
except ModuleError as e:
    logger.error(f"Operation failed: {e}")
    # Handle error appropriately
```

### Configuration

Most modules accept configuration through the ConfigManager:

```python
config = get_config_manager()
module_config = config.get('module_config')
module = Module(**module_config)
```

### Logging

All modules use the centralized logger:

```python
logger = setup_logger(__name__)
logger.info("Starting operation")
```

### Type Hints

All functions include type hints for better IDE support:

```python
def process_data(data: pd.DataFrame, 
                config: Dict[str, Any]) -> Tuple[np.ndarray, List[str]]:
    pass
```

---

## Version Compatibility

- Python: 3.12+
- PyTorch: 2.0+
- NumPy: 1.24+
- Pandas: 2.0+

For detailed examples and tutorials, see the [notebooks directory](../notebooks/).