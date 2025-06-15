# Data Directory

This directory contains all datasets and data processing artifacts for the SMS/Email Spam Filter project.

## Directory Structure

```
data/
├── SMSSPamCollection          # Raw dataset (5,574 messages)
├── processed/                 # Processed datasets (excluded from git)
│   ├── train.csv             # Training set (80%)
│   ├── validation.csv        # Validation set (10%)
│   └── test.csv              # Test set (10%)
├── cache/                    # Cached feature extractions (excluded from git)
├── temp/                     # Temporary processing files (excluded from git)
└── backup/                   # Data backups (excluded from git)
```

## Raw Dataset

**File**: `SMSSPamCollection`
- **Format**: Tab-separated values (TSV)
- **Structure**: `label\tmessage`
- **Size**: 467KB, 5,575 lines
- **Labels**: 
  - `ham`: Legitimate messages (4,827 messages, 86.6%)
  - `spam`: Spam messages (747 messages, 13.4%)

## Data Processing Notes

- All processed datasets are excluded from version control due to size
- Original dataset is tracked for reproducibility
- Processing configuration is defined in `config/data_config.py`
- Use stratified sampling to maintain class balance across splits

## Usage

```python
# Load raw dataset
from config.data_config import DATASET_CONFIG
import pandas as pd

# Read the raw data
df = pd.read_csv(
    DATASET_CONFIG['raw_data_path'], 
    sep='\t', 
    names=['label', 'message'],
    encoding=DATASET_CONFIG['encoding']
)
```

## Data Quality Checks

- Verify message count matches expected 5,574
- Check for missing values
- Validate label distribution
- Ensure proper encoding (UTF-8)
- Monitor for data drift in production 