# Models Directory

This directory contains trained models, model artifacts, and related files for the SMS/Email Spam Filter project.

## Directory Structure

```
models/
├── baseline/                  # Baseline models (Naive Bayes, Logistic Regression)
├── advanced/                  # Advanced models (ensemble, deep learning)
├── production/                # Production-ready models
├── experiments/               # Experimental models
├── checkpoints/               # Training checkpoints (excluded from git)
├── cache/                     # Model cache files (excluded from git)
└── metadata/                  # Model metadata and performance logs
```

## Model Naming Convention

Models should follow this naming pattern:
```
{model_type}_{version}_{performance_metric}_{date}.{extension}
```

Examples:
- `naive_bayes_v1_f1_0.85_20250615.pkl`
- `logistic_regression_v2_f1_0.89_20250618.joblib`
- `ensemble_v1_f1_0.92_20250625.pkl`

## Performance Targets

All models must meet these minimum requirements:
- **Precision**: ≥ 92%
- **Recall**: ≥ 88%
- **F1-Score**: ≥ 90%
- **Inference Time**: < 50ms per message

## Model Storage

- **Small models** (< 100MB): Stored in git with Git LFS
- **Large models** (> 100MB): Excluded from git, stored in external storage
- **Metadata**: Always tracked in git for reproducibility

## Model Metadata

Each model should include:
- Performance metrics (precision, recall, F1-score)
- Training parameters and hyperparameters
- Feature engineering details
- Training data version
- Inference time benchmarks
- Model size and memory requirements

## Usage

```python
# Load a production model
import joblib
from pathlib import Path

model_path = Path('models/production/logistic_regression_v2_f1_0.89_20250618.joblib')
model = joblib.load(model_path)

# Make predictions
predictions = model.predict(X_test)
```

## Model Versioning

- Use semantic versioning (v1.0.0, v1.1.0, etc.)
- Tag major performance improvements
- Keep model lineage documentation
- Archive old models after validation

## Deployment Notes

- Production models must pass all validation tests
- Include model signature and input/output schemas
- Monitor model performance in production
- Implement A/B testing for model updates 