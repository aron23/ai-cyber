# Project Directory Structure

```
vit-network-traffic-analysis/
│
├── notebooks/                    # Educational Jupyter notebooks
│   ├── 01_environment_setup.ipynb
│   ├── 02_data_exploration.ipynb
│   ├── 03_data_pipeline.ipynb
│   ├── 04_image_encoding_strategies.ipynb
│   ├── 05_feature_enhancement.ipynb
│   ├── 06_vit_architecture.ipynb
│   ├── 07_supervised_training.ipynb
│   ├── 08_self_supervised_training.ipynb
│   ├── 09_few_shot_learning.ipynb
│   ├── 10_novel_class_detection.ipynb
│   ├── 11_performance_optimization.ipynb
│   ├── 12_evaluation_suite.ipynb
│   ├── 13_production_pipeline.ipynb
│   ├── 14_model_deployment.ipynb
│   └── 15_tutorial_complete.ipynb
│
├── src/                         # Source code modules
│   ├── __init__.py
│   ├── data/
│   │   ├── __init__.py
│   │   ├── data_loader.py      # GCS data loading utilities
│   │   ├── preprocessor.py     # Data preprocessing
│   │   └── packet_to_image.py  # Image encoding logic
│   │
│   ├── models/
│   │   ├── __init__.py
│   │   ├── vit_packet.py       # ViT architecture
│   │   ├── patch_embed.py      # Custom patch embedding
│   │   └── position_embed.py   # Position encoding
│   │
│   ├── training/
│   │   ├── __init__.py
│   │   ├── trainer.py          # Main training loop
│   │   ├── losses.py           # Custom loss functions
│   │   └── metrics.py          # Evaluation metrics
│   │
│   ├── few_shot/
│   │   ├── __init__.py
│   │   ├── prototypical_net.py # Prototypical networks
│   │   ├── maml.py             # MAML implementation
│   │   └── episode_sampler.py  # Episode sampling
│   │
│   ├── inference/
│   │   ├── __init__.py
│   │   ├── predictor.py        # Inference pipeline
│   │   └── api.py              # REST API endpoints
│   │
│   └── utils/
│       ├── __init__.py
│       ├── visualization.py    # Plotting utilities
│       ├── config.py           # Configuration management
│       └── logging.py          # Logging setup
│
├── configs/                     # Configuration files
│   ├── data_config.yaml
│   ├── model_config.yaml
│   ├── training_config.yaml
│   └── deployment_config.yaml
│
├── experiments/                 # Experiment tracking
│   ├── runs/                   # Individual experiment runs
│   └── best_models/            # Best performing models
│
├── tests/                      # Unit tests
│   ├── test_data_loader.py
│   ├── test_packet_to_image.py
│   ├── test_vit_model.py
│   └── test_few_shot.py
│
├── scripts/                    # Utility scripts
│   ├── setup_gcp.sh           # GCP setup script
│   ├── download_data.py       # Data download script
│   ├── train_model.py         # Training script
│   └── deploy_model.py        # Deployment script
│
├── docs/                       # Documentation
│   ├── architecture.md        # System architecture
│   ├── api_reference.md       # API documentation
│   ├── deployment_guide.md    # Deployment instructions
│   └── troubleshooting.md     # Common issues
│
├── data/                       # Local data cache (gitignored)
│   ├── raw/                   # Raw packet data
│   ├── processed/             # Processed images
│   └── features/              # Extracted features
│
├── models/                     # Saved models (gitignored)
│   ├── checkpoints/           # Training checkpoints
│   ├── production/            # Production-ready models
│   └── experiments/           # Experimental models
│
├── outputs/                    # Results and visualizations
│   ├── figures/               # Generated plots
│   ├── reports/               # Evaluation reports
│   └── logs/                  # Training logs
│
├── deployment/                 # Deployment artifacts
│   ├── docker/
│   │   ├── Dockerfile         # Container definition
│   │   └── requirements.txt   # Container dependencies
│   ├── kubernetes/
│   │   ├── deployment.yaml    # K8s deployment
│   │   └── service.yaml       # K8s service
│   └── vertex-ai/
│       ├── endpoint_config.json
│       └── predictor.py
│
├── .gitignore
├── README.md
├── requirements.txt            # Project dependencies
├── setup.py                   # Package setup
└── LICENSE
```

## Key Directories Explained

### `/notebooks`
Contains all educational Jupyter notebooks numbered in order of execution. Each notebook is self-contained with markdown explanations.

### `/src`
Production-ready Python modules organized by functionality:
- `data/`: Data loading and preprocessing
- `models/`: Model architectures
- `training/`: Training utilities
- `few_shot/`: Few-shot learning components
- `inference/`: Prediction and API serving

### `/configs`
YAML configuration files for different components, enabling easy hyperparameter tuning and experiment management.

### `/experiments`
MLflow or Weights & Biases experiment tracking, storing run metadata and artifacts.

### `/deployment`
Everything needed for production deployment including Docker containers and Kubernetes manifests.

## File Naming Conventions

- **Notebooks**: `{number}_{descriptive_name}.ipynb`
- **Python modules**: `{functionality}_{component}.py`
- **Configs**: `{component}_config.yaml`
- **Models**: `{model_type}_{version}_{timestamp}.pt` 