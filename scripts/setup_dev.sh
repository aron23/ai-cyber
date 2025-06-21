#!/bin/bash

# Setup script for ML development environment
set -e

echo "🚀 Setting up ML development environment..."

# Create necessary directories
echo "Creating project directories..."
mkdir -p data/{raw,processed,external}
mkdir -p models/{experiments,production}
mkdir -p notebooks/{research,analysis,experiments}
mkdir -p src/{data,features,models,utils,deployment}
mkdir -p logs
mkdir -p tests/{unit,integration}
mkdir -p monitoring/{prometheus,grafana/{dashboards,datasources}}
mkdir -p vertex_ai
mkdir -p mlruns mlartifacts

# Create .env file if it doesn't exist
if [ ! -f .env ]; then
    echo "Creating .env file..."
    cat > .env << EOF
# Environment Configuration
ENVIRONMENT=development
PYTHONPATH=/workspace

# MLflow Configuration
MLFLOW_TRACKING_URI=http://localhost:5000
MLFLOW_EXPERIMENT_NAME=default

# API Configuration
API_HOST=0.0.0.0
API_PORT=8080

# Jupyter Configuration
JUPYTER_PORT=8888

# Database Configuration
POSTGRES_USER=mlflow
POSTGRES_PASSWORD=mlflow
POSTGRES_DB=mlflow
POSTGRES_HOST=postgres
POSTGRES_PORT=5432

# Monitoring
PROMETHEUS_PORT=9090
GRAFANA_PORT=3000
EOF
fi

# Create .gitignore if it doesn't exist
if [ ! -f .gitignore ]; then
    echo "Creating .gitignore file..."
    cat > .gitignore << EOF
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
env/
venv/
ENV/
.venv
pip-log.txt
pip-delete-this-directory.txt
.pytest_cache/
*.egg-info/
.coverage
htmlcov/
.tox/
.mypy_cache/
.ruff_cache/

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# ML/Data
data/raw/*
data/processed/*
data/external/*
!data/*/.gitkeep
models/experiments/*
models/production/*
!models/*/.gitkeep
mlruns/
mlartifacts/
*.h5
*.pkl
*.joblib
*.onnx
*.pt
*.pth

# IDE
.vscode/
.idea/
*.swp
*.swo
*~

# Environment
.env
.env.local
.env.*.local

# Logs
logs/
*.log

# Docker
*.pid

# OS
.DS_Store
Thumbs.db

# DVC
.dvc/cache
.dvc/tmp
EOF
fi

# Create Prometheus configuration
echo "Creating Prometheus configuration..."
cat > monitoring/prometheus.yml << EOF
global:
  scrape_interval: 15s
  evaluation_interval: 15s

scrape_configs:
  - job_name: 'mlflow'
    static_configs:
      - targets: ['mlflow:5000']
  
  - job_name: 'api'
    static_configs:
      - targets: ['dev-env:8080']
    metrics_path: '/metrics'
  
  - job_name: 'jupyter'
    static_configs:
      - targets: ['jupyter:8888']
EOF

# Create Grafana datasource configuration
echo "Creating Grafana datasource configuration..."
cat > monitoring/grafana/datasources/prometheus.yml << EOF
apiVersion: 1

datasources:
  - name: Prometheus
    type: prometheus
    access: proxy
    url: http://prometheus:9090
    isDefault: true
    editable: true
EOF

# Create requirements.txt if it doesn't exist
if [ ! -f requirements.txt ]; then
    echo "Creating requirements.txt..."
    cat > requirements.txt << EOF
# Core ML Libraries
numpy==1.24.3
pandas==2.0.3
scikit-learn==1.3.0
matplotlib==3.7.2
seaborn==0.12.2

# Deep Learning
torch==2.0.1
torchvision==0.15.2
tensorflow==2.13.0
keras==2.13.1

# MLOps
mlflow==2.7.1
dvc==3.20.0
wandb==0.15.8
optuna==3.3.0

# Development Tools
black==23.7.0
flake8==6.1.0
mypy==1.5.1
pytest==7.4.0
pytest-cov==4.1.0
pre-commit==3.3.3
isort==5.12.0
pylint==2.17.5

# API Development
fastapi==0.103.0
uvicorn==0.23.2
pydantic==2.3.0

# Utilities
requests==2.31.0
pyyaml==6.0.1
python-dotenv==1.0.0
tqdm==4.66.1
click==8.1.6

# Jupyter
jupyter==1.0.0
jupyterlab==4.0.3
notebook==7.0.0
ipykernel==6.25.0
EOF
fi

# Make scripts executable
chmod +x scripts/*.sh 2>/dev/null || true

# Build Docker images
echo "Building Docker images..."
docker-compose build

# Initialize git if not already initialized
if [ ! -d .git ]; then
    echo "Initializing git repository..."
    git init
    git add .
    git commit -m "Initial project setup"
fi

echo "✅ Development environment setup complete!"
echo ""
echo "To start the development environment, run:"
echo "  docker-compose up -d"
echo ""
echo "Access services at:"
echo "  - Jupyter Lab: http://localhost:8888"
echo "  - MLflow: http://localhost:5000"
echo "  - Grafana: http://localhost:3000 (admin/admin)"
echo "  - Prometheus: http://localhost:9090"
echo ""
echo "To enter the development container:"
echo "  docker-compose exec dev-env bash"