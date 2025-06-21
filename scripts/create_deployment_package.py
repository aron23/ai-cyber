#!/usr/bin/env python3

import os
import json
import shutil
import argparse
import subprocess
from pathlib import Path
from typing import Dict, Any, List
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DeploymentPackager:
    def __init__(self, model_path: str, model_name: str, version: str):
        self.model_path = Path(model_path)
        self.model_name = model_name
        self.version = version
        self.build_dir = Path("build") / f"{model_name}_{version}"
        self.dist_dir = Path("dist")
        
    def create_package_structure(self):
        # Clean and create directories
        if self.build_dir.exists():
            shutil.rmtree(self.build_dir)
        
        self.build_dir.mkdir(parents=True, exist_ok=True)
        self.dist_dir.mkdir(exist_ok=True)
        
        # Create package subdirectories
        (self.build_dir / "model").mkdir()
        (self.build_dir / "src").mkdir()
        (self.build_dir / "configs").mkdir()
        (self.build_dir / "scripts").mkdir()
    
    def copy_model_files(self):
        # Copy model
        shutil.copy2(self.model_path, self.build_dir / "model" / "model.pth")
        
        # Copy inference code
        inference_files = [
            "src/inference/api.py",
            "src/inference/__init__.py",
            "src/utils/preprocessing.py",
            "src/utils/postprocessing.py"
        ]
        
        for file_path in inference_files:
            src_path = Path(file_path)
            if src_path.exists():
                dest_path = self.build_dir / file_path
                dest_path.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(src_path, dest_path)
    
    def create_requirements(self):
        requirements = [
            "torch==1.12.0",
            "numpy==1.22.0",
            "fastapi==0.95.0",
            "uvicorn==0.21.0",
            "pydantic==1.10.0",
            "python-multipart==0.0.6",
            "prometheus-client==0.16.0",
            "psutil==5.9.0"
        ]
        
        req_path = self.build_dir / "requirements.txt"
        with open(req_path, 'w') as f:
            f.write('\n'.join(requirements))
    
    def create_dockerfile(self):
        dockerfile_content = f'''FROM python:3.9-slim

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    libgomp1 \\
    && rm -rf /var/lib/apt/lists/*

# Set working directory
WORKDIR /app

# Copy requirements and install dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application files
COPY model/ ./model/
COPY src/ ./src/
COPY configs/ ./configs/
COPY scripts/ ./scripts/

# Set environment variables
ENV MODEL_PATH=/app/model/model.pth
ENV MODEL_NAME={self.model_name}
ENV MODEL_VERSION={self.version}
ENV PYTHONPATH=/app

# Expose port
EXPOSE 8000

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \\
    CMD python -c "import requests; requests.get('http://localhost:8000/health')"

# Run the application
CMD ["uvicorn", "src.inference.api:app", "--host", "0.0.0.0", "--port", "8000"]
'''
        
        dockerfile_path = self.build_dir / "Dockerfile"
        with open(dockerfile_path, 'w') as f:
            f.write(dockerfile_content)
    
    def create_deployment_config(self):
        config = {
            "model_name": self.model_name,
            "version": self.version,
            "runtime": {
                "python_version": "3.9",
                "framework": "pytorch",
                "framework_version": "1.12.0"
            },
            "resources": {
                "cpu": "2",
                "memory": "4Gi",
                "gpu": "0"
            },
            "scaling": {
                "min_replicas": 1,
                "max_replicas": 3,
                "target_cpu_utilization": 80
            },
            "health_check": {
                "path": "/health",
                "interval_seconds": 30,
                "timeout_seconds": 10
            }
        }
        
        config_path = self.build_dir / "configs" / "deployment.json"
        with open(config_path, 'w') as f:
            json.dump(config, f, indent=2)
    
    def create_startup_script(self):
        script_content = '''#!/bin/bash

# Startup script for model deployment

# Set environment variables
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

# Log startup
echo "Starting model server..."
echo "Model: $MODEL_NAME"
echo "Version: $MODEL_VERSION"

# Start the API server
exec uvicorn src.inference.api:app --host 0.0.0.0 --port 8000 --workers 1
'''
        
        script_path = self.build_dir / "scripts" / "startup.sh"
        with open(script_path, 'w') as f:
            f.write(script_content)
        
        # Make executable
        script_path.chmod(0o755)
    
    def create_python_wheel(self):
        # Create setup.py
        setup_content = f'''from setuptools import setup, find_packages

setup(
    name="{self.model_name}_model",
    version="{self.version}",
    packages=find_packages(),
    install_requires=[
        "torch>=1.12.0",
        "numpy>=1.22.0",
    ],
    package_data={{
        "": ["*.pth", "*.json", "*.yaml"],
    }},
    python_requires=">=3.8",
    author="ML Team",
    description="Deployment package for {self.model_name} model",
)
'''
        
        setup_path = self.build_dir / "setup.py"
        with open(setup_path, 'w') as f:
            f.write(setup_content)
        
        # Build wheel
        logger.info("Building Python wheel...")
        subprocess.run(
            ["python", "setup.py", "bdist_wheel"],
            cwd=self.build_dir,
            capture_output=True
        )
        
        # Copy wheel to dist
        wheel_dir = self.build_dir / "dist"
        if wheel_dir.exists():
            for wheel_file in wheel_dir.glob("*.whl"):
                shutil.copy2(wheel_file, self.dist_dir)
    
    def build_docker_image(self):
        image_tag = f"{self.model_name}:{self.version}"
        
        logger.info(f"Building Docker image: {image_tag}")
        
        # Build command
        build_cmd = [
            "docker", "build",
            "-t", image_tag,
            "-f", str(self.build_dir / "Dockerfile"),
            str(self.build_dir)
        ]
        
        # Note: In CI environment, this would actually run
        # For now, we'll just log the command
        logger.info(f"Would run: {' '.join(build_cmd)}")
        
        # Also create a docker-compose file
        compose_content = f'''version: '3.8'

services:
  {self.model_name}:
    image: {image_tag}
    ports:
      - "8000:8000"
    environment:
      - MODEL_NAME={self.model_name}
      - MODEL_VERSION={self.version}
    deploy:
      resources:
        limits:
          cpus: '2'
          memory: 4G
    restart: unless-stopped
    healthcheck:
      test: ["CMD", "curl", "-f", "http://localhost:8000/health"]
      interval: 30s
      timeout: 10s
      retries: 3
'''
        
        compose_path = self.dist_dir / f"docker-compose-{self.model_name}.yml"
        with open(compose_path, 'w') as f:
            f.write(compose_content)
    
    def create_tarball(self):
        tarball_name = f"{self.model_name}_{self.version}.tar.gz"
        tarball_path = self.dist_dir / tarball_name
        
        logger.info(f"Creating deployment package: {tarball_path}")
        
        # Create tarball
        import tarfile
        with tarfile.open(tarball_path, "w:gz") as tar:
            tar.add(self.build_dir, arcname=f"{self.model_name}_{self.version}")
        
        return tarball_path
    
    def create_manifest(self):
        manifest = {
            "package_info": {
                "name": self.model_name,
                "version": self.version,
                "created_at": subprocess.check_output(['date', '-Iseconds']).decode().strip()
            },
            "contents": {
                "model": "model/model.pth",
                "dockerfile": "Dockerfile",
                "requirements": "requirements.txt",
                "deployment_config": "configs/deployment.json",
                "startup_script": "scripts/startup.sh"
            },
            "artifacts": {
                "docker_image": f"{self.model_name}:{self.version}",
                "python_wheel": f"{self.model_name}_model-{self.version}-py3-none-any.whl",
                "tarball": f"{self.model_name}_{self.version}.tar.gz"
            }
        }
        
        manifest_path = self.dist_dir / f"{self.model_name}_{self.version}_manifest.json"
        with open(manifest_path, 'w') as f:
            json.dump(manifest, f, indent=2)
        
        return manifest_path
    
    def build(self):
        logger.info(f"Creating deployment package for {self.model_name} v{self.version}")
        
        # Create package structure
        self.create_package_structure()
        
        # Copy necessary files
        self.copy_model_files()
        
        # Create configuration files
        self.create_requirements()
        self.create_dockerfile()
        self.create_deployment_config()
        self.create_startup_script()
        
        # Build artifacts
        self.create_python_wheel()
        self.build_docker_image()
        tarball_path = self.create_tarball()
        manifest_path = self.create_manifest()
        
        logger.info("Deployment package created successfully!")
        logger.info(f"Artifacts in {self.dist_dir}:")
        for artifact in self.dist_dir.glob("*"):
            logger.info(f"  - {artifact.name}")
        
        return {
            "tarball": str(tarball_path),
            "manifest": str(manifest_path),
            "docker_image": f"{self.model_name}:{self.version}"
        }


def main():
    parser = argparse.ArgumentParser(description="Create deployment package for ML model")
    parser.add_argument("--model-path", required=True, help="Path to the model file")
    parser.add_argument("--model-name", required=True, help="Name of the model")
    parser.add_argument("--version", required=True, help="Version of the model")
    
    args = parser.parse_args()
    
    packager = DeploymentPackager(
        model_path=args.model_path,
        model_name=args.model_name,
        version=args.version
    )
    
    results = packager.build()
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()