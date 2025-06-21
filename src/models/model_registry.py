"""Model Registry for managing ML models with MLflow."""

import json
import logging
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple, Union

import mlflow
import pandas as pd
import yaml
from mlflow.entities import ViewType
from mlflow.exceptions import MlflowException
from mlflow.models import Model
from mlflow.tracking import MlflowClient
from pydantic import BaseModel, Field


class ModelMetadata(BaseModel):
    """Model metadata schema."""

    name: str
    version: str
    stage: str = "None"
    description: Optional[str] = None
    tags: Dict[str, str] = Field(default_factory=dict)
    metrics: Dict[str, float] = Field(default_factory=dict)
    params: Dict[str, Any] = Field(default_factory=dict)
    dataset_info: Optional[Dict[str, Any]] = None
    training_info: Optional[Dict[str, Any]] = None
    created_by: Optional[str] = None
    created_at: Optional[datetime] = None


class ModelRegistry:
    """Centralized model registry for managing ML models."""

    def __init__(
        self,
        tracking_uri: Optional[str] = None,
        registry_uri: Optional[str] = None,
        config_path: Optional[str] = "config/mlflow_config.yaml",
    ):
        """Initialize ModelRegistry.

        Args:
            tracking_uri: MLflow tracking server URI
            registry_uri: MLflow model registry URI
            config_path: Path to MLflow configuration file
        """
        self.logger = logging.getLogger(__name__)
        
        # Load configuration
        self.config = self._load_config(config_path)
        
        # Set up MLflow
        self.tracking_uri = tracking_uri or self.config["mlflow"]["tracking"]["uri"]
        self.registry_uri = registry_uri or self.tracking_uri
        
        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_registry_uri(self.registry_uri)
        
        self.client = MlflowClient(
            tracking_uri=self.tracking_uri,
            registry_uri=self.registry_uri
        )
        
        self.logger.info(f"ModelRegistry initialized with tracking URI: {self.tracking_uri}")

    def _load_config(self, config_path: str) -> Dict[str, Any]:
        """Load configuration from YAML file."""
        if config_path and Path(config_path).exists():
            with open(config_path, "r") as f:
                return yaml.safe_load(f)
        return {"mlflow": {"tracking": {"uri": "http://localhost:5000"}}}

    def register_model(
        self,
        model: Any,
        model_name: str,
        metrics: Dict[str, float],
        params: Optional[Dict[str, Any]] = None,
        metadata: Optional[ModelMetadata] = None,
        artifact_path: str = "model",
        conda_env: Optional[Dict[str, Any]] = None,
        code_paths: Optional[List[str]] = None,
        registered_model_name: Optional[str] = None,
        await_registration_for: int = 300,
    ) -> Tuple[str, str]:
        """Register a model with MLflow.

        Args:
            model: The model object to register
            model_name: Name for the model
            metrics: Model performance metrics
            params: Model parameters
            metadata: Additional model metadata
            artifact_path: Path to save model artifacts
            conda_env: Conda environment specification
            code_paths: List of code paths to include
            registered_model_name: Name for the registered model
            await_registration_for: Time to wait for registration (seconds)

        Returns:
            Tuple of (run_id, model_version)
        """
        try:
            # Start MLflow run
            with mlflow.start_run() as run:
                run_id = run.info.run_id
                
                # Log parameters
                if params:
                    mlflow.log_params(params)
                
                # Log metrics
                mlflow.log_metrics(metrics)
                
                # Log metadata as tags
                if metadata:
                    tags = {
                        "model_name": model_name,
                        "created_by": metadata.created_by or "unknown",
                        "stage": metadata.stage,
                    }
                    if metadata.description:
                        tags["description"] = metadata.description
                    mlflow.set_tags(tags)
                
                # Log model
                if hasattr(model, "save_pretrained"):  # Hugging Face models
                    mlflow.transformers.log_model(
                        transformers_model=model,
                        artifact_path=artifact_path,
                        conda_env=conda_env,
                        code_paths=code_paths,
                    )
                elif hasattr(model, "save"):  # Keras/TensorFlow models
                    mlflow.tensorflow.log_model(
                        tf_saved_model_dir=model,
                        artifact_path=artifact_path,
                        conda_env=conda_env,
                        code_paths=code_paths,
                    )
                elif hasattr(model, "state_dict"):  # PyTorch models
                    mlflow.pytorch.log_model(
                        pytorch_model=model,
                        artifact_path=artifact_path,
                        conda_env=conda_env,
                        code_paths=code_paths,
                    )
                else:  # Sklearn and other models
                    mlflow.sklearn.log_model(
                        sk_model=model,
                        artifact_path=artifact_path,
                        conda_env=conda_env,
                        code_paths=code_paths,
                    )
                
                # Register model
                if registered_model_name:
                    model_uri = f"runs:/{run_id}/{artifact_path}"
                    model_version = mlflow.register_model(
                        model_uri=model_uri,
                        name=registered_model_name,
                        await_registration_for=await_registration_for,
                    )
                    
                    self.logger.info(
                        f"Model registered: {registered_model_name} "
                        f"version {model_version.version}"
                    )
                    
                    return run_id, model_version.version
                
                return run_id, "unregistered"
                
        except Exception as e:
            self.logger.error(f"Error registering model: {str(e)}")
            raise

    def get_model(
        self,
        name: str,
        version: Optional[str] = None,
        stage: Optional[str] = None,
    ) -> Any:
        """Retrieve a model from the registry.

        Args:
            name: Registered model name
            version: Model version (if not specified, gets latest)
            stage: Model stage (Production, Staging, etc.)

        Returns:
            Loaded model object
        """
        try:
            if version:
                model_uri = f"models:/{name}/{version}"
            elif stage:
                model_uri = f"models:/{name}/{stage}"
            else:
                # Get latest version
                model_uri = f"models:/{name}/latest"
            
            # Load model based on flavor
            model = mlflow.pyfunc.load_model(model_uri)
            
            self.logger.info(f"Loaded model: {model_uri}")
            return model
            
        except Exception as e:
            self.logger.error(f"Error loading model: {str(e)}")
            raise

    def promote_model(
        self,
        name: str,
        version: str,
        stage: str,
        archive_existing: bool = True,
    ) -> None:
        """Promote a model to a new stage.

        Args:
            name: Registered model name
            version: Model version to promote
            stage: Target stage (Staging, Production, Archived)
            archive_existing: Whether to archive existing models in target stage
        """
        try:
            # Validate stage
            valid_stages = ["None", "Staging", "Production", "Archived"]
            if stage not in valid_stages:
                raise ValueError(f"Invalid stage: {stage}. Must be one of {valid_stages}")
            
            # Archive existing models in target stage if requested
            if archive_existing and stage in ["Staging", "Production"]:
                for mv in self.client.search_model_versions(f"name='{name}'"):
                    if mv.current_stage == stage:
                        self.client.transition_model_version_stage(
                            name=name,
                            version=mv.version,
                            stage="Archived",
                        )
                        self.logger.info(
                            f"Archived model {name} version {mv.version} "
                            f"from {stage}"
                        )
            
            # Promote model to new stage
            self.client.transition_model_version_stage(
                name=name,
                version=version,
                stage=stage,
            )
            
            self.logger.info(f"Promoted model {name} version {version} to {stage}")
            
        except Exception as e:
            self.logger.error(f"Error promoting model: {str(e)}")
            raise

    def list_models(
        self,
        filter_string: Optional[str] = None,
        max_results: int = 100,
    ) -> pd.DataFrame:
        """List all registered models.

        Args:
            filter_string: Filter string for searching models
            max_results: Maximum number of results to return

        Returns:
            DataFrame with model information
        """
        try:
            models = self.client.search_registered_models(
                filter_string=filter_string,
                max_results=max_results,
            )
            
            data = []
            for model in models:
                for version in model.latest_versions:
                    data.append({
                        "name": model.name,
                        "version": version.version,
                        "stage": version.current_stage,
                        "description": model.description,
                        "last_updated": version.last_updated_timestamp,
                        "run_id": version.run_id,
                        "status": version.status,
                    })
            
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.error(f"Error listing models: {str(e)}")
            raise

    def get_model_metrics(
        self,
        name: str,
        version: str,
    ) -> Dict[str, float]:
        """Get metrics for a specific model version.

        Args:
            name: Registered model name
            version: Model version

        Returns:
            Dictionary of metrics
        """
        try:
            # Get model version info
            model_version = self.client.get_model_version(name=name, version=version)
            
            # Get run metrics
            run = self.client.get_run(model_version.run_id)
            
            return run.data.metrics
            
        except Exception as e:
            self.logger.error(f"Error getting model metrics: {str(e)}")
            raise

    def compare_models(
        self,
        model_names: List[str],
        versions: Optional[List[str]] = None,
        metrics: Optional[List[str]] = None,
    ) -> pd.DataFrame:
        """Compare metrics across multiple models.

        Args:
            model_names: List of model names to compare
            versions: List of versions (if None, uses latest)
            metrics: List of metrics to compare (if None, uses all)

        Returns:
            DataFrame with model comparison
        """
        try:
            data = []
            
            for i, name in enumerate(model_names):
                version = versions[i] if versions else "latest"
                
                if version == "latest":
                    model = self.client.get_registered_model(name)
                    version = model.latest_versions[-1].version
                
                model_metrics = self.get_model_metrics(name, version)
                
                row = {
                    "model_name": name,
                    "version": version,
                }
                
                if metrics:
                    row.update({m: model_metrics.get(m, None) for m in metrics})
                else:
                    row.update(model_metrics)
                
                data.append(row)
            
            return pd.DataFrame(data)
            
        except Exception as e:
            self.logger.error(f"Error comparing models: {str(e)}")
            raise

    def delete_model_version(
        self,
        name: str,
        version: str,
    ) -> None:
        """Delete a specific model version.

        Args:
            name: Registered model name
            version: Model version to delete
        """
        try:
            self.client.delete_model_version(name=name, version=version)
            self.logger.info(f"Deleted model {name} version {version}")
            
        except Exception as e:
            self.logger.error(f"Error deleting model version: {str(e)}")
            raise

    def export_model_info(
        self,
        name: str,
        version: str,
        output_path: str,
    ) -> None:
        """Export model information to JSON file.

        Args:
            name: Registered model name
            version: Model version
            output_path: Path to save model info
        """
        try:
            # Get model version info
            model_version = self.client.get_model_version(name=name, version=version)
            
            # Get run info
            run = self.client.get_run(model_version.run_id)
            
            # Compile model info
            model_info = {
                "name": name,
                "version": version,
                "stage": model_version.current_stage,
                "run_id": model_version.run_id,
                "created_timestamp": model_version.creation_timestamp,
                "last_updated_timestamp": model_version.last_updated_timestamp,
                "description": model_version.description,
                "metrics": run.data.metrics,
                "params": run.data.params,
                "tags": run.data.tags,
            }
            
            # Save to file
            with open(output_path, "w") as f:
                json.dump(model_info, f, indent=2, default=str)
            
            self.logger.info(f"Exported model info to {output_path}")
            
        except Exception as e:
            self.logger.error(f"Error exporting model info: {str(e)}")
            raise