import os
import json
import shutil
import hashlib
import tarfile
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Any
import logging
from dataclasses import dataclass, asdict

logger = logging.getLogger(__name__)


@dataclass
class ModelArtifact:
    name: str
    version: str
    model_path: str
    metadata: Dict[str, Any]
    created_at: str
    size_bytes: int
    checksum: str
    dependencies: List[str]
    tags: List[str] = None
    
    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class ArtifactManager:
    def __init__(self, artifact_store_path: str = "artifact_store"):
        self.artifact_store = Path(artifact_store_path)
        self.artifact_store.mkdir(exist_ok=True)
        self.metadata_file = self.artifact_store / "artifacts_metadata.json"
        self.artifacts: Dict[str, ModelArtifact] = {}
        self._load_metadata()
    
    def _load_metadata(self):
        if self.metadata_file.exists():
            with open(self.metadata_file, 'r') as f:
                data = json.load(f)
                for key, artifact_data in data.items():
                    self.artifacts[key] = ModelArtifact(**artifact_data)
    
    def _save_metadata(self):
        data = {
            key: artifact.to_dict() 
            for key, artifact in self.artifacts.items()
        }
        with open(self.metadata_file, 'w') as f:
            json.dump(data, f, indent=2)
    
    def _calculate_checksum(self, file_path: str) -> str:
        sha256_hash = hashlib.sha256()
        with open(file_path, "rb") as f:
            for byte_block in iter(lambda: f.read(4096), b""):
                sha256_hash.update(byte_block)
        return sha256_hash.hexdigest()
    
    def _get_file_size(self, file_path: str) -> int:
        return os.path.getsize(file_path)
    
    def package_model(self, model_name: str, version: str, 
                     model_path: str, metadata: Dict[str, Any],
                     dependencies_file: str = None,
                     additional_files: List[str] = None,
                     tags: List[str] = None) -> ModelArtifact:
        logger.info(f"Packaging model {model_name} version {version}")
        
        # Create version directory
        version_dir = self.artifact_store / model_name / version
        version_dir.mkdir(parents=True, exist_ok=True)
        
        # Copy model file
        model_filename = Path(model_path).name
        dest_model_path = version_dir / model_filename
        shutil.copy2(model_path, dest_model_path)
        
        # Load dependencies
        dependencies = []
        if dependencies_file and Path(dependencies_file).exists():
            with open(dependencies_file, 'r') as f:
                dependencies = [line.strip() for line in f if line.strip()]
        
        # Create package archive
        package_path = version_dir / f"{model_name}_{version}.tar.gz"
        with tarfile.open(package_path, "w:gz") as tar:
            # Add model
            tar.add(dest_model_path, arcname=model_filename)
            
            # Add dependencies file
            if dependencies_file:
                tar.add(dependencies_file, arcname="requirements.txt")
            
            # Add additional files
            if additional_files:
                for file_path in additional_files:
                    if Path(file_path).exists():
                        tar.add(file_path, arcname=Path(file_path).name)
            
            # Add metadata
            metadata_content = json.dumps({
                "model_name": model_name,
                "version": version,
                "metadata": metadata,
                "dependencies": dependencies,
                "created_at": datetime.now().isoformat(),
                "tags": tags or []
            }, indent=2)
            
            metadata_info = tarfile.TarInfo(name="metadata.json")
            metadata_info.size = len(metadata_content.encode())
            tar.addfile(metadata_info, fileobj=os.BytesIO(metadata_content.encode()))
        
        # Create artifact record
        artifact = ModelArtifact(
            name=model_name,
            version=version,
            model_path=str(package_path),
            metadata=metadata,
            created_at=datetime.now().isoformat(),
            size_bytes=self._get_file_size(package_path),
            checksum=self._calculate_checksum(package_path),
            dependencies=dependencies,
            tags=tags or []
        )
        
        # Store artifact
        artifact_key = f"{model_name}:{version}"
        self.artifacts[artifact_key] = artifact
        self._save_metadata()
        
        logger.info(f"Model packaged successfully: {package_path}")
        return artifact
    
    def get_artifact(self, model_name: str, version: str = None) -> Optional[ModelArtifact]:
        if version:
            artifact_key = f"{model_name}:{version}"
            return self.artifacts.get(artifact_key)
        else:
            # Get latest version
            matching_artifacts = [
                (key, artifact) for key, artifact in self.artifacts.items()
                if artifact.name == model_name
            ]
            if matching_artifacts:
                # Sort by created_at timestamp
                matching_artifacts.sort(key=lambda x: x[1].created_at, reverse=True)
                return matching_artifacts[0][1]
            return None
    
    def list_artifacts(self, model_name: str = None, 
                      tags: List[str] = None) -> List[ModelArtifact]:
        artifacts = list(self.artifacts.values())
        
        # Filter by model name
        if model_name:
            artifacts = [a for a in artifacts if a.name == model_name]
        
        # Filter by tags
        if tags:
            artifacts = [
                a for a in artifacts 
                if a.tags and any(tag in a.tags for tag in tags)
            ]
        
        # Sort by created_at
        artifacts.sort(key=lambda x: x.created_at, reverse=True)
        return artifacts
    
    def extract_artifact(self, artifact: ModelArtifact, 
                        extract_path: str) -> Dict[str, str]:
        logger.info(f"Extracting artifact {artifact.name}:{artifact.version}")
        
        extract_dir = Path(extract_path)
        extract_dir.mkdir(parents=True, exist_ok=True)
        
        extracted_files = {}
        
        with tarfile.open(artifact.model_path, "r:gz") as tar:
            for member in tar.getmembers():
                tar.extract(member, path=extract_dir)
                extracted_files[member.name] = str(extract_dir / member.name)
        
        logger.info(f"Extracted {len(extracted_files)} files to {extract_dir}")
        return extracted_files
    
    def tag_artifact(self, model_name: str, version: str, tags: List[str]):
        artifact_key = f"{model_name}:{version}"
        if artifact_key in self.artifacts:
            artifact = self.artifacts[artifact_key]
            if artifact.tags is None:
                artifact.tags = []
            artifact.tags.extend(tags)
            artifact.tags = list(set(artifact.tags))  # Remove duplicates
            self._save_metadata()
            logger.info(f"Tagged {artifact_key} with {tags}")
        else:
            logger.warning(f"Artifact {artifact_key} not found")
    
    def promote_artifact(self, model_name: str, version: str, 
                        stage: str = "production"):
        self.tag_artifact(model_name, version, [f"stage:{stage}"])
        logger.info(f"Promoted {model_name}:{version} to {stage}")
    
    def delete_artifact(self, model_name: str, version: str):
        artifact_key = f"{model_name}:{version}"
        if artifact_key in self.artifacts:
            artifact = self.artifacts[artifact_key]
            
            # Delete physical files
            artifact_path = Path(artifact.model_path)
            if artifact_path.exists():
                shutil.rmtree(artifact_path.parent)
            
            # Remove from metadata
            del self.artifacts[artifact_key]
            self._save_metadata()
            
            logger.info(f"Deleted artifact {artifact_key}")
        else:
            logger.warning(f"Artifact {artifact_key} not found")
    
    def cleanup_old_versions(self, model_name: str, keep_latest: int = 3):
        # Get all versions of the model
        model_artifacts = [
            (key, artifact) for key, artifact in self.artifacts.items()
            if artifact.name == model_name
        ]
        
        # Sort by created_at
        model_artifacts.sort(key=lambda x: x[1].created_at, reverse=True)
        
        # Delete old versions
        for key, artifact in model_artifacts[keep_latest:]:
            logger.info(f"Cleaning up old version: {key}")
            self.delete_artifact(artifact.name, artifact.version)
    
    def verify_artifact(self, artifact: ModelArtifact) -> bool:
        if not Path(artifact.model_path).exists():
            logger.error(f"Artifact file not found: {artifact.model_path}")
            return False
        
        calculated_checksum = self._calculate_checksum(artifact.model_path)
        if calculated_checksum != artifact.checksum:
            logger.error(f"Checksum mismatch for {artifact.name}:{artifact.version}")
            return False
        
        logger.info(f"Artifact {artifact.name}:{artifact.version} verified successfully")
        return True
    
    def get_artifact_info(self, model_name: str, version: str) -> Dict[str, Any]:
        artifact = self.get_artifact(model_name, version)
        if not artifact:
            return None
        
        info = artifact.to_dict()
        info['size_mb'] = artifact.size_bytes / (1024 * 1024)
        info['is_valid'] = self.verify_artifact(artifact)
        
        # Get stage from tags
        if artifact.tags:
            stage_tags = [tag for tag in artifact.tags if tag.startswith('stage:')]
            if stage_tags:
                info['stage'] = stage_tags[0].split(':')[1]
        
        return info
    
    def create_model_lineage(self, model_name: str) -> Dict[str, Any]:
        # Get all versions of the model
        model_artifacts = [
            artifact for artifact in self.artifacts.values()
            if artifact.name == model_name
        ]
        
        # Sort by created_at
        model_artifacts.sort(key=lambda x: x.created_at)
        
        lineage = {
            "model_name": model_name,
            "total_versions": len(model_artifacts),
            "versions": []
        }
        
        for artifact in model_artifacts:
            version_info = {
                "version": artifact.version,
                "created_at": artifact.created_at,
                "size_mb": artifact.size_bytes / (1024 * 1024),
                "tags": artifact.tags or [],
                "metadata": artifact.metadata
            }
            lineage["versions"].append(version_info)
        
        return lineage