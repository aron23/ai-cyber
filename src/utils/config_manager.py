"""
Configuration Manager Module

This module provides a centralized configuration management system for the project.
It handles loading, validation, merging, and environment variable overrides for
YAML configuration files.

Key Features:
    - Automatic loading of all YAML files from config directory
    - Dot notation access to nested configuration values
    - Configuration merging for combining multiple configs
    - Environment variable overrides with PAYLOADBYTE_ prefix
    - Configuration validation and persistence
    - Singleton pattern for consistent access across modules

Example Usage:
    >>> from src.utils.config_manager import get_config_manager
    >>> config = get_config_manager()
    >>> 
    >>> # Get entire configuration
    >>> data_config = config.get('data_config')
    >>> 
    >>> # Get specific nested value
    >>> batch_size = config.get('data_config', 'pipeline.batch_size', default=32)
    >>> 
    >>> # Override from environment
    >>> # export PAYLOADBYTE_DATA_CONFIG_BATCH_SIZE=64
    >>> config.override_from_env()
    >>> 
    >>> # Merge multiple configurations
    >>> merged = config.merge_configs('base_config', 'experiment_config')

Environment Variables:
    Configuration values can be overridden using environment variables with
    the prefix PAYLOADBYTE_. The variable name maps to config paths:
    PAYLOADBYTE_DATA_CONFIG_BATCH_SIZE -> data_config.batch_size
"""

import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union
import os
from copy import deepcopy
from src.utils.logger import setup_logger

logger = setup_logger('config_manager')


class ConfigManager:
    """Manages configuration for the data pipeline."""
    
    def __init__(self, config_dir: Optional[Union[str, Path]] = None):
        """
        Initialize configuration manager.
        
        Args:
            config_dir: Directory containing configuration files
        """
        if config_dir is None:
            # Default to configs directory relative to project root
            self.config_dir = Path(__file__).parent.parent.parent / 'configs'
        else:
            self.config_dir = Path(config_dir)
        
        if not self.config_dir.exists():
            raise ValueError(f"Configuration directory not found: {self.config_dir}")
        
        self.configs = {}
        self._load_all_configs()
    
    def _load_all_configs(self):
        """Load all YAML configuration files from config directory."""
        yaml_files = list(self.config_dir.glob('*.yaml')) + list(self.config_dir.glob('*.yml'))
        
        for config_file in yaml_files:
            config_name = config_file.stem
            try:
                self.configs[config_name] = self._load_yaml_file(config_file)
                logger.info(f"Loaded configuration: {config_name}")
            except Exception as e:
                logger.error(f"Failed to load {config_file}: {e}")
    
    def _load_yaml_file(self, file_path: Path) -> Dict[str, Any]:
        """Load a single YAML file."""
        with open(file_path, 'r') as f:
            return yaml.safe_load(f)
    
    def get(self, config_name: str, key: Optional[str] = None, default: Any = None) -> Any:
        """
        Get configuration value.
        
        Args:
            config_name: Name of the configuration (e.g., 'data_config')
            key: Dot-separated path to nested key (e.g., 'data.batch_size')
            default: Default value if key not found
            
        Returns:
            Configuration value or entire config if key is None
        """
        if config_name not in self.configs:
            logger.warning(f"Configuration '{config_name}' not found")
            return default
        
        config = self.configs[config_name]
        
        if key is None:
            return deepcopy(config)
        
        # Navigate nested dictionary using dot notation
        keys = key.split('.')
        value = config
        
        for k in keys:
            if isinstance(value, dict) and k in value:
                value = value[k]
            else:
                return default
        
        return deepcopy(value)
    
    def set(self, config_name: str, key: str, value: Any):
        """
        Set configuration value.
        
        Args:
            config_name: Name of the configuration
            key: Dot-separated path to nested key
            value: Value to set
        """
        if config_name not in self.configs:
            self.configs[config_name] = {}
        
        # Navigate to the parent of the target key
        keys = key.split('.')
        current = self.configs[config_name]
        
        for k in keys[:-1]:
            if k not in current:
                current[k] = {}
            current = current[k]
        
        # Set the value
        current[keys[-1]] = value
        logger.debug(f"Set {config_name}.{key} = {value}")
    
    def merge_configs(self, *config_names: str) -> Dict[str, Any]:
        """
        Merge multiple configurations.
        
        Args:
            *config_names: Names of configurations to merge
            
        Returns:
            Merged configuration dictionary
        """
        merged = {}
        
        for name in config_names:
            if name in self.configs:
                merged = self._deep_merge(merged, self.configs[name])
            else:
                logger.warning(f"Configuration '{name}' not found for merging")
        
        return merged
    
    def _deep_merge(self, dict1: Dict, dict2: Dict) -> Dict:
        """Deep merge two dictionaries."""
        result = deepcopy(dict1)
        
        for key, value in dict2.items():
            if key in result and isinstance(result[key], dict) and isinstance(value, dict):
                result[key] = self._deep_merge(result[key], value)
            else:
                result[key] = deepcopy(value)
        
        return result
    
    def override_from_env(self):
        """Override configuration values from environment variables."""
        # Environment variables should be prefixed with 'PAYLOADBYTE_'
        prefix = 'PAYLOADBYTE_'
        
        for env_key, env_value in os.environ.items():
            if env_key.startswith(prefix):
                # Convert environment variable to config path
                # PAYLOADBYTE_DATA_CONFIG_BATCH_SIZE -> data_config.batch_size
                path_parts = env_key[len(prefix):].lower().split('_')
                
                if len(path_parts) >= 2:
                    config_name = '_'.join(path_parts[:-1])
                    key = path_parts[-1]
                    
                    # Try to parse value as appropriate type
                    try:
                        if env_value.lower() in ['true', 'false']:
                            value = env_value.lower() == 'true'
                        elif env_value.isdigit():
                            value = int(env_value)
                        elif '.' in env_value and env_value.replace('.', '').isdigit():
                            value = float(env_value)
                        else:
                            value = env_value
                        
                        self.set(config_name, key, value)
                        logger.info(f"Override from env: {config_name}.{key} = {value}")
                    except Exception as e:
                        logger.error(f"Failed to parse env variable {env_key}: {e}")
    
    def save_config(self, config_name: str, file_path: Optional[Union[str, Path]] = None):
        """
        Save configuration to file.
        
        Args:
            config_name: Name of configuration to save
            file_path: Path to save file (defaults to original location)
        """
        if config_name not in self.configs:
            raise ValueError(f"Configuration '{config_name}' not found")
        
        if file_path is None:
            file_path = self.config_dir / f"{config_name}.yaml"
        else:
            file_path = Path(file_path)
        
        with open(file_path, 'w') as f:
            yaml.dump(self.configs[config_name], f, default_flow_style=False, sort_keys=False)
        
        logger.info(f"Saved configuration to {file_path}")
    
    def validate_config(self, config_name: str, schema: Optional[Dict] = None) -> bool:
        """
        Validate configuration against schema.
        
        Args:
            config_name: Name of configuration to validate
            schema: Optional schema dictionary
            
        Returns:
            True if valid, False otherwise
        """
        if config_name not in self.configs:
            logger.error(f"Configuration '{config_name}' not found")
            return False
        
        # Basic validation - check required keys
        if config_name == 'data_config':
            required_keys = ['data', 'preprocessing', 'pipeline']
            config = self.configs[config_name]
            
            for key in required_keys:
                if key not in config:
                    logger.error(f"Missing required key '{key}' in {config_name}")
                    return False
        
        logger.info(f"Configuration '{config_name}' validated successfully")
        return True
    
    def get_data_config(self) -> Dict[str, Any]:
        """Get data pipeline configuration."""
        return self.get('data_config')
    
    def get_preprocessing_config(self) -> Dict[str, Any]:
        """Get preprocessing configuration."""
        return self.get('preprocessing_config')
    
    def get_logging_config(self) -> Dict[str, Any]:
        """Get logging configuration."""
        return self.get('logging_config')
    
    def __repr__(self):
        return f"ConfigManager(configs={list(self.configs.keys())})"


# Singleton instance
_config_manager = None


def get_config_manager(config_dir: Optional[Union[str, Path]] = None) -> ConfigManager:
    """
    Get or create the configuration manager singleton.
    
    Args:
        config_dir: Configuration directory path
        
    Returns:
        ConfigManager instance
    """
    global _config_manager
    
    if _config_manager is None:
        _config_manager = ConfigManager(config_dir)
        _config_manager.override_from_env()
    
    return _config_manager


if __name__ == "__main__":
    # Example usage
    config_mgr = get_config_manager()
    
    # Get entire configuration
    data_config = config_mgr.get_data_config()
    print("Data configuration loaded")
    
    # Get specific value
    batch_size = config_mgr.get('data_config', 'pipeline.batch_size')
    print(f"Batch size: {batch_size}")
    
    # Validate configurations
    for config_name in config_mgr.configs:
        is_valid = config_mgr.validate_config(config_name)
        print(f"{config_name}: {'valid' if is_valid else 'invalid'}")