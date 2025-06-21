"""
Logging Module

This module provides centralized logging configuration for the project.
It loads logging settings from a YAML configuration file and provides
easy-to-use functions for getting configured logger instances.

Example:
    >>> from src.utils.logger import setup_logger
    >>> logger = setup_logger('my_module')
    >>> logger.info('Starting data processing')
    >>> logger.error('Failed to load file', exc_info=True)
"""

import logging
import logging.config
import yaml
from pathlib import Path


def setup_logger(name: str = None) -> logging.Logger:
    """
    Set up logger using the configuration file.
    
    This function initializes a logger with settings from the project's
    logging configuration YAML file. If the config file is not found,
    it falls back to a basic configuration.
    
    Args:
        name: Logger name, typically __name__ of the calling module.
              If None, returns the root logger.
        
    Returns:
        logging.Logger: Configured logger instance ready for use.
        
    Example:
        >>> logger = setup_logger(__name__)
        >>> logger.info('Module initialized')
    """
    config_path = Path(__file__).parent.parent.parent / 'configs' / 'logging_config.yaml'
    
    if config_path.exists():
        with open(config_path, 'r') as f:
            config = yaml.safe_load(f.read())
            logging.config.dictConfig(config)
    else:
        # Fallback to basic configuration if config file not found
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
    
    return logging.getLogger(name)


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger instance with the given name.
    
    This is a convenience function that returns an existing logger
    or creates a new one if it doesn't exist. Unlike setup_logger,
    this doesn't reload the configuration.
    
    Args:
        name: Logger name, typically __name__ of the calling module.
        
    Returns:
        logging.Logger: Logger instance with the specified name.
        
    Note:
        Use setup_logger() for initial configuration, and get_logger()
        for subsequent access to the same logger instance.
        
    Example:
        >>> logger = get_logger('data_processor')
        >>> logger.debug('Processing batch %d', batch_num)
    """
    return logging.getLogger(name)