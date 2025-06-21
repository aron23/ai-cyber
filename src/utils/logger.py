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
    # Get project root directory
    project_root = Path(__file__).parent.parent.parent
    config_path = project_root / 'configs' / 'logging_config.yaml'
    
    try:
        if config_path.exists():
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f.read())
                
                # Resolve log file paths relative to project root
                if 'handlers' in config:
                    for handler_name, handler_config in config['handlers'].items():
                        if 'filename' in handler_config:
                            # Convert relative path to absolute path from project root
                            log_file = project_root / handler_config['filename']
                            # Ensure the log directory exists
                            log_file.parent.mkdir(parents=True, exist_ok=True)
                            # Update the config with absolute path
                            handler_config['filename'] = str(log_file)
                
                logging.config.dictConfig(config)
        else:
            # Fallback to basic configuration if config file not found
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
            )
    except Exception as e:
        # If anything goes wrong with logging config, fall back to basic config
        print(f"Warning: Failed to configure logging from config file: {e}")
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


def reset_logging_config():
    """
    Reset the logging configuration and clear all existing loggers.
    
    This function is useful when you need to reinitialize logging,
    especially in Jupyter notebooks where modules might be cached.
    """
    # Clear all existing handlers
    for handler in logging.root.handlers[:]:
        logging.root.removeHandler(handler)
    
    # Clear existing loggers
    logging.getLogger().handlers.clear()
    
    # Reset logging configuration
    logging.shutdown()
    
    # Reinitialize basic logging
    logging.basicConfig(force=True)


def setup_logger_safe(name: str = None) -> logging.Logger:
    """
    Set up logger with error handling and automatic fallback.
    
    This is a safer version of setup_logger that will always return
    a working logger, even if the configuration fails.
    
    Args:
        name: Logger name, typically __name__ of the calling module.
              If None, returns the root logger.
        
    Returns:
        logging.Logger: Configured logger instance ready for use.
    """
    try:
        return setup_logger(name)
    except Exception as e:
        print(f"Warning: Logger setup failed ({e}), using basic configuration")
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
            force=True
        )
    return logging.getLogger(name)