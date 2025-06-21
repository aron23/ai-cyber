import pytest
import logging
from pathlib import Path
import sys

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.utils.logger import setup_logger, get_logger


def test_setup_logger():
    """Test logger setup functionality."""
    logger = setup_logger('test_logger')
    assert isinstance(logger, logging.Logger)
    assert logger.name == 'test_logger'


def test_get_logger():
    """Test getting logger instance."""
    logger1 = get_logger('test_logger_1')
    logger2 = get_logger('test_logger_1')
    assert logger1 is logger2  # Should return same instance
    
    logger3 = get_logger('test_logger_2')
    assert logger1 is not logger3  # Different names should return different instances


def test_logger_levels():
    """Test logger can handle different log levels."""
    logger = setup_logger('level_test')
    
    # Test that logger can handle all log levels without error
    logger.debug("Debug message")
    logger.info("Info message")
    logger.warning("Warning message")
    logger.error("Error message")
    logger.critical("Critical message")