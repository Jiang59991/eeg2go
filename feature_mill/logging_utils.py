#!/usr/bin/env python3
"""
Logging utilities for the EEG2Go project
"""

import os
import logging
from datetime import datetime
from typing import Optional


def setup_experiment_logging(
    experiment_name: str,
    log_dir: str = "logs",
    log_level: int = logging.INFO,
    include_console: bool = True
) -> str:
    """
    Setup logging for an experiment with both file and console output
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to store log files
        log_level: Logging level
        include_console: Whether to include console output
    
    Returns:
        str: Path to the log file
    """
    # Create log directory if it doesn't exist
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    # Create log file name with timestamp
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")
    
    # Get the root logger
    logger = logging.getLogger()
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # File handler
    file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
    file_handler.setLevel(log_level)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    
    # Console handler (if requested)
    if include_console:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(log_level)
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
    
    # Set root logger level
    logger.setLevel(log_level)
    
    return log_file


def setup_module_logging(
    module_name: str,
    log_file: Optional[str] = None,
    log_level: int = logging.INFO
) -> logging.Logger:
    """
    Setup logging for a specific module
    
    Args:
        module_name: Name of the module
        log_file: Path to log file (optional)
        log_level: Logging level
    
    Returns:
        logging.Logger: Configured logger
    """
    logger = logging.getLogger(module_name)
    
    # Clear existing handlers
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.setLevel(log_level)
    return logger


def get_log_file_path(experiment_name: str, log_dir: str = "logs") -> str:
    """
    Generate a log file path for an experiment
    
    Args:
        experiment_name: Name of the experiment
        log_dir: Directory to store log files
    
    Returns:
        str: Path to the log file
    """
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return os.path.join(log_dir, f"{experiment_name}_{timestamp}.log")


def log_experiment_start(logger: logging.Logger, experiment_type: str, **kwargs):
    """
    Log experiment start information
    
    Args:
        logger: Logger instance
        experiment_type: Type of experiment
        **kwargs: Additional parameters to log
    """
    logger.info("=" * 60)
    logger.info(f"Starting experiment: {experiment_type}")
    logger.info(f"Start time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    
    for key, value in kwargs.items():
        logger.info(f"{key}: {value}")
    
    logger.info("=" * 60)


def log_experiment_end(logger: logging.Logger, experiment_type: str, duration: float, **kwargs):
    """
    Log experiment end information
    
    Args:
        logger: Logger instance
        experiment_type: Type of experiment
        duration: Duration in seconds
        **kwargs: Additional parameters to log
    """
    logger.info("=" * 60)
    logger.info(f"Experiment completed: {experiment_type}")
    logger.info(f"End time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Duration: {duration:.2f} seconds")
    
    for key, value in kwargs.items():
        logger.info(f"{key}: {value}")
    
    logger.info("=" * 60) 