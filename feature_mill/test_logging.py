#!/usr/bin/env python3
"""
Test script for logging functionality
"""

import os
import logging
from datetime import datetime
from logging_utils import setup_experiment_logging, log_experiment_start, log_experiment_end

def test_logging():
    """Test the logging functionality"""
    
    print("Testing logging functionality...")
    
    # Setup logging
    log_file = setup_experiment_logging(
        experiment_name="test_logging",
        log_dir="logs",
        log_level=logging.INFO,
        include_console=True
    )
    
    logger = logging.getLogger()
    
    # Log start
    log_experiment_start(
        logger=logger,
        experiment_type="test",
        test_param="test_value"
    )
    
    # Test different log levels
    logger.debug("This is a debug message")
    logger.info("This is an info message")
    logger.warning("This is a warning message")
    logger.error("This is an error message")
    
    # Test experiment end
    log_experiment_end(
        logger=logger,
        experiment_type="test",
        duration=1.5,
        status="success"
    )
    
    print(f"Log file created: {log_file}")
    
    # Check if log file exists and has content
    if os.path.exists(log_file):
        with open(log_file, 'r', encoding='utf-8') as f:
            content = f.read()
            print(f"Log file size: {len(content)} characters")
            print("Log file content:")
            print("-" * 40)
            print(content)
            print("-" * 40)
    else:
        print("ERROR: Log file was not created!")
    
    print("Logging test completed!")

if __name__ == "__main__":
    test_logging() 