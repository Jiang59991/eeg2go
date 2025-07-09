#!/usr/bin/env python3
"""
Example script demonstrating how to run experiments with logging to file
"""

import os
import sys
from datetime import datetime
from experiment_engine import run_experiment
from logging_utils import setup_experiment_logging, log_experiment_start, log_experiment_end
import logging

def main():
    """Run correlation experiment with logging to file"""
    
    # Experiment parameters
    experiment_type = "correlation"
    dataset_id = 1
    feature_set_id = 1
    output_dir = "data/experiments/correlation_with_logging"
    
    # Setup logging
    log_file = setup_experiment_logging(
        experiment_name=f"{experiment_type}_experiment",
        log_dir="logs",
        log_level=logging.INFO,
        include_console=True
    )
    
    # Get logger
    logger = logging.getLogger()
    
    # Extra arguments for the experiment
    extra_args = {
        "target_vars": ["age", "sex"],
        "correlation_method": "pearson",
        "plot_correlations": True
    }
    
    # Log experiment start
    log_experiment_start(
        logger=logger,
        experiment_type=experiment_type,
        dataset_id=dataset_id,
        feature_set_id=feature_set_id,
        output_dir=output_dir,
        log_file=log_file
    )
    
    start_time = datetime.now()
    
    try:
        # Run experiment with logging
        result = run_experiment(
            experiment_type=experiment_type,
            dataset_id=dataset_id,
            feature_set_id=feature_set_id,
            output_dir=output_dir,
            extra_args=extra_args,
            log_file=log_file
        )
        
        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # Log experiment end
        log_experiment_end(
            logger=logger,
            experiment_type=experiment_type,
            duration=duration,
            output_dir=result['output_dir'],
            status=result['status']
        )
        
        print(f"\nExperiment completed successfully!")
        print(f"Results saved to: {result['output_dir']}")
        print(f"Log file: {log_file}")
        print(f"Duration: {result['duration']:.2f} seconds")
        
    except Exception as e:
        # Calculate duration even if failed
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.error(f"Experiment failed: {e}")
        log_experiment_end(
            logger=logger,
            experiment_type=experiment_type,
            duration=duration,
            status="failed",
            error=str(e)
        )
        
        print(f"Experiment failed: {e}")
        print(f"Check log file for details: {log_file}")
        sys.exit(1)

if __name__ == "__main__":
    main() 