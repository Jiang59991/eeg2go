#!/usr/bin/env python3
"""
Example script demonstrating how to run experiments with logging to file
"""

import os
import sys
from datetime import datetime
from feature_mill.experiment_engine import run_experiment
from logging_config import logger

def main():
    """Run correlation experiment with logging to file"""
    
    # Experiment parameters
    experiment_type = "correlation"
    dataset_id = 1
    feature_set_id = 5
    output_dir = "data/experiments/correlation_with_logging"
    
    logger.info("Experiment started: ...")
    
    # Extra arguments for the experiment
    extra_args = {
        "target_vars": ["age"],
        "method": "pearson",
        "min_corr": 0.05,
        "top_n": 20,
        "plot_corr_matrix": True,
        "plot_scatter": True,
        "save_detailed_results": True
    }
    
    logger.info(f"Experiment started: type={experiment_type}, dataset_id={dataset_id}, feature_set_id={feature_set_id}, output_dir={output_dir}")
    logger.info(f"Parameters: min_corr={extra_args['min_corr']}, method={extra_args['method']}")
    
    start_time = datetime.now()
    
    try:
        # Run experiment with logging
        result = run_experiment(
            experiment_type=experiment_type,
            dataset_id=dataset_id,
            feature_set_id=feature_set_id,
            output_dir=output_dir,
            extra_args=extra_args
        )
        
        # Calculate duration
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.info(f"Experiment completed successfully.")
        logger.info(f"Results saved to: {result.get('output_dir', output_dir)}")
        logger.info(f"Duration: {duration:.2f} seconds")
        if 'experiment_result_id' in result:
            logger.info(f"Experiment result ID: {result['experiment_result_id']}")
        if 'main_output_files' in result:
            logger.info(f"Main output files: {result['main_output_files']}")
        
        logger.info("Experiment completed successfully.")
        
    except Exception as e:
        # Calculate duration even if failed
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        logger.error(f"Experiment failed after {duration:.2f} seconds: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 