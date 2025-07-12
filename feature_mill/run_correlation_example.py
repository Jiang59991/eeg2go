#!/usr/bin/env python3
"""
Correlation Analysis Experiment Example Script

This script demonstrates how to use experiment_engine to run a correlation analysis experiment.
It is specifically designed to test age-related EEG features and now supports recording-level statistical analysis.
"""

import os
import logging
import sys
import sqlite3
from logging_config import logger
from feature_mill.experiment_engine import run_experiment, list_experiments, get_experiment_info

def get_age_correlation_featureset_id(db_path='database/eeg2go.db'):
    """Get the feature set ID for 'age_correlation_features'"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # Search for the 'age_correlation_features' feature set
    c.execute("SELECT id, name, description FROM feature_sets WHERE name = 'age_correlation_features'")
    row = c.fetchone()
    
    if row:
        set_id, name, description = row
        # print(f"Found feature set '{name}' (ID: {set_id})")
        # print(f"Description: {description}")
        
        # Get the number of features in the set
        c.execute("SELECT COUNT(*) FROM feature_set_items WHERE feature_set_id = ?", (set_id,))
        feature_count = c.fetchone()[0]
        # print(f"Number of features: {feature_count}")
        
    else:
        # print("✗ 'age_correlation_features' feature set not found")
        # print("Please run the following command to create the feature set:")
        # print("  python setup_age_correlation.py")
        # print("  or")
        # print("  python database/age_correlation_features.py")
        set_id = None
    
    conn.close()
    return set_id

def main():
    """Main function"""
    # log_path = os.path.join('data', 'processed', 'age_correlation_analysis', 'experiment.log')
    # setup_logging(log_file=log_path, log_level=logging.INFO)

    logger.info("=" * 60)
    logger.info("EEG2Go Age Correlation Analysis Experiment Example (Recording-Level)")
    logger.info("=" * 60)

    # Check available experiments
    available_experiments = list_experiments()
    # print(f"Available experiment modules: {available_experiments}")
    
    if 'correlation' not in available_experiments:
        logger.error("Error: 'correlation' experiment module not found")
        return
    
    # Get experiment info
    experiment_info = get_experiment_info('correlation')
    logger.info(f"\nExperiment Info:")
    
    # Check for errors
    if 'error' in experiment_info:
        logger.error(f"  Error: {experiment_info['error']}")
        return
    
    # Safely get info
    logger.info(f"  Name: {experiment_info.get('name', 'Unknown')}")
    logger.info(f"  Module: {experiment_info.get('module', 'Unknown')}")
    logger.info(f"  Has run function: {experiment_info.get('has_run_function', False)}")
    
    # # Dynamically get age correlation feature set ID
    # print(f"\nGetting age correlation feature set...")
    # feature_set_id = get_age_correlation_featureset_id()
    # if not feature_set_id:
    #     print("Error: Unable to get age correlation feature set ID")
    #     return
    
    # Experiment parameters - specifically for age correlation analysis (recording-level)
    experiment_params = {
        'experiment_type': 'correlation',
        'dataset_id': 1,  # Please adjust according to your actual dataset ID
        'feature_set_id': 1,  # Dynamically get feature set ID
        'output_dir': 'data/experiments/age_correlation_analysis_1',
        'extra_args': {
            'target_vars': ['age'],  # Analyze age-related variables
            'method': 'pearson',  # Use Pearson correlation coefficient
            'min_corr': 0.1,  # Lower threshold to show more correlations
            'top_n': 20,  # Show top 20 most correlated features
            'plot_corr_matrix': True,
            'plot_scatter': True,
            'save_detailed_results': True  # Save detailed results
        },
        'db_path': 'database/eeg2go.db'
    }
    
    logger.info(f"\nExperiment Parameters:")
    logger.info(f"  Dataset ID: {experiment_params['dataset_id']}")
    logger.info(f"  Feature Set ID: {experiment_params['feature_set_id']}")
    logger.info(f"  Output Directory: {experiment_params['output_dir']}")
    logger.info(f"  Target Variables: {experiment_params['extra_args']['target_vars']}")
    logger.info(f"  Analysis Method: {experiment_params['extra_args']['method']}")
    logger.info(f"  Minimum Correlation: {experiment_params['extra_args']['min_corr']}")
    logger.info(f"  Top N Features: {experiment_params['extra_args']['top_n']}")
    
    logger.info(f"\nExpected Results (Recording-Level):")
    logger.info(f"  1. Alpha peak frequency_mean - should be negatively correlated with age")
    logger.info(f"  2. Alpha power_mean - should be negatively correlated with age")
    logger.info(f"  3. Theta/Alpha ratio_mean - should be positively correlated with age")
    logger.info(f"  4. Beta power_mean - should be negatively correlated with age")
    logger.info(f"  5. Spectral edge frequency_mean - should be negatively correlated with age")
    logger.info(f"  6. Alpha asymmetry_mean - may be correlated with age")
    logger.info(f"  7. Various statistics (std, min, max, median, count) may also be correlated with age")
    
    logger.info(f"\nRecording-level Statistics Explanation:")
    logger.info(f"  - _mean: Mean value of all epochs")
    logger.info(f"  - _std: Standard deviation of all epochs")
    logger.info(f"  - _min: Minimum value of all epochs")
    logger.info(f"  - _max: Maximum value of all epochs")
    logger.info(f"  - _median: Median value of all epochs")
    logger.info(f"  - _count: Number of valid epochs")
    
    try:
        # Run experiment
        logger.info("\nStarting experiment...")
        result = run_experiment(**experiment_params)
        
        logger.info("\n" + "=" * 60)
        logger.info("Experiment finished!")
        logger.info("=" * 60)
        logger.info(f"Status: {result['status']}")
        logger.info(f"Output Directory: {result['output_dir']}")
        logger.info(f"Duration: {result['duration']:.2f} seconds")
        logger.info(f"\nExperiment Summary:")
        logger.info(result['summary'])
        
        # Show result files
        output_dir = experiment_params['output_dir']
        if os.path.exists(output_dir):
            logger.info(f"\nGenerated result files:")
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    logger.info(f"  {file} ({size} bytes)")
        
    except Exception as e:
        logger.error(f"\nExperiment failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 