#!/usr/bin/env python3
import pandas as pd
import numpy as np
from scipy import stats
from scipy.stats import f_oneway, kruskal
import argparse
import logging
from pathlib import Path
import sqlite3
import json
from typing import Dict, List, Tuple, Optional

# Set up logging
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

def load_stage_data(csv_path: str) -> pd.DataFrame:
    """
    Load stage feature data from a CSV file.

    Args:
        csv_path (str): Path to the input CSV file.

    Returns:
        pd.DataFrame: Loaded DataFrame containing stage feature data.
    """
    logger.info(f"Loading data: {csv_path}")
    df = pd.read_csv(csv_path)
    logger.info(f"Data shape: {df.shape}")
    logger.info(f"Stage distribution: {df['stage'].value_counts()}")
    logger.info(f"Number of features: {df['feature'].nunique()}")
    return df

def calculate_anova_stats(data: pd.DataFrame, feature: str) -> Dict[str, float]:
    """
    Calculate ANOVA statistics for a given feature across sleep stages.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        feature (str): Feature name to analyze.

    Returns:
        Dict[str, float]: Dictionary with keys 'stat', 'p', and 'effect_size' (eta squared).
    """
    stages = ['W', 'N1', 'N2', 'N3', 'REM']
    groups = []
    for stage in stages:
        stage_data = data[(data['feature'] == feature) & (data['stage'] == stage)]['median'].values
        if len(stage_data) > 0:
            groups.append(stage_data)
    if len(groups) < 2:
        return {'stat': np.nan, 'p': np.nan, 'effect_size': np.nan}
    f_stat, p_value = f_oneway(*groups)
    total_n = sum(len(g) for g in groups)
    if total_n > 0:
        grand_mean = np.mean(np.concatenate(groups))
        ss_between = sum(len(g) * (np.mean(g) - grand_mean)**2 for g in groups)
        ss_total = sum((x - grand_mean)**2 for g in groups for x in g)
        eta_squared = ss_between / ss_total if ss_total > 0 else 0
    else:
        eta_squared = np.nan
    return {
        'stat': f_stat,
        'p': p_value,
        'effect_size': eta_squared
    }

def calculate_kruskal_stats(data: pd.DataFrame, feature: str) -> Dict[str, float]:
    """
    Calculate Kruskal-Wallis statistics for a given feature across sleep stages.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        feature (str): Feature name to analyze.

    Returns:
        Dict[str, float]: Dictionary with keys 'stat', 'p', and 'effect_size' (epsilon squared).
    """
    stages = ['W', 'N1', 'N2', 'N3', 'REM']
    groups = []
    for stage in stages:
        stage_data = data[(data['feature'] == feature) & (data['stage'] == stage)]['median'].values
        if len(stage_data) > 0:
            groups.append(stage_data)
    if len(groups) < 2:
        return {'stat': np.nan, 'p': np.nan, 'effect_size': np.nan}
    h_stat, p_value = kruskal(*groups)
    total_n = sum(len(g) for g in groups)
    if total_n > 0:
        epsilon_squared = (h_stat - len(groups) + 1) / (total_n - len(groups))
        epsilon_squared = max(0, epsilon_squared)
    else:
        epsilon_squared = np.nan
    return {
        'stat': h_stat,
        'p': p_value,
        'effect_size': epsilon_squared
    }

def calculate_cohens_d(
    data: pd.DataFrame, feature: str, reference_stage: str = 'W'
) -> Dict[str, float]:
    """
    Calculate Cohen's d for a feature relative to a reference stage.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        feature (str): Feature name to analyze.
        reference_stage (str): Reference stage for Cohen's d calculation.

    Returns:
        Dict[str, float]: Dictionary mapping stage names to Cohen's d values.
    """
    stages = ['W', 'N1', 'N2', 'N3', 'REM']
    cohens_d: Dict[str, float] = {}
    ref_data = data[(data['feature'] == feature) & (data['stage'] == reference_stage)]['median'].values
    if len(ref_data) == 0:
        return {stage: np.nan for stage in stages if stage != reference_stage}
    ref_mean = np.mean(ref_data)
    ref_std = np.std(ref_data, ddof=1)
    for stage in stages:
        if stage == reference_stage:
            continue
        stage_data = data[(data['feature'] == feature) & (data['stage'] == stage)]['median'].values
        if len(stage_data) == 0:
            cohens_d[stage] = np.nan
            continue
        stage_mean = np.mean(stage_data)
        stage_std = np.std(stage_data, ddof=1)
        n1, n2 = len(ref_data), len(stage_data)
        pooled_std = np.sqrt(((n1 - 1) * ref_std ** 2 + (n2 - 1) * stage_std ** 2) / (n1 + n2 - 2))
        if pooled_std > 0:
            cohens_d[stage] = (stage_mean - ref_mean) / pooled_std
        else:
            cohens_d[stage] = np.nan
    return cohens_d

def analyze_feature_differences(
    data: pd.DataFrame, test_type: str = 'anova', top_n: int = 10
) -> pd.DataFrame:
    """
    Analyze feature differences across sleep stages using a specified statistical test.

    Args:
        data (pd.DataFrame): DataFrame containing the data.
        test_type (str): Statistical test type ('anova' or 'kruskal').
        top_n (int): Number of top features to report.

    Returns:
        pd.DataFrame: DataFrame with analysis results for all features.
    """
    logger.info(f"Starting {test_type.upper()} analysis...")
    features = data['feature'].unique()
    results = []
    for i, feature in enumerate(features):
        if i % 100 == 0:
            logger.info(f"Processing feature {i+1}/{len(features)}: {feature}")
        if test_type.lower() == 'anova':
            stats_result = calculate_anova_stats(data, feature)
        elif test_type.lower() == 'kruskal':
            stats_result = calculate_kruskal_stats(data, feature)
        else:
            raise ValueError(f"Unsupported test type: {test_type}")
        cohens_d = calculate_cohens_d(data, feature)
        abs_cohens_d = [abs(d) for d in cohens_d.values() if not np.isnan(d)]
        mean_abs_cohens_d = np.mean(abs_cohens_d) if abs_cohens_d else 0
        result = {
            'feature': feature,
            'stat': stats_result['stat'],
            'p': stats_result['p'],
            'effect_size': stats_result['effect_size'],
            'mean_abs_cohens_d': mean_abs_cohens_d,
            **{f'cohens_d_{stage}': cohens_d.get(stage, np.nan) for stage in ['N1', 'N2', 'N3', 'REM']}
        }
        results.append(result)
    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values('effect_size', ascending=False)
    results_df['rank'] = range(1, len(results_df) + 1)
    top_results = results_df.head(top_n)
    logger.info(f"Analysis complete, Top-{top_n} features:")
    for _, row in top_results.iterrows():
        logger.info(f"  {row['rank']}. {row['feature']} (η²/ε²={row['effect_size']:.4f}, p={row['p']:.2e})")
    return results_df

def save_results(
    results_df: pd.DataFrame,
    output_csv: str,
    output_db: Optional[str] = None,
    dataset_id: int = 1,
    feature_set_id: int = 6,
    test_type: str = 'anova'
) -> None:
    """
    Save analysis results to CSV and optionally to a database.

    Args:
        results_df (pd.DataFrame): DataFrame with analysis results.
        output_csv (str): Path to output CSV file.
        output_db (Optional[str]): Path to output database file.
        dataset_id (int): Dataset ID for database records.
        feature_set_id (int): Feature set ID for database records.
        test_type (str): Statistical test type used.
    """
    logger.info(f"Saving results to: {output_csv}")
    results_df.to_csv(output_csv, index=False)
    if output_db:
        logger.info(f"Saving results to database: {output_db}")
        conn = sqlite3.connect(output_db)
        cursor = conn.cursor()
        # Create task record
        task_parameters = {
            "test_type": test_type,
            "top_n": len(results_df),
            "input_file": str(Path(output_csv).name),
            "output_file": output_csv
        }
        cursor.execute("""
        INSERT INTO tasks 
        (task_type, status, parameters, dataset_id, feature_set_id, experiment_type, 
         progress, processed_count, total_count, output_dir, duration_seconds, notes)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            'experiment',
            'completed',
            json.dumps(task_parameters),
            dataset_id,
            feature_set_id,
            'sleep_stage_statistics',
            100.0,
            1,
            1,
            str(Path(output_csv).parent),
            0,
            f"Sleep stage difference analysis using {test_type.upper()} test"
        ))
        task_id = cursor.lastrowid
        logger.info(f"Created task record with ID: {task_id}")
        # Create unique experiment definition
        experiment_def_name = f"sleep_stage_differences_{test_type}"
        experiment_def_description = f"Sleep stage difference analysis using {test_type.upper()} test"
        default_params = {
            "test_type": test_type,
            "stages": ["W", "N1", "N2", "N3", "REM"],
            "reference_stage": "W",
            "top_n": 10,
            "effect_size_type": "eta_squared" if test_type == "anova" else "epsilon_squared"
        }
        cursor.execute("SELECT id FROM experiment_definitions WHERE name = ?", (experiment_def_name,))
        existing_def = cursor.fetchone()
        if existing_def:
            experiment_def_id = existing_def[0]
            logger.info(f"Using existing experiment definition: {experiment_def_name} (ID: {experiment_def_id})")
        else:
            cursor.execute("""
            INSERT INTO experiment_definitions 
            (name, type, description, default_parameters)
            VALUES (?, ?, ?, ?)
            """, (
                experiment_def_name,
                'sleep_stage_statistics',
                experiment_def_description,
                str(default_params).replace("'", '"')
            ))
            experiment_def_id = cursor.lastrowid
            logger.info(f"Created new experiment definition: {experiment_def_name} (ID: {experiment_def_id})")
        # Create experiment result record
        experiment_name = f"stage_differences_{test_type}"
        experiment_summary = f"Cross-stage difference analysis ({test_type.upper()}), Top-{len(results_df)} features"
        output_dir = str(Path(output_csv).parent)
        cursor.execute("""
        INSERT INTO experiment_results 
        (experiment_type, experiment_name, dataset_id, feature_set_id, parameters, summary, status, output_dir, task_id)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            'sleep_stage_statistics',
            experiment_name,
            dataset_id,
            feature_set_id,
            f'{{"test_type": "{test_type}", "top_n": {len(results_df)}, "input_file": "{Path(output_csv).name}", "experiment_def_id": {experiment_def_id}}}',
            experiment_summary,
            'completed',
            output_dir,
            task_id
        ))
        experiment_result_id = cursor.lastrowid
        # Add experiment metadata
        metadata_items = [
            ("total_features", str(len(results_df)), "number"),
            ("test_type", test_type, "string"),
            ("reference_stage", "W", "string"),
            ("stages_analyzed", "W,N1,N2,N3,REM", "string"),
            ("effect_size_type", "eta_squared" if test_type == "anova" else "epsilon_squared", "string"),
            ("input_data_shape", f"{len(results_df)} features", "string"),
            ("analysis_timestamp", str(pd.Timestamp.now()), "string"),
            ("top_feature", results_df.iloc[0]['feature'] if len(results_df) > 0 else "N/A", "string"),
            ("max_effect_size", f"{results_df.iloc[0]['effect_size']:.4f}" if len(results_df) > 0 else "N/A", "string")
        ]
        for key, value, value_type in metadata_items:
            cursor.execute("""
            INSERT INTO experiment_metadata 
            (experiment_result_id, key, value, value_type)
            VALUES (?, ?, ?, ?)
            """, (experiment_result_id, key, value, value_type))
        logger.info(f"Added {len(metadata_items)} metadata items")
        # Save statistics for each feature to experiment_feature_results
        for _, row in results_df.iterrows():
            cursor.execute("""
            INSERT INTO experiment_feature_results 
            (experiment_result_id, feature_name, target_variable, result_type, metric_name, 
             metric_value, metric_unit, significance_level, rank_position, additional_data)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_result_id,
                row['feature'],
                'sleep_stage',
                'statistic',
                f'{test_type}_statistic',
                row['stat'],
                'statistic',
                f"p={row['p']:.2e}" if row['p'] < 0.001 else f"p={row['p']:.3f}",
                row['rank'],
                f'{{"effect_size": {row["effect_size"]}, "mean_abs_cohens_d": {row["mean_abs_cohens_d"]}}}'
            ))
            cursor.execute("""
            INSERT INTO experiment_feature_results 
            (experiment_result_id, feature_name, target_variable, result_type, metric_name, 
             metric_value, metric_unit, significance_level, rank_position)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_result_id,
                row['feature'],
                'sleep_stage',
                'statistic',
                'p_value',
                row['p'],
                'probability',
                f"p={row['p']:.2e}" if row['p'] < 0.001 else f"p={row['p']:.3f}",
                row['rank']
            ))
            cursor.execute("""
            INSERT INTO experiment_feature_results 
            (experiment_result_id, feature_name, target_variable, result_type, metric_name, 
             metric_value, metric_unit, significance_level, rank_position)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_result_id,
                row['feature'],
                'sleep_stage',
                'statistic',
                'effect_size',
                row['effect_size'],
                'eta_squared' if test_type == 'anova' else 'epsilon_squared',
                f"η²={row['effect_size']:.4f}" if test_type == 'anova' else f"ε²={row['effect_size']:.4f}",
                row['rank']
            ))
            for stage in ['N1', 'N2', 'N3', 'REM']:
                cohens_d_key = f'cohens_d_{stage}'
                if cohens_d_key in row and not pd.isna(row[cohens_d_key]):
                    cursor.execute("""
                    INSERT INTO experiment_feature_results 
                    (experiment_result_id, feature_name, target_variable, result_type, metric_name, 
                     metric_value, metric_unit, significance_level, rank_position)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        experiment_result_id,
                        row['feature'],
                        f'sleep_stage_vs_{stage}',
                        'effect_size',
                        'cohens_d',
                        row[cohens_d_key],
                        'standardized_difference',
                        f"d={row[cohens_d_key]:.3f}",
                        row['rank']
                    ))
        conn.commit()
        conn.close()
        logger.info(f"Results saved to database, experiment_result_id: {experiment_result_id}")

def main() -> None:
    """
    Main function to parse arguments, run analysis, and save results.
    """
    parser = argparse.ArgumentParser(description='Cross-stage feature difference statistical analysis')
    parser.add_argument('--input', required=True, help='Input CSV file path')
    parser.add_argument('--output-csv', required=True, help='Output CSV file path')
    parser.add_argument('--output-db', help='Output database path (optional)')
    parser.add_argument('--test-type', choices=['anova', 'kruskal'], default='anova', 
                       help='Statistical test type (default: anova)')
    parser.add_argument('--top-n', type=int, default=10, help='Top-N features (default: 10)')
    parser.add_argument('--dataset-id', type=int, default=1, help='Dataset ID (default: 1)')
    parser.add_argument('--feature-set-id', type=int, default=6, help='Feature set ID (default: 6)')
    args = parser.parse_args()
    data = load_stage_data(args.input)
    results = analyze_feature_differences(data, args.test_type, args.top_n)
    save_results(results, args.output_csv, args.output_db, 
                args.dataset_id, args.feature_set_id, args.test_type)
    logger.info("Statistical analysis complete!")

if __name__ == '__main__':
    main()
