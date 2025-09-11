#!/usr/bin/env python3
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import sqlite3
import json
from pathlib import Path
import argparse
import logging
from typing import Dict, List, Tuple

import matplotlib
matplotlib.use('Agg')

plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
sns.set_style("whitegrid")

logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

def load_data(stage_feature_csv: str, feature_stats_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Load stage feature data and feature statistics data from CSV files.

    Args:
        stage_feature_csv (str): Path to the stage feature data CSV file.
        feature_stats_csv (str): Path to the feature statistics results CSV file.

    Returns:
        Tuple[pd.DataFrame, pd.DataFrame]: (stage_data, stats_data)
    """
    logger.info("Loading data...")
    stage_data = pd.read_csv(stage_feature_csv)
    logger.info(f"Stage feature data: {stage_data.shape}")
    stats_data = pd.read_csv(feature_stats_csv)
    logger.info(f"Feature stats data: {stats_data.shape}")
    return stage_data, stats_data

def get_experiment_metadata(db_path: str, experiment_result_id: int = None) -> Dict:
    """
    Retrieve experiment metadata from the database.

    Args:
        db_path (str): Path to the SQLite database.
        experiment_result_id (int, optional): Experiment result ID. If None, use the latest.

    Returns:
        Dict: Metadata dictionary containing experiment information.
    """
    conn = sqlite3.connect(db_path)
    if experiment_result_id is None:
        cursor = conn.execute("""
        SELECT id, experiment_type, experiment_name, dataset_id, feature_set_id, 
               parameters, summary, run_time, output_dir
        FROM experiment_results 
        WHERE experiment_type = 'sleep_stage_statistics'
        ORDER BY run_time DESC LIMIT 1
        """)
    else:
        cursor = conn.execute("""
        SELECT id, experiment_type, experiment_name, dataset_id, feature_set_id, 
               parameters, summary, run_time, output_dir
        FROM experiment_results 
        WHERE id = ?
        """, (experiment_result_id,))
    result = cursor.fetchone()
    if not result:
        logger.warning("No experiment results found")
        return {}
    metadata = {}
    cursor = conn.execute("""
    SELECT key, value, value_type FROM experiment_metadata 
    WHERE experiment_result_id = ?
    """, (result[0],))
    for key, value, value_type in cursor.fetchall():
        metadata[key] = value
    cursor = conn.execute("SELECT name FROM datasets WHERE id = ?", (result[3],))
    dataset_result = cursor.fetchone()
    dataset_name = dataset_result[0] if dataset_result else "Unknown"
    cursor = conn.execute("SELECT name FROM feature_sets WHERE id = ?", (result[4],))
    feature_set_result = cursor.fetchone()
    feature_set_name = feature_set_result[0] if feature_set_result else "Unknown"
    conn.close()
    return {
        'experiment_id': result[0],
        'experiment_name': result[2],
        'dataset_name': dataset_name,
        'feature_set_name': feature_set_name,
        'run_time': result[7],
        'parameters': json.loads(result[5]) if result[5] else {},
        'metadata': metadata
    }

def create_heatmap(
    stage_data: pd.DataFrame, 
    stats_data: pd.DataFrame, 
    top_n: int = 30, 
    output_path: str = None
) -> plt.Figure:
    """
    Create a heatmap for the top N features across sleep stages, showing standardized differences relative to W.

    Args:
        stage_data (pd.DataFrame): Stage feature data.
        stats_data (pd.DataFrame): Feature statistics data.
        top_n (int, optional): Number of top features to display.
        output_path (str, optional): Path to save the heatmap image.

    Returns:
        plt.Figure: The matplotlib Figure object of the heatmap.
    """
    logger.info(f"Creating heatmap for top {top_n} features...")
    top_features = stats_data.head(top_n)['feature'].tolist()
    stages = ['W', 'N1', 'N2', 'N3', 'REM']
    heatmap_data = []
    for feature in top_features:
        row = []
        for stage in stages:
            if stage == 'W':
                row.append(0)
            else:
                cohens_d_key = f'cohens_d_{stage}'
                if cohens_d_key in stats_data.columns:
                    value = stats_data[stats_data['feature'] == feature][cohens_d_key].iloc[0]
                    row.append(value if not pd.isna(value) else 0)
                else:
                    row.append(0)
        heatmap_data.append(row)
    heatmap_df = pd.DataFrame(heatmap_data, index=top_features, columns=stages)
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.3)))
    cmap = sns.diverging_palette(10, 220, sep=80, n=7)
    sns.heatmap(
        heatmap_df, 
        annot=True, 
        fmt='.2f', 
        cmap=cmap, 
        center=0,
        cbar_kws={'label': "Cohen's d (vs W)"},
        ax=ax
    )
    ax.set_title(f'Top {top_n} Features: Standardized Differences vs Wake Stage', fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Sleep Stage', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Heatmap saved to: {output_path}")
    return fig

def create_distribution_plots(
    stage_data: pd.DataFrame, 
    stats_data: pd.DataFrame, 
    top_n: int = 6, 
    output_path: str = None
) -> plt.Figure:
    """
    Create box plots for the top N features, showing their distribution across sleep stages.

    Args:
        stage_data (pd.DataFrame): Stage feature data.
        stats_data (pd.DataFrame): Feature statistics data.
        top_n (int, optional): Number of top features to display.
        output_path (str, optional): Path to save the distribution plots image.

    Returns:
        plt.Figure: The matplotlib Figure object of the distribution plots.
    """
    logger.info(f"Creating distribution plots for top {top_n} features...")
    top_features = stats_data.head(top_n)['feature'].tolist()
    plot_data = []
    for feature in top_features:
        feature_data = stage_data[stage_data['feature'] == feature]
        for _, row in feature_data.iterrows():
            plot_data.append({
                'feature': feature,
                'stage': row['stage'],
                'median': row['median'],
                'iqr': row['iqr'],
                'n_epochs': row['n_epochs']
            })
    plot_df = pd.DataFrame(plot_data)
    n_features = len(top_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    colors = sns.color_palette("husl", 5)
    stage_colors = dict(zip(['W', 'N1', 'N2', 'N3', 'REM'], colors))
    for i, feature in enumerate(top_features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        feature_data = plot_df[plot_df['feature'] == feature]
        box_data = []
        labels = []
        for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
            stage_data = feature_data[feature_data['stage'] == stage]
            if len(stage_data) > 0:
                median_val = stage_data['median'].iloc[0]
                iqr_val = stage_data['iqr'].iloc[0]
                n_samples = int(stage_data['n_epochs'].iloc[0])
                np.random.seed(42)
                simulated_data = np.random.normal(median_val, iqr_val/1.35, n_samples)
                box_data.append(simulated_data)
                labels.append(stage)
        if box_data:
            bp = ax.boxplot(box_data, tick_labels=labels, patch_artist=True)
            for patch, stage in zip(bp['boxes'], labels):
                patch.set_facecolor(stage_colors[stage])
                patch.set_alpha(0.7)
        ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Feature Value')
        ax.grid(True, alpha=0.3)
        ax.tick_params(axis='x', rotation=45)
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    fig.suptitle(f'Top {top_n} Features: Distribution Across Sleep Stages', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Distribution plots saved to: {output_path}")
    return fig

def create_metadata_summary(
    metadata: Dict, 
    output_path: str = None
) -> plt.Figure:
    """
    Create a summary figure displaying experiment metadata and analysis results.

    Args:
        metadata (Dict): Metadata dictionary containing experiment information.
        output_path (str, optional): Path to save the metadata summary image.

    Returns:
        plt.Figure: The matplotlib Figure object of the metadata summary.
    """
    logger.info("Creating metadata summary...")
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    info_text = []
    info_text.append("Experiment 2: Sleep Stage Difference Analysis")
    info_text.append("=" * 40)
    info_text.append("")
    info_text.append(f"Experiment Name: {metadata.get('experiment_name', 'N/A')}")
    info_text.append(f"Dataset: {metadata.get('dataset_name', 'N/A')}")
    info_text.append(f"Feature Set: {metadata.get('feature_set_name', 'N/A')}")
    info_text.append(f"Run Time: {metadata.get('run_time', 'N/A')}")
    info_text.append("")
    params = metadata.get('parameters', {})
    info_text.append("Analysis Parameters:")
    info_text.append(f"  Statistical Test: {params.get('test_type', 'N/A').upper()}")
    info_text.append(f"  Feature Count: {params.get('top_n', 'N/A')}")
    info_text.append("")
    meta = metadata.get('metadata', {})
    info_text.append("Analysis Results:")
    info_text.append(f"  Total Features: {meta.get('total_features', 'N/A')}")
    info_text.append(f"  Reference Stage: {meta.get('reference_stage', 'N/A')}")
    info_text.append(f"  Analyzed Stages: {meta.get('stages_analyzed', 'N/A')}")
    info_text.append(f"  Effect Size Type: {meta.get('effect_size_type', 'N/A')}")
    info_text.append(f"  Top Feature: {meta.get('top_feature', 'N/A')}")
    info_text.append(f"  Max Effect Size: {meta.get('max_effect_size', 'N/A')}")
    ax.text(
        0.05, 0.95, '\n'.join(info_text), 
        transform=ax.transAxes, 
        fontsize=12, 
        verticalalignment='top',
        fontfamily='monospace',
        bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8)
    )
    plt.tight_layout()
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metadata summary saved to: {output_path}")
    return fig

def main() -> None:
    """
    Main function to parse arguments, load data, generate visualizations, and save outputs.
    """
    parser = argparse.ArgumentParser(description='Experiment 2 Visualization: Sleep Stage Difference Analysis')
    parser.add_argument('--stage-data', required=True, help='Stage feature data CSV path')
    parser.add_argument('--stats-data', required=True, help='Feature statistics results CSV path')
    parser.add_argument('--db-path', required=True, help='Database path')
    parser.add_argument('--output-dir', default='outputs/experiment2', help='Output directory')
    parser.add_argument('--top-n-heatmap', type=int, default=30, help='Number of features to display in heatmap')
    parser.add_argument('--top-n-distribution', type=int, default=6, help='Number of features to display in distribution plots')
    parser.add_argument('--experiment-id', type=int, help='Experiment ID (optional, default use latest)')
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    stage_data, stats_data = load_data(args.stage_data, args.stats_data)
    metadata = get_experiment_metadata(args.db_path, args.experiment_id)
    logger.info("Generating visualizations...")
    heatmap_path = output_dir / 'heatmap_top30_features.png'
    create_heatmap(stage_data, stats_data, args.top_n_heatmap, str(heatmap_path))
    distribution_path = output_dir / 'distribution_top6_features.png'
    create_distribution_plots(stage_data, stats_data, args.top_n_distribution, str(distribution_path))
    metadata_path = output_dir / 'experiment_metadata.png'
    create_metadata_summary(metadata, str(metadata_path))
    stage_data.to_csv(output_dir / 'stage_feature_table.csv', index=False)
    stats_data.to_csv(output_dir / 'feature_stats.csv', index=False)
    logger.info(f"All visualizations saved to: {output_dir}")

if __name__ == '__main__':
    main()
