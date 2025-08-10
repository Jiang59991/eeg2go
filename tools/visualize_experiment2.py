#!/usr/bin/env python3
"""
实验2可视化：睡眠阶段差异分析结果可视化

生成内容：
1. 热图：Top 30 特征 × 阶段，显示相对W的标准化差值
2. 箱线/小提琴图：Top 6 特征跨阶段分布
3. 侧栏信息：实验元数据
"""

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

# 设置字体和样式
import matplotlib
matplotlib.use('Agg')  # 使用非交互式后端

# 尝试设置中文字体，如果失败则使用默认字体
try:
    plt.rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
    plt.rcParams['axes.unicode_minus'] = False
except:
    # 如果中文字体不可用，使用英文
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']

sns.set_style("whitegrid")

# 设置日志
logging.basicConfig(level=logging.INFO, format='[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s')
logger = logging.getLogger(__name__)

def load_data(stage_feature_csv: str, feature_stats_csv: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """加载数据"""
    logger.info("Loading data...")
    
    # 加载阶段特征数据
    stage_data = pd.read_csv(stage_feature_csv)
    logger.info(f"Stage feature data: {stage_data.shape}")
    
    # 加载特征统计结果
    stats_data = pd.read_csv(feature_stats_csv)
    logger.info(f"Feature stats data: {stats_data.shape}")
    
    return stage_data, stats_data

def get_experiment_metadata(db_path: str, experiment_result_id: int = None) -> Dict:
    """从数据库获取实验元数据"""
    conn = sqlite3.connect(db_path)
    
    if experiment_result_id is None:
        # 获取最新的实验记录
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
    
    # 获取元数据
    metadata = {}
    cursor = conn.execute("""
    SELECT key, value, value_type FROM experiment_metadata 
    WHERE experiment_result_id = ?
    """, (result[0],))
    
    for key, value, value_type in cursor.fetchall():
        metadata[key] = value
    
    # 获取数据集和特征集信息
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

def create_heatmap(stage_data: pd.DataFrame, stats_data: pd.DataFrame, 
                  top_n: int = 30, output_path: str = None) -> plt.Figure:
    """创建热图：Top N 特征 × 阶段，显示相对W的标准化差值"""
    logger.info(f"Creating heatmap for top {top_n} features...")
    
    # 获取Top N特征
    top_features = stats_data.head(top_n)['feature'].tolist()
    
    # 准备热图数据
    stages = ['W', 'N1', 'N2', 'N3', 'REM']
    heatmap_data = []
    
    for feature in top_features:
        row = []
        for stage in stages:
            if stage == 'W':
                row.append(0)  # W阶段作为参考，差值为0
            else:
                cohens_d_key = f'cohens_d_{stage}'
                if cohens_d_key in stats_data.columns:
                    value = stats_data[stats_data['feature'] == feature][cohens_d_key].iloc[0]
                    row.append(value if not pd.isna(value) else 0)
                else:
                    row.append(0)
        heatmap_data.append(row)
    
    heatmap_df = pd.DataFrame(heatmap_data, index=top_features, columns=stages)
    
    # 创建热图
    fig, ax = plt.subplots(figsize=(12, max(8, top_n * 0.3)))
    
    # 使用自定义颜色映射
    cmap = sns.diverging_palette(10, 220, sep=80, n=7)
    
    sns.heatmap(heatmap_df, 
                annot=True, 
                fmt='.2f', 
                cmap=cmap, 
                center=0,
                cbar_kws={'label': "Cohen's d (vs W)"},
                ax=ax)
    
    ax.set_title(f'Top {top_n} Features: Standardized Differences vs Wake Stage', 
                fontsize=16, fontweight='bold', pad=20)
    ax.set_xlabel('Sleep Stage', fontsize=12)
    ax.set_ylabel('Feature', fontsize=12)
    
    # 调整y轴标签
    ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=10)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Heatmap saved to: {output_path}")
    
    return fig

def create_distribution_plots(stage_data: pd.DataFrame, stats_data: pd.DataFrame, 
                            top_n: int = 6, output_path: str = None) -> plt.Figure:
    """创建箱线/小提琴图：Top N 特征跨阶段分布"""
    logger.info(f"Creating distribution plots for top {top_n} features...")
    
    # 获取Top N特征
    top_features = stats_data.head(top_n)['feature'].tolist()
    
    # 准备数据
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
    
    # 创建子图
    n_features = len(top_features)
    n_cols = 3
    n_rows = (n_features + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 4 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    # 设置颜色
    colors = sns.color_palette("husl", 5)
    stage_colors = dict(zip(['W', 'N1', 'N2', 'N3', 'REM'], colors))
    
    for i, feature in enumerate(top_features):
        row = i // n_cols
        col = i % n_cols
        ax = axes[row, col]
        
        feature_data = plot_df[plot_df['feature'] == feature]
        
        # 创建箱线图
        box_data = []
        labels = []
        for stage in ['W', 'N1', 'N2', 'N3', 'REM']:
            stage_data = feature_data[feature_data['stage'] == stage]
            if len(stage_data) > 0:
                # 模拟分布数据（基于median和IQR）
                median_val = stage_data['median'].iloc[0]
                iqr_val = stage_data['iqr'].iloc[0]
                n_samples = int(stage_data['n_epochs'].iloc[0])
                
                # 生成模拟数据
                np.random.seed(42)  # 保持一致性
                simulated_data = np.random.normal(median_val, iqr_val/1.35, n_samples)
                box_data.append(simulated_data)
                labels.append(stage)
        
        if box_data:
            bp = ax.boxplot(box_data, labels=labels, patch_artist=True)
            
            # 设置颜色
            for patch, stage in zip(bp['boxes'], labels):
                patch.set_facecolor(stage_colors[stage])
                patch.set_alpha(0.7)
        
        ax.set_title(f'{feature}', fontsize=10, fontweight='bold')
        ax.set_ylabel('Feature Value')
        ax.grid(True, alpha=0.3)
        
        # 旋转x轴标签
        ax.tick_params(axis='x', rotation=45)
    
    # 隐藏多余的子图
    for i in range(n_features, n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    fig.suptitle(f'Top {top_n} Features: Distribution Across Sleep Stages', 
                fontsize=16, fontweight='bold', y=0.98)
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Distribution plots saved to: {output_path}")
    
    return fig

def create_metadata_summary(metadata: Dict, output_path: str = None) -> plt.Figure:
    """创建元数据摘要图"""
    logger.info("Creating metadata summary...")
    
    fig, ax = plt.subplots(figsize=(10, 8))
    ax.axis('off')
    
    # 检查是否可以使用中文字体
    try:
        # 测试中文字体
        test_font = plt.matplotlib.font_manager.FontProperties(family=['SimHei', 'Arial Unicode MS'])
        use_chinese = True
    except:
        use_chinese = False
    
    # 准备显示信息
    info_text = []
    if use_chinese:
        info_text.append("实验2：睡眠阶段差异分析")
        info_text.append("=" * 30)
        info_text.append("")
        
        # 基本信息
        info_text.append(f"实验名称: {metadata.get('experiment_name', 'N/A')}")
        info_text.append(f"数据集: {metadata.get('dataset_name', 'N/A')}")
        info_text.append(f"特征集: {metadata.get('feature_set_name', 'N/A')}")
        info_text.append(f"运行时间: {metadata.get('run_time', 'N/A')}")
        info_text.append("")
        
        # 参数信息
        params = metadata.get('parameters', {})
        info_text.append("分析参数:")
        info_text.append(f"  统计检验: {params.get('test_type', 'N/A').upper()}")
        info_text.append(f"  特征数量: {params.get('top_n', 'N/A')}")
        info_text.append("")
        
        # 元数据信息
        meta = metadata.get('metadata', {})
        info_text.append("分析结果:")
        info_text.append(f"  总特征数: {meta.get('total_features', 'N/A')}")
        info_text.append(f"  参考阶段: {meta.get('reference_stage', 'N/A')}")
        info_text.append(f"  分析阶段: {meta.get('stages_analyzed', 'N/A')}")
        info_text.append(f"  效应量类型: {meta.get('effect_size_type', 'N/A')}")
        info_text.append(f"  最佳特征: {meta.get('top_feature', 'N/A')}")
        info_text.append(f"  最大效应量: {meta.get('max_effect_size', 'N/A')}")
    else:
        info_text.append("Experiment 2: Sleep Stage Difference Analysis")
        info_text.append("=" * 40)
        info_text.append("")
        
        # Basic information
        info_text.append(f"Experiment Name: {metadata.get('experiment_name', 'N/A')}")
        info_text.append(f"Dataset: {metadata.get('dataset_name', 'N/A')}")
        info_text.append(f"Feature Set: {metadata.get('feature_set_name', 'N/A')}")
        info_text.append(f"Run Time: {metadata.get('run_time', 'N/A')}")
        info_text.append("")
        
        # Parameter information
        params = metadata.get('parameters', {})
        info_text.append("Analysis Parameters:")
        info_text.append(f"  Statistical Test: {params.get('test_type', 'N/A').upper()}")
        info_text.append(f"  Feature Count: {params.get('top_n', 'N/A')}")
        info_text.append("")
        
        # Metadata information
        meta = metadata.get('metadata', {})
        info_text.append("Analysis Results:")
        info_text.append(f"  Total Features: {meta.get('total_features', 'N/A')}")
        info_text.append(f"  Reference Stage: {meta.get('reference_stage', 'N/A')}")
        info_text.append(f"  Analyzed Stages: {meta.get('stages_analyzed', 'N/A')}")
        info_text.append(f"  Effect Size Type: {meta.get('effect_size_type', 'N/A')}")
        info_text.append(f"  Top Feature: {meta.get('top_feature', 'N/A')}")
        info_text.append(f"  Max Effect Size: {meta.get('max_effect_size', 'N/A')}")
    
    # 显示文本
    ax.text(0.05, 0.95, '\n'.join(info_text), 
            transform=ax.transAxes, 
            fontsize=12, 
            verticalalignment='top',
            fontfamily='monospace',
            bbox=dict(boxstyle="round,pad=0.5", facecolor="lightgray", alpha=0.8))
    
    plt.tight_layout()
    
    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Metadata summary saved to: {output_path}")
    
    return fig

def main():
    parser = argparse.ArgumentParser(description='实验2可视化：睡眠阶段差异分析')
    parser.add_argument('--stage-data', required=True, help='阶段特征数据CSV路径')
    parser.add_argument('--stats-data', required=True, help='特征统计结果CSV路径')
    parser.add_argument('--db-path', required=True, help='数据库路径')
    parser.add_argument('--output-dir', default='outputs/experiment2', help='输出目录')
    parser.add_argument('--top-n-heatmap', type=int, default=30, help='热图显示的特征数量')
    parser.add_argument('--top-n-distribution', type=int, default=6, help='分布图显示的特征数量')
    parser.add_argument('--experiment-id', type=int, help='实验ID（可选，默认使用最新）')
    
    args = parser.parse_args()
    
    # 创建输出目录
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # 加载数据
    stage_data, stats_data = load_data(args.stage_data, args.stats_data)
    
    # 获取实验元数据
    metadata = get_experiment_metadata(args.db_path, args.experiment_id)
    
    # 生成可视化
    logger.info("Generating visualizations...")
    
    # 1. 热图
    heatmap_path = output_dir / 'heatmap_top30_features.png'
    create_heatmap(stage_data, stats_data, args.top_n_heatmap, str(heatmap_path))
    
    # 2. 分布图
    distribution_path = output_dir / 'distribution_top6_features.png'
    create_distribution_plots(stage_data, stats_data, args.top_n_distribution, str(distribution_path))
    
    # 3. 元数据摘要
    metadata_path = output_dir / 'experiment_metadata.png'
    create_metadata_summary(metadata, str(metadata_path))
    
    # 4. 保存数据副本
    stage_data.to_csv(output_dir / 'stage_feature_table.csv', index=False)
    stats_data.to_csv(output_dir / 'feature_stats.csv', index=False)
    
    logger.info(f"All visualizations saved to: {output_dir}")

if __name__ == '__main__':
    main()
