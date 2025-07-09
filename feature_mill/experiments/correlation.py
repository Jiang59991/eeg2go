#!/usr/bin/env python3
"""
相关性分析实验模块

该模块实现EEG特征与目标变量（如年龄、性别等）的相关性分析。
支持多种相关性分析方法，包括皮尔逊相关系数、斯皮尔曼相关系数等。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
from sklearn.preprocessing import LabelEncoder
from datetime import datetime
from logging_config import logger  # 使用全局logger
warnings.filterwarnings('ignore')

# Set English font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run(df_feat: pd.DataFrame, df_meta: pd.DataFrame, output_dir: str, **kwargs) -> str:
    """
    运行相关性分析实验
    
    Args:
        df_feat: 特征矩阵DataFrame
        df_meta: 元数据DataFrame
        output_dir: 输出目录
        **kwargs: 额外参数
            - target_vars: 目标变量列表，默认['age', 'sex']
            - method: 相关性分析方法，默认'pearson'
            - min_corr: 最小相关系数阈值，默认0.3
            - top_n: 显示前N个最相关特征，默认20
            - plot_corr_matrix: 是否绘制相关性矩阵，默认True
            - plot_scatter: 是否绘制散点图，默认True
            - save_detailed_results: 是否保存详细结果，默认True
    
    Returns:
        str: 实验摘要
    """
    # 获取参数
    target_vars = kwargs.get('target_vars', ['age', 'sex'])
    method = kwargs.get('method', 'pearson')
    min_corr = kwargs.get('min_corr', 0.3)
    top_n = kwargs.get('top_n', 20)
    plot_corr_matrix = kwargs.get('plot_corr_matrix', True)
    plot_scatter = kwargs.get('plot_scatter', True)
    save_detailed_results = kwargs.get('save_detailed_results', True)
    
    logger.info(f"Start correlation analysis experiment")
    logger.info(f"Feature matrix shape: {df_feat.shape}")
    logger.info(f"Metadata shape: {df_meta.shape}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 合并特征矩阵和元数据
    df_combined = pd.merge(df_feat, df_meta, on='recording_id', how='inner')
    
    logger.info(f"Correlation analysis completed, results saved to: {output_dir}")
    
    # 数据预处理
    df_processed = df_combined.copy()
    
    logger.info(f"Merged data shape: {df_combined.shape}")
    logger.info(f"Number of recordings: {len(df_combined)}")
    
    # 处理缺失值
    missing_ratio = df_processed.isnull().sum() / len(df_processed)
    columns_to_drop_missing = missing_ratio[missing_ratio > 0.9].index.tolist()
    df_processed = df_processed.drop(columns=columns_to_drop_missing)
    
    logger.info(f"Initial data shape: {df_processed.shape}")
    
    # 移除缺失值过多的列
    missing_ratio = df_processed.isnull().sum() / len(df_processed)
    logger.info(f"Missing value statistics:")
    logger.info(f"- Columns with >50% missing: {len(missing_ratio[missing_ratio > 0.5])}")
    logger.info(f"- Columns with >80% missing: {len(missing_ratio[missing_ratio > 0.8])}")
    logger.info(f"- Columns with >90% missing: {len(missing_ratio[missing_ratio > 0.9])}")
    
    # 移除缺失值过多的列
    columns_to_drop_missing = missing_ratio[missing_ratio > 0.9].index.tolist()
    df_processed = df_processed.drop(columns=columns_to_drop_missing)
    logger.info(f"Columns with >90% missing values: {len(columns_to_drop_missing)}")
    logger.info(f"Shape after removing high-missing columns: {df_processed.shape}")
    
    # 只保留数值型列
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns.tolist()
    non_numeric_columns = df_processed.select_dtypes(exclude=[np.number]).columns.tolist()
    
    logger.info(f"Numeric columns: {len(numeric_columns)}")
    logger.info(f"Non-numeric columns: {len(non_numeric_columns)}")
    
    # 移除非数值型列
    columns_to_remove = [col for col in non_numeric_columns if col != 'recording_id']
    df_processed = df_processed.drop(columns=columns_to_remove)
    logger.info(f"Non-numeric columns removed: {len(columns_to_remove)}")
    logger.info(f"Shape after removing non-numeric columns: {df_processed.shape}")
    
    # 移除recording_id列，只保留特征和目标变量
    if 'recording_id' in df_processed.columns:
        df_processed = df_processed.drop(columns=['recording_id'])
    
    # 分析每个目标变量
    results = {}
    
    for target_var in target_vars:
        df = df_processed.copy()
        
        logger.info(f"Target variables: {target_vars}")
        
        # 获取特征列（排除目标变量）
        feature_cols = [col for col in df.columns if col not in target_vars]
        
        logger.info(f"Feature columns: {len(feature_cols)}")
        
        if len(feature_cols) == 0:
            logger.warning("WARNING: No feature columns available for analysis!")
            logger.warning("This means all columns are either recording_id or target variables")
            continue
        
        # 检查目标变量是否存在
        if target_var not in df.columns:
            logger.warning(f"Warning: Target variable {target_var} not in data")
            # 为不存在的目标变量创建空结果
            results[target_var] = {
                'all_results': pd.DataFrame(),
                'top_results': pd.DataFrame(),
                'total_features': 0,
                'significant_count': 0
            }
            continue
        
        logger.info(f"Analyzing target variable: {target_var}")
        logger.info(f"Number of features to analyze: {len(feature_cols)}")
        
        # 显示一些特征数据样本
        # logger.info(f"Sample feature data:")
        # for feature in feature_cols[:3]:  # 只显示前3个特征
        #     logger.info(f"  {feature}: mean={df[feature].mean():.4f}, std={df[feature].std():.4f}, range=[{df[feature].min():.4f}, {df[feature].max():.4f}]")
        
        # 处理目标变量
        if target_var in ['sex', 'race', 'ethnicity']:
            # 分类变量，使用标签编码
            le = LabelEncoder()
            df[target_var] = le.fit_transform(df[target_var].astype(str))
            logger.info(f"Converted categorical variable {target_var} to numeric using LabelEncoder")
        
        # 计算相关性
        correlations = []
        for feature in feature_cols:
            # 移除缺失值
            valid_data = df[[feature, target_var]].dropna()
            
            if len(valid_data) < 10:  # 至少需要10个有效数据点
                continue
            
            try:
                if target_var in ['sex', 'race', 'ethnicity']:
                    # 对于分类变量，使用点双列相关系数（point-biserial correlation）
                    # 这相当于皮尔逊相关系数，但更适合二分类变量
                    corr, p_value = stats.pearsonr(valid_data[feature], valid_data[target_var])
                    correlation_type = 'point-biserial'
                else:
                    # 对于连续变量，使用指定的相关系数方法
                    if method == 'pearson':
                        corr, p_value = stats.pearsonr(valid_data[feature], valid_data[target_var])
                        correlation_type = 'pearson'
                    elif method == 'spearman':
                        corr, p_value = stats.spearmanr(valid_data[feature], valid_data[target_var])
                        correlation_type = 'spearman'
                    else:
                        logger.warning(f"Unknown correlation method: {method}, using pearson")
                        corr, p_value = stats.pearsonr(valid_data[feature], valid_data[target_var])
                        correlation_type = 'pearson'
                
                # 检查p-value是否有效
                if np.isnan(p_value) or p_value < 0 or p_value > 1:
                    logger.warning(f"Invalid p-value for feature {feature}: {p_value}")
                    continue
                
                # 判断显著性
                if p_value < 0.001:
                    significance = '***'
                elif p_value < 0.01:
                    significance = '**'
                elif p_value < 0.05:
                    significance = '*'
                else:
                    significance = 'ns'
                
                correlations.append({
                    'feature': feature,
                    'correlation': corr,
                    'p_value': p_value,
                    'significance': significance,
                    'n_samples': len(valid_data),
                    'correlation_type': correlation_type
                })
            except Exception as e:
                logger.warning(f"Failed to calculate correlation for feature {feature}: {e}")
                continue
        
        # 转换为DataFrame并排序
        result_df = pd.DataFrame(correlations)
        if len(result_df) > 0:
            result_df = result_df.sort_values('correlation', key=abs, ascending=False)
            result_df['rank'] = range(1, len(result_df) + 1)
            
            # 筛选显著相关的特征
            significant_features = result_df[abs(result_df['correlation']) >= min_corr]
            
            results[target_var] = {
                'all_results': result_df,  # 保存所有特征的结果
                'top_results': significant_features.head(top_n),  # 只保存满足阈值的特征
                'total_features': len(result_df),
                'significant_count': len(significant_features)
            }
            
            logger.info(f"Found {len(result_df)} total features analyzed")
            logger.info(f"Found {len(significant_features)} features with correlation >= {min_corr}")
            logger.info(f"Significant features: {results[target_var]['significant_count']}")
            
            # 显示前几个最相关的特征
            logger.info(f"Top correlations:")
            for _, row in result_df.head(5).iterrows():
                logger.info(f"  {row['feature']}: corr={row['correlation']:.4f}, p={row['p_value']:.4f}")
        else:
            results[target_var] = {
                'all_results': pd.DataFrame(),
                'top_results': pd.DataFrame(),
                'total_features': 0,
                'significant_count': 0
            }
    
    # 保存结果
    for target_var, result in results.items():
        # 保存所有特征的相关性结果（不管相关性高低）
        if len(result['all_results']) > 0:
            # 保存所有结果到 _all.csv 文件
            all_results_file = os.path.join(output_dir, f'correlation_{target_var}_all.csv')
            result['all_results'].to_csv(all_results_file, index=False)
            
            # 保存前N个最相关的结果到 _top.csv 文件
            if save_detailed_results:
                top_results_file = os.path.join(output_dir, f'correlation_{target_var}_top.csv')
                result['top_results'].to_csv(top_results_file, index=False)
            
            # 保存汇总结果
            summary_file = os.path.join(output_dir, f'correlation_{target_var}_summary.csv')
            summary_data = {
                'target_variable': [target_var],
                'total_features': [result['total_features']],
                'significant_features': [result['significant_count']],
                'correlation_method': [method],
                'min_correlation_threshold': [min_corr]
            }
            pd.DataFrame(summary_data).to_csv(summary_file, index=False)
        else:
            # 即使没有特征数据，也要保存空的结果文件
            all_results_file = os.path.join(output_dir, f'correlation_{target_var}_all.csv')
            pd.DataFrame(columns=['feature', 'correlation', 'p_value', 'significance', 'n_samples', 'correlation_type']).to_csv(all_results_file, index=False)
            
            top_results_file = os.path.join(output_dir, f'correlation_{target_var}_top.csv')
            pd.DataFrame(columns=['feature', 'correlation', 'p_value', 'significance', 'n_samples', 'correlation_type']).to_csv(top_results_file, index=False)
            
            # 保存汇总结果
            summary_file = os.path.join(output_dir, f'correlation_{target_var}_summary.csv')
            summary_data = {
                'target_variable': [target_var],
                'total_features': [result['total_features']],
                'significant_features': [result['significant_count']],
                'correlation_method': [method],
                'min_correlation_threshold': [min_corr]
            }
            pd.DataFrame(summary_data).to_csv(summary_file, index=False)
    
    # 绘制相关性矩阵
    if plot_corr_matrix and len(results) > 0:
        try:
            # 重新准备数据用于绘图
            df_plot = df_combined.copy()
            
            # 处理分类变量
            for target_var in target_vars:
                if target_var in ['sex', 'race', 'ethnicity'] and target_var in df_plot.columns:
                    le = LabelEncoder()
                    df_plot[target_var] = le.fit_transform(df_plot[target_var].astype(str))
            
            # 选择前10个最相关的特征进行可视化
            all_features = []
            for target_var, result in results.items():
                if len(result['all_results']) > 0:
                    top_features = result['all_results'].head(10)['feature'].tolist()
                    all_features.extend(top_features)
            
            if all_features:
                # 去重
                unique_features = list(set(all_features))[:20]  # 最多20个特征
                
                # 确保所有特征都在数据中
                available_features = [f for f in unique_features if f in df_plot.columns]
                available_targets = [t for t in target_vars if t in df_plot.columns]
                
                if available_features and available_targets:
                    # 创建相关性矩阵
                    corr_matrix = df_plot[available_features + available_targets].corr(method=method)
                    
                    plt.figure(figsize=(12, 10))
                    sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, 
                               square=True, fmt='.2f', cbar_kws={'shrink': 0.8})
                    plt.title(f'Feature Correlation Matrix ({method.capitalize()})')
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, 'correlation_matrix.png'), dpi=300, bbox_inches='tight')
                    plt.close()
        except Exception as e:
            logger.warning(f"Failed to create correlation matrix plot: {e}")
    
    # 绘制散点图
    if plot_scatter and len(results) > 0:
        try:
            # 重新准备数据用于绘图
            df_plot = df_combined.copy()
            
            # 处理分类变量
            for target_var in target_vars:
                if target_var in ['sex', 'race', 'ethnicity'] and target_var in df_plot.columns:
                    le = LabelEncoder()
                    df_plot[target_var] = le.fit_transform(df_plot[target_var].astype(str))
            
            for target_var, result in results.items():
                if len(result['all_results']) > 0 and target_var in df_plot.columns:
                    top_features = result['all_results'].head(5)['feature'].tolist()
                    
                    # 确保特征在数据中
                    available_features = [f for f in top_features if f in df_plot.columns]
                    
                    if available_features:
                        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
                        axes = axes.flatten()
                        
                        for i, feature in enumerate(available_features[:6]):
                            if i < len(axes):
                                ax = axes[i]
                                
                                # 确保数据长度匹配
                                valid_data = df_plot[[target_var, feature]].dropna()
                                if len(valid_data) > 0:
                                    ax.scatter(valid_data[target_var], valid_data[feature], alpha=0.6)
                                    ax.set_xlabel(target_var)
                                    ax.set_ylabel(feature)
                                    ax.set_title(f'{feature} vs {target_var}')
                                    
                                    # 添加趋势线
                                    if len(valid_data) > 1:
                                        try:
                                            z = np.polyfit(valid_data[target_var], valid_data[feature], 1)
                                            p = np.poly1d(z)
                                            ax.plot(valid_data[target_var], p(valid_data[target_var]), "r--", alpha=0.8)
                                        except Exception as e:
                                            logger.warning(f"Failed to add trend line for {feature}: {e}")
                                else:
                                    ax.text(0.5, 0.5, 'No valid data', ha='center', va='center', transform=ax.transAxes)
                                    ax.set_title(f'{feature} vs {target_var} (No data)')
                        
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f'scatter_{target_var}.png'), 
                                   dpi=300, bbox_inches='tight')
                        plt.close()
                    else:
                        logger.warning(f"No available features for scatter plot with {target_var}")
        except Exception as e:
            logger.warning(f"Failed to create scatter plots: {e}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
    
    # 新增：绘制相关性条形图
    if plot_corr_matrix and len(results) > 0:
        try:
            for target_var, result in results.items():
                if len(result['all_results']) > 0:
                    # 获取前20个最相关的特征
                    top_features = result['all_results'].head(20)
                    
                    plt.figure(figsize=(12, 8))
                    
                    # 创建条形图
                    bars = plt.barh(range(len(top_features)), top_features['correlation'])
                    
                    # 根据显著性设置颜色
                    colors = []
                    for significance in top_features['significance']:
                        if significance == '***':
                            colors.append('red')
                        elif significance == '**':
                            colors.append('orange')
                        elif significance == '*':
                            colors.append('yellow')
                        else:
                            colors.append('lightgray')
                    
                    for i, (bar, color) in enumerate(zip(bars, colors)):
                        bar.set_color(color)
                    
                    # 设置标签
                    plt.yticks(range(len(top_features)), top_features['feature'])
                    plt.xlabel(f'Correlation Coefficient ({target_var})')
                    plt.title(f'Top 20 Correlated Features with {target_var}')
                    
                    # 添加显著性标记
                    for i, (corr, significance) in enumerate(zip(top_features['correlation'], top_features['significance'])):
                        if significance != 'ns':
                            plt.text(corr + (0.01 if corr > 0 else -0.01), i, significance, 
                                   va='center', ha='left' if corr > 0 else 'right', fontweight='bold')
                    
                    plt.grid(axis='x', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'correlation_bars_{target_var}.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
        except Exception as e:
            logger.warning(f"Failed to create correlation bar plots: {e}")
    
    # 新增：绘制相关性分布图
    if plot_corr_matrix and len(results) > 0:
        try:
            for target_var, result in results.items():
                if len(result['all_results']) > 0:
                    correlations = result['all_results']['correlation']
                    
                    plt.figure(figsize=(10, 6))
                    
                    # 绘制直方图
                    plt.hist(correlations, bins=30, alpha=0.7, color='skyblue', edgecolor='black')
                    plt.axvline(x=0, color='red', linestyle='--', alpha=0.8, label='No correlation')
                    plt.axvline(x=min_corr, color='orange', linestyle='--', alpha=0.8, label=f'Threshold ({min_corr})')
                    plt.axvline(x=-min_corr, color='orange', linestyle='--', alpha=0.8)
                    
                    plt.xlabel('Correlation Coefficient')
                    plt.ylabel('Number of Features')
                    plt.title(f'Distribution of Correlation Coefficients with {target_var}')
                    plt.legend()
                    plt.grid(alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'correlation_distribution_{target_var}.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
        except Exception as e:
            logger.warning(f"Failed to create correlation distribution plots: {e}")
    
    # 新增：绘制显著性统计图
    if plot_corr_matrix and len(results) > 0:
        try:
            for target_var, result in results.items():
                if len(result['all_results']) > 0:
                    # 统计不同显著性水平的特征数量
                    significance_counts = result['all_results']['significance'].value_counts()
                    
                    plt.figure(figsize=(8, 6))
                    
                    # 创建饼图 - 修复标签长度问题
                    colors = ['red', 'orange', 'yellow', 'lightgray']
                    labels = ['*** (p<0.001)', '** (p<0.01)', '* (p<0.05)', 'ns (p≥0.05)']
                    
                    # 确保所有显著性水平都存在
                    for label in labels:
                        if label not in significance_counts.index:
                            significance_counts[label] = 0
                    
                    # 只使用实际存在的显著性水平
                    actual_significance = significance_counts[significance_counts > 0]
                    actual_labels = []
                    actual_colors = []
                    
                    for i, label in enumerate(labels):
                        if label in actual_significance.index:
                            actual_labels.append(label)
                            actual_colors.append(colors[i])
                    
                    if len(actual_significance) > 0:
                        plt.pie(actual_significance.values, labels=actual_labels, colors=actual_colors, 
                               autopct='%1.1f%%', startangle=90)
                        plt.title(f'Significance Levels of Features vs {target_var}')
                        plt.axis('equal')
                        plt.tight_layout()
                        plt.savefig(os.path.join(output_dir, f'significance_pie_{target_var}.png'), 
                                   dpi=300, bbox_inches='tight')
                        plt.close()
                    else:
                        logger.warning(f"No significance data available for {target_var}")
                        plt.close()
        except Exception as e:
            logger.warning(f"Failed to create significance pie charts: {e}")
            import traceback
            logger.warning(f"Traceback: {traceback.format_exc()}")
    
    # 新增：绘制特征类型分析图
    if plot_corr_matrix and len(results) > 0:
        try:
            for target_var, result in results.items():
                if len(result['all_results']) > 0:
                    # 分析特征类型（基于特征名称）
                    feature_types = []
                    for feature in result['all_results']['feature']:
                        if '_mean' in feature:
                            feature_types.append('mean')
                        elif '_std' in feature:
                            feature_types.append('std')
                        elif '_min' in feature:
                            feature_types.append('min')
                        elif '_max' in feature:
                            feature_types.append('max')
                        elif '_median' in feature:
                            feature_types.append('median')
                        elif '_count' in feature:
                            feature_types.append('count')
                        else:
                            feature_types.append('other')
                    
                    result['all_results']['feature_type'] = feature_types
                    
                    # 按特征类型分组计算平均相关性
                    type_correlations = result['all_results'].groupby('feature_type')['correlation'].agg(['mean', 'count']).reset_index()
                    
                    plt.figure(figsize=(10, 6))
                    
                    # 创建条形图
                    bars = plt.bar(range(len(type_correlations)), type_correlations['mean'])
                    
                    # 设置颜色
                    colors = ['skyblue', 'lightgreen', 'lightcoral', 'gold', 'plum', 'lightgray']
                    for i, (bar, color) in enumerate(zip(bars, colors[:len(bars)])):
                        bar.set_color(color)
                    
                    plt.xlabel('Feature Type')
                    plt.ylabel('Average Correlation Coefficient')
                    plt.title(f'Average Correlation by Feature Type vs {target_var}')
                    plt.xticks(range(len(type_correlations)), type_correlations['feature_type'])
                    
                    # 添加数量标签
                    for i, (mean_val, count_val) in enumerate(zip(type_correlations['mean'], type_correlations['count'])):
                        plt.text(i, mean_val + (0.01 if mean_val > 0 else -0.01), f'n={count_val}', 
                               ha='center', va='bottom' if mean_val > 0 else 'top')
                    
                    plt.grid(axis='y', alpha=0.3)
                    plt.tight_layout()
                    plt.savefig(os.path.join(output_dir, f'feature_type_analysis_{target_var}.png'), 
                               dpi=300, bbox_inches='tight')
                    plt.close()
        except Exception as e:
            logger.warning(f"Failed to create feature type analysis plots: {e}")
    
    # 生成实验摘要
    summary_parts = []
    summary_parts.append(f"Correlation Analysis Results")
    summary_parts.append(f"Method: {method}")
    summary_parts.append(f"Minimum correlation threshold: {min_corr}")
    summary_parts.append(f"")
    
    for target_var, result in results.items():
        summary_parts.append(f"Target Variable: {target_var}")
        summary_parts.append(f"  Total features analyzed: {result['total_features']}")
        summary_parts.append(f"  Significant features (|corr| >= {min_corr}): {result['significant_count']}")
        
        if len(result['all_results']) > 0:
            summary_parts.append(f"  Top 5 correlations:")
            for _, row in result['all_results'].head(5).iterrows():
                summary_parts.append(f"    {row['feature']}: {row['correlation']:.4f} (p={row['p_value']:.4f})")
        summary_parts.append("")
    
    # 添加可视化文件说明
    summary_parts.append("Generated Visualization Files:")
    summary_parts.append("  - correlation_matrix.png: Overall correlation matrix")
    for target_var in results.keys():
        summary_parts.append(f"  - scatter_{target_var}.png: Scatter plots for top features")
        summary_parts.append(f"  - correlation_bars_{target_var}.png: Top 20 correlated features bar chart")
        summary_parts.append(f"  - correlation_distribution_{target_var}.png: Correlation coefficient distribution")
        summary_parts.append(f"  - significance_pie_{target_var}.png: Significance levels pie chart")
        summary_parts.append(f"  - feature_type_analysis_{target_var}.png: Feature type analysis")
    summary_parts.append("")
    
    summary = "\n".join(summary_parts)
    
    return summary


def merge_features_and_metadata(df_feat: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """Merge feature matrix and metadata"""
    if 'recording_id' not in df_feat.columns:
        raise ValueError("Feature matrix missing 'recording_id' column")
    
    df_feat['recording_id'] = df_feat['recording_id'].astype(int)
    df_meta['recording_id'] = df_meta['recording_id'].astype(int)
    
    df_combined = pd.merge(df_feat, df_meta, on='recording_id', how='inner')
    
    logger.info(f"Merged data shape: {df_combined.shape}")
    logger.info(f"Number of recordings: {len(df_combined)}")
    return df_combined


def preprocess_data(df: pd.DataFrame, target_vars: list) -> pd.DataFrame:
    """Data preprocessing for recording-level analysis"""
    df_processed = df.copy()
    
    logger.info(f"Initial data shape: {df_processed.shape}")
    
    # Process target variables
    for var in target_vars:
        if var in df_processed.columns:
            if var == 'sex':
                df_processed[var] = df_processed[var].map({'M': 1, 'F': 0, 'Male': 1, 'Female': 0})
            elif var in ['age', 'age_days']:
                df_processed[var] = pd.to_numeric(df_processed[var], errors='coerce')
    
    # 检查缺失值情况
    missing_ratio = df_processed.isnull().sum() / len(df_processed)
    logger.info(f"Missing value statistics:")
    logger.info(f"- Columns with >50% missing: {len(missing_ratio[missing_ratio > 0.5])}")
    logger.info(f"- Columns with >80% missing: {len(missing_ratio[missing_ratio > 0.8])}")
    logger.info(f"- Columns with >90% missing: {len(missing_ratio[missing_ratio > 0.9])}")
    
    # 降低缺失值阈值
    columns_to_drop_missing = missing_ratio[missing_ratio > 0.9].index
    df_processed = df_processed.drop(columns=columns_to_drop_missing)
    logger.info(f"Columns with >90% missing values: {len(columns_to_drop_missing)}")
    logger.info(f"Shape after removing high-missing columns: {df_processed.shape}")
    
    # Remove non-numeric columns
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    non_numeric_columns = df_processed.select_dtypes(exclude=[np.number]).columns
    
    logger.info(f"Numeric columns: {len(numeric_columns)}")
    logger.info(f"Non-numeric columns: {len(non_numeric_columns)}")
    
    # 保留recording_id和目标变量
    columns_to_keep = ['recording_id'] + target_vars
    columns_to_remove = [col for col in non_numeric_columns if col not in columns_to_keep]
    
    df_processed = df_processed.drop(columns=columns_to_remove)
    logger.info(f"Non-numeric columns removed: {len(columns_to_remove)}")
    logger.info(f"Shape after removing non-numeric columns: {df_processed.shape}")
    
    return df_processed


def perform_correlation_analysis(df: pd.DataFrame, target_vars: list, method: str, 
                                min_corr: float, top_n: int) -> dict:
    """Perform correlation analysis on recording-level features"""
    results = {}
    
    # 添加调试信息
    logger.info(f"Target variables: {target_vars}")
    
    # Get feature columns (exclude recording_id and target_vars)
    feature_cols = [col for col in df.columns if col not in ['recording_id'] + target_vars]
    
    logger.info(f"Feature columns: {len(feature_cols)}")
    
    # 检查是否有特征列
    if len(feature_cols) == 0:
        logger.warning("WARNING: No feature columns available for analysis!")
        logger.warning("This means all columns are either recording_id or target variables")
        # 为每个目标变量创建空结果
        for target_var in target_vars:
            if target_var in df.columns:
                results[target_var] = {
                    'all_results': pd.DataFrame(),
                    'top_results': pd.DataFrame(),
                    'significant_count': 0,
                    'total_features': 0
                }
        return results
    
    for target_var in target_vars:
        if target_var not in df.columns:
            logger.warning(f"Warning: Target variable {target_var} not in data")
            continue
        
        logger.info(f"Analyzing target variable: {target_var}")
        logger.info(f"Number of features to analyze: {len(feature_cols)}")
        
        correlations = []
        for feature in feature_cols:
            # 移除缺失值
            valid_data = df[[feature, target_var]].dropna()
            
            if len(valid_data) < 10:  # 至少需要10个有效数据点
                continue
            
            try:
                if target_var in ['sex', 'race', 'ethnicity']:
                    # 对于分类变量，使用点双列相关系数（point-biserial correlation）
                    # 这相当于皮尔逊相关系数，但更适合二分类变量
                    corr, p_value = stats.pearsonr(valid_data[feature], valid_data[target_var])
                    correlation_type = 'point-biserial'
                else:
                    # 对于连续变量，使用指定的相关系数方法
                    if method == 'pearson':
                        corr, p_value = stats.pearsonr(valid_data[feature], valid_data[target_var])
                        correlation_type = 'pearson'
                    elif method == 'spearman':
                        corr, p_value = stats.spearmanr(valid_data[feature], valid_data[target_var])
                        correlation_type = 'spearman'
                    else:
                        logger.warning(f"Unknown correlation method: {method}, using pearson")
                        corr, p_value = stats.pearsonr(valid_data[feature], valid_data[target_var])
                        correlation_type = 'pearson'
                
                # 检查p-value是否有效
                if np.isnan(p_value) or p_value < 0 or p_value > 1:
                    logger.warning(f"Invalid p-value for feature {feature}: {p_value}")
                    continue
                
                # 判断显著性
                if p_value < 0.001:
                    significance = '***'
                elif p_value < 0.01:
                    significance = '**'
                elif p_value < 0.05:
                    significance = '*'
                else:
                    significance = 'ns'
                
                correlations.append({
                    'feature': feature,
                    'correlation': corr,
                    'p_value': p_value,
                    'significance': significance,
                    'n_samples': len(valid_data),
                    'correlation_type': correlation_type
                })
            except Exception as e:
                logger.warning(f"Failed to calculate correlation for feature {feature}: {e}")
                continue
        
        # Create result DataFrame
        result_df = pd.DataFrame(correlations)
        result_df = result_df.sort_values('correlation', ascending=False)
        result_df = result_df[result_df['correlation'] >= min_corr]
        
        results[target_var] = {
            'all_results': result_df,
            'top_results': result_df.head(top_n),
            'significant_count': len(result_df[result_df['significance'] != 'ns']),
            'total_features': len(feature_cols)
        }
        
        logger.info(f"Found {len(result_df)} features with correlation >= {min_corr}")
        logger.info(f"Significant features: {results[target_var]['significant_count']}")
        
        # 添加调试：显示前几个相关性结果
        if len(result_df) > 0:
            logger.info(f"Top correlations:")
            for _, row in result_df.head(5).iterrows():
                logger.info(f"  {row['feature']}: corr={row['correlation']:.4f}, p={row['p_value']:.4f}")
    
    return results


def get_significance_level(p_value: float) -> str:
    """Get significance level"""
    if p_value < 0.001:
        return '***'
    elif p_value < 0.01:
        return '**'
    elif p_value < 0.05:
        return '*'
    else:
        return 'ns'


def save_correlation_results(results: dict, output_dir: str, save_detailed_results: bool = True):
    """Save correlation analysis results"""
    for target_var, result in results.items():
        if save_detailed_results:
            all_results_path = os.path.join(output_dir, f"correlation_{target_var}_all.csv")
            result['all_results'].to_csv(all_results_path, index=False)
        
        top_results_path = os.path.join(output_dir, f"correlation_{target_var}_top.csv")
        result['top_results'].to_csv(top_results_path, index=False)
    
    summary_data = []
    for target_var, result in results.items():
        # 添加安全检查，避免除零错误
        total_features = result['total_features']
        significant_count = result['significant_count']
        significant_ratio = significant_count / total_features if total_features > 0 else 0.0
        
        summary_data.append({
            'target_variable': target_var,
            'total_features': total_features,
            'significant_features': significant_count,
            'significant_ratio': significant_ratio
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(output_dir, "correlation_summary.csv")
    summary_df.to_csv(summary_path, index=False)


def plot_correlation_matrix(results: dict, output_dir: str):
    """Plot correlation matrix for recording-level features"""
    for target_var, result in results.items():
        if len(result['top_results']) == 0:
            continue
        
        plt.figure(figsize=(12, 8))
        
        top_features = result['top_results']['feature'].tolist()
        correlations = result['top_results']['correlation'].tolist()
        p_values = result['top_results']['p_value'].tolist()
        
        colors = ['red' if p < 0.05 else 'lightcoral' for p in p_values]
        
        bars = plt.barh(range(len(top_features)), correlations, color=colors)
        
        for i, (corr, p_val) in enumerate(zip(correlations, p_values)):
            if p_val < 0.05:
                plt.text(corr + (0.01 if corr > 0 else -0.01), i, 
                        get_significance_level(p_val), 
                        va='center', ha='left' if corr > 0 else 'right')
        
        plt.yticks(range(len(top_features)), top_features)
        plt.xlabel(f'Correlation ({target_var})')
        plt.title(f'Top correlated recording-level features with {target_var}')
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"correlation_matrix_{target_var}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def plot_scatter_plots(df: pd.DataFrame, results: dict, output_dir: str, top_n: int):
    """Plot scatter plots for recording-level features"""
    for target_var, result in results.items():
        if target_var not in df.columns or len(result['top_results']) == 0:
            continue
        
        top_features = result['top_results']['feature'].head(min(top_n, 6)).tolist()
        
        if len(top_features) == 0:
            continue
        
        n_features = len(top_features)
        n_cols = min(3, n_features)
        n_rows = (n_features + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=(5*n_cols, 4*n_rows))
        if n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten()
        
        for i, feature in enumerate(top_features):
            if i >= len(axes):
                break
                
            ax = axes[i]
            
            ax.scatter(df[target_var], df[feature], alpha=0.6, s=20)
            
            z = np.polyfit(df[target_var], df[feature], 1)
            p = np.poly1d(z)
            ax.plot(df[target_var], p(df[target_var]), "r--", alpha=0.8)
            
            corr = result['top_results'][result['top_results']['feature'] == feature]['correlation'].iloc[0]
            p_val = result['top_results'][result['top_results']['feature'] == feature]['p_value'].iloc[0]
            
            ax.set_xlabel(target_var)
            ax.set_ylabel(feature)
            ax.set_title(f'{feature}\nr={corr:.3f}, p={p_val:.3e}')
            ax.grid(True, alpha=0.3)
        
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"scatter_plots_{target_var}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def generate_summary(results: dict, df: pd.DataFrame, target_vars: list, method: str) -> str:
    """Generate experiment summary for recording-level analysis"""
    summary_lines = []
    summary_lines.append("=" * 50)
    summary_lines.append("Recording-Level Correlation Analysis Experiment Summary")
    summary_lines.append("=" * 50)
    summary_lines.append(f"Method: {method}")
    summary_lines.append(f"Number of recordings: {len(df)}")
    summary_lines.append(f"Number of recording-level features: {len([col for col in df.columns if col not in ['recording_id'] + target_vars])}")
    summary_lines.append("")
    
    for target_var in target_vars:
        if target_var in results:
            result = results[target_var]
            summary_lines.append(f"Target variable: {target_var}")
            summary_lines.append(f"  Total recording-level features: {result['total_features']}")
            summary_lines.append(f"  Significant features: {result['significant_count']}")
            
            # 修复除零错误
            total_features = result['total_features']
            significant_count = result['significant_count']
            if total_features > 0:
                significant_ratio = significant_count / total_features
                summary_lines.append(f"  Significant ratio: {significant_ratio:.2%}")
            else:
                summary_lines.append(f"  Significant ratio: N/A (no features available)")
            
            if len(result['top_results']) > 0:
                summary_lines.append("  Top 5 correlated recording-level features:")
                for _, row in result['top_results'].head(5).iterrows():
                    summary_lines.append(f"    {row['feature']}: r={row['correlation']:.3f}, p={row['p_value']:.3e}")
            else:
                summary_lines.append("  No features met the correlation threshold")
            summary_lines.append("")
    
    summary_lines.append("Result files:")
    summary_lines.append("  - correlation_*.csv: Correlation analysis results")
    summary_lines.append("  - correlation_summary.csv: Summary results")
    summary_lines.append("  - correlation_matrix_*.png: Correlation matrix plots")
    summary_lines.append("  - scatter_plots_*.png: Scatter plots")
    summary_lines.append("")
    summary_lines.append("Note: Features are aggregated at recording level (mean, std, min, max, median, count)")
    
    return "\n".join(summary_lines)
