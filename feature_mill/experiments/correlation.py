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
import time
from logging_config import logger  # 使用全局logger
import json # Added for saving results_index.json
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
            - generate_plots: 是否生成图表，默认True
    
    Returns:
        str: 实验摘要
    """
    # 获取参数
    target_vars = kwargs.get('target_vars', ['age'])  # 默认只分析age，不自动包含sex
    method = kwargs.get('method', 'pearson')
    min_corr = kwargs.get('min_corr', 0.3)
    top_n = kwargs.get('top_n', 20)
    generate_plots = kwargs.get('generate_plots', True)
    
    # 验证target_vars参数
    if not target_vars:
        logger.warning("No target variables specified, defaulting to ['age']")
        target_vars = ['age']
    
    logger.info(f"Target variables to analyze: {target_vars}")
    
    logger.info(f"Start correlation analysis experiment")
    logger.info(f"Feature matrix shape: {df_feat.shape}")
    logger.info(f"Metadata shape: {df_meta.shape}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 合并特征矩阵和元数据
    logger.info(f"Before merge - Feature matrix shape: {df_feat.shape}")
    logger.info(f"Before merge - Metadata shape: {df_meta.shape}")
    logger.info(f"Feature matrix recording_id range: {df_feat['recording_id'].min()} to {df_feat['recording_id'].max()}")
    logger.info(f"Metadata recording_id range: {df_meta['recording_id'].min()} to {df_meta['recording_id'].max()}")
    
    # 检查recording_id的唯一性
    feat_unique_ids = df_feat['recording_id'].nunique()
    meta_unique_ids = df_meta['recording_id'].nunique()
    logger.info(f"Feature matrix unique recording_ids: {feat_unique_ids}")
    logger.info(f"Metadata unique recording_ids: {meta_unique_ids}")
    
    # 使用inner join确保数据对齐
    df_combined = pd.merge(df_feat, df_meta, on='recording_id', how='inner')
    
    logger.info(f"After merge - Combined data shape: {df_combined.shape}")
    logger.info(f"Number of recordings after merge: {len(df_combined)}")
    
    # 验证数据对齐
    if len(df_combined) != len(df_feat):
        logger.warning(f"WARNING: Data alignment issue! Feature matrix has {len(df_feat)} rows, but merged data has {len(df_combined)} rows")
        logger.warning("This suggests some recording_ids in feature matrix don't exist in metadata")
    
    # 数据预处理
    df_processed = df_combined.copy()
    
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
    
    # 处理分类变量（只处理目标变量，不处理其他分类变量）
    for col in non_numeric_columns:
        if col in df_processed.columns and col != 'recording_id' and col in target_vars:
            logger.info(f"Processing categorical target variable: {col}")
            le = LabelEncoder()
            df_processed[col] = le.fit_transform(df_processed[col].astype(str))
        elif col in df_processed.columns and col != 'recording_id':
            logger.debug(f"Skipping non-target categorical variable: {col}")
            # 移除非目标变量的分类列，避免后续处理问题
            df_processed = df_processed.drop(columns=[col])
            logger.info(f"Removed non-target categorical variable: {col}")
    
    # 填充缺失值
    df_processed = df_processed.fillna(df_processed.median())
    
    # 数据质量检查
    logger.info("Data quality check after preprocessing:")
    
    # 检查无穷大值
    inf_counts = df_processed.isin([np.inf, -np.inf]).sum()
    if inf_counts.sum() > 0:
        logger.warning(f"Found infinite values in columns: {inf_counts[inf_counts > 0].to_dict()}")
    
    # 检查NaN值
    nan_counts = df_processed.isnull().sum()
    if nan_counts.sum() > 0:
        logger.warning(f"Found NaN values in columns: {nan_counts[nan_counts > 0].to_dict()}")
    
    # 检查零方差特征（排除目标变量）
    zero_var_features = []
    for col in df_processed.columns:
        if col not in ['recording_id'] + target_vars:
            if df_processed[col].std() == 0:
                zero_var_features.append(col)
    
    if zero_var_features:
        logger.warning(f"Found {len(zero_var_features)} features with zero variance: {zero_var_features[:5]}")
    
    # 检查目标变量的数据质量
    logger.info("Target variables data quality:")
    for target_var in target_vars:
        if target_var in df_processed.columns:
            target_data = df_processed[target_var]
            logger.info(f"  {target_var}: type={target_data.dtype}, unique_values={target_data.nunique()}, missing={target_data.isnull().sum()}")
            if target_data.dtype == 'object':
                logger.info(f"    Sample values: {target_data.dropna().unique()[:5].tolist()}")
    
    logger.info(f"Final processed data shape: {df_processed.shape}")
    
    # 验证数据对齐的正确性
    validate_data_alignment(df_feat, df_meta, df_processed, target_vars)
    
    # 执行相关性分析
    results = perform_correlation_analysis(df_processed, target_vars, method, min_corr, top_n)
    
    # 保存结果
    save_correlation_results(results, output_dir)
    
    # 可视化
    if generate_plots:
        plot_correlation_matrix(results, output_dir, min_corr)
        plot_scatter_plots(df_processed, results, output_dir, top_n, min_corr)
    
    # 生成摘要
    summary = generate_summary(results, df_processed, target_vars, method)
    
    logger.info(f"Correlation analysis completed, results saved to: {output_dir}")
    return summary


def validate_data_alignment(df_feat: pd.DataFrame, df_meta: pd.DataFrame, df_processed: pd.DataFrame, target_vars: list):
    """Validate that data alignment is correct"""
    logger.info("=" * 50)
    logger.info("DATA ALIGNMENT VALIDATION")
    logger.info("=" * 50)
    
    # 检查recording_id的一致性
    feat_ids = set(df_feat['recording_id'].unique())
    meta_ids = set(df_meta['recording_id'].unique())
    processed_ids = set(df_processed['recording_id'].unique())
    
    # 交集：两个数据集都有的recording_ids
    common_ids = feat_ids.intersection(meta_ids)
    
    # 只在feature matrix中有的
    feat_only = feat_ids - meta_ids
    # 只在metadata中有的
    meta_only = meta_ids - feat_ids
    
    logger.info(f"Feature matrix unique recording_ids: {len(feat_ids)}")
    logger.info(f"Metadata unique recording_ids: {len(meta_ids)}")
    logger.info(f"Common recording_ids: {len(common_ids)}")
    logger.info(f"Only in feature matrix: {len(feat_only)}")
    logger.info(f"Only in metadata: {len(feat_only)}")
    logger.info(f"Final processed recording_ids: {len(processed_ids)}")
    
    if len(feat_only) > 0:
        logger.warning(f"WARNING: {len(feat_only)} recording_ids in feature matrix not found in metadata")
        logger.warning(f"Sample missing IDs: {list(feat_only)[:5]}")
    
    if len(meta_only) > 0:
        logger.warning(f"WARNING: {len(meta_only)} recording_ids in metadata not found in feature matrix")
        logger.warning(f"Sample missing IDs: {list(meta_only)[:5]}")
    
    # 验证目标变量的分布
    for target_var in target_vars:
        if target_var in df_processed.columns:
            target_values = df_processed[target_var]
            logger.info(f"Target variable '{target_var}' statistics:")
            logger.info(f"  - Non-null values: {target_values.count()}")
            logger.info(f"  - Unique values: {target_values.nunique()}")
            logger.info(f"  - Range: {target_values.min()} to {target_values.max()}")
            logger.info(f"  - Sample values: {target_values.dropna().head(3).tolist()}")
    
    logger.info("=" * 50)


def merge_features_and_metadata(df_feat: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """Merge feature matrix and metadata with proper alignment"""
    if 'recording_id' not in df_feat.columns:
        raise ValueError("Feature matrix missing 'recording_id' column")
    if 'recording_id' not in df_meta.columns:
        raise ValueError("Metadata missing 'recording_id' column")
    
    # 确保recording_id是整数类型
    df_feat['recording_id'] = df_feat['recording_id'].astype(int)
    df_meta['recording_id'] = df_meta['recording_id'].astype(int)
    
    # 检查数据完整性
    feat_missing = df_feat['recording_id'].isnull().sum()
    meta_missing = df_meta['recording_id'].isnull().sum()
    
    if feat_missing > 0:
        logger.warning(f"Feature matrix has {feat_missing} missing recording_ids")
    if meta_missing > 0:
        logger.warning(f"Metadata has {meta_missing} missing recording_ids")
    
    # 使用inner join确保数据对齐
    df_combined = pd.merge(df_feat, df_meta, on='recording_id', how='inner')
    
    # 验证合并结果
    if len(df_combined) == 0:
        raise ValueError("No matching recording_ids found between feature matrix and metadata!")
    
    if len(df_combined) < len(df_feat):
        logger.warning(f"Data loss detected: {len(df_feat) - len(df_combined)} recordings from feature matrix were not found in metadata")
    
    logger.info(f"Successfully merged {len(df_combined)} recordings")
    logger.info(f"Final merged data shape: {df_combined.shape}")
    
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
    
    # Get feature columns (exclude recording_id, target_vars, and count-related features)
    feature_cols = [col for col in df.columns if col not in ['recording_id'] + target_vars]
    
    # 排除count相关的特征，这些通常不是有意义的特征值
    feature_cols = [col for col in feature_cols if not col.endswith('_count')]
    
    logger.info(f"Feature columns: {len(feature_cols)}")
    logger.info(f"Excluded count-related features, remaining features: {len(feature_cols)}")
    
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
        logger.info(f"Target variable type: {df[target_var].dtype}")
        logger.info(f"Target variable unique values: {df[target_var].nunique()}")
        logger.info(f"Target variable missing values: {df[target_var].isnull().sum()}")
        logger.info(f"Sample target values: {df[target_var].head().tolist()}")
        
        # 验证数据对齐：检查特征和目标变量的长度是否一致
        target_length = len(df[target_var])
        logger.info(f"Target variable length: {target_length}")
        logger.info(f"DataFrame total length: {len(df)}")
        
        if target_length != len(df):
            logger.error(f"ERROR: Data alignment issue! Target variable length ({target_length}) != DataFrame length ({len(df)})")
        
        logger.info(f"Sample feature values (first 3 features):")
        for i, feature in enumerate(feature_cols[:3]):
            feature_length = len(df[feature])
            logger.info(f"  {feature}: type={df[feature].dtype}, length={feature_length}, missing={df[feature].isnull().sum()}, sample={df[feature].head().tolist()}")
            if feature_length != target_length:
                logger.error(f"    ERROR: Feature {feature} length ({feature_length}) != target length ({target_length})")
        
        correlations = []
        start_time = time.time()
        
        for i, feature in enumerate(feature_cols):
            # 每50个特征输出一次进度
            if i % 50 == 0:
                elapsed = time.time() - start_time
                logger.info(f"进度: {i}/{len(feature_cols)} ({i/len(feature_cols)*100:.1f}%), 耗时: {elapsed:.1f}s")
            
            # 移除缺失值
            valid_data = df[[feature, target_var]].dropna()
            
            if len(valid_data) < 10:  # 至少需要10个有效数据点
                logger.debug(f"Skipping {feature}: insufficient valid data points ({len(valid_data)} < 10)")
                continue
            
            # 数据质量检查
            feature_values = valid_data[feature]
            target_values = valid_data[target_var]
            
            # 检查是否有足够的变异
            if feature_values.std() == 0:
                logger.debug(f"Skipping {feature}: feature has zero variance")
                continue
            
            if target_values.std() == 0:
                logger.debug(f"Skipping {feature}: target variable has zero variance")
                continue
            
            # 检查是否有无穷大或NaN值
            if np.any(np.isinf(feature_values)) or np.any(np.isnan(feature_values)):
                logger.debug(f"Skipping {feature}: feature contains inf or NaN values")
                continue
            
            if np.any(np.isinf(target_values)) or np.any(np.isnan(target_values)):
                logger.debug(f"Skipping {feature}: target variable contains inf or NaN values")
                continue
            
            try:
                # 额外的数据质量检查：确保数据适合相关性分析
                if len(feature_values) < 3:
                    logger.debug(f"Skipping {feature}: insufficient data points for correlation")
                    continue
                
                # 检查数据是否都是数值型
                if not np.issubdtype(feature_values.dtype, np.number) or not np.issubdtype(target_values.dtype, np.number):
                    logger.debug(f"Skipping {feature}: non-numeric data types")
                    continue
                
                if target_var in ['sex', 'race', 'ethnicity']:
                    # 对于分类变量，使用点双列相关系数（point-biserial correlation）
                    # 这相当于皮尔逊相关系数，但更适合二分类变量
                    corr, p_value = stats.pearsonr(feature_values, target_values)
                    correlation_type = 'point-biserial'
                else:
                    # 对于连续变量，使用指定的相关系数方法
                    if method == 'pearson':
                        corr, p_value = stats.pearsonr(feature_values, target_values)
                        correlation_type = 'pearson'
                    elif method == 'spearman':
                        corr, p_value = stats.spearmanr(feature_values, target_values)
                        correlation_type = 'spearman'
                    else:
                        logger.warning(f"Unknown correlation method: {method}, using pearson")
                        corr, p_value = stats.pearsonr(feature_values, target_values)
                        correlation_type = 'pearson'
                
                # 严格检查相关系数和p-value是否有效
                if np.isnan(corr) or np.isinf(corr):
                    logger.debug(f"Skipping {feature}: invalid correlation coefficient ({corr})")
                    continue
                
                if np.isnan(p_value) or np.isinf(p_value):
                    logger.debug(f"Skipping {feature}: invalid p-value ({p_value})")
                    continue
                
                # 检查p-value范围（应该是0到1之间）
                if not (0 <= p_value <= 1):
                    logger.debug(f"Skipping {feature}: p-value out of range ({p_value})")
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
                logger.debug(f"Failed to calculate correlation for feature {feature}: {e}")
                # 记录更详细的错误信息用于调试
                if len(valid_data) > 0:
                    logger.debug(f"  Feature stats: mean={feature_values.mean():.6f}, std={feature_values.std():.6f}, range=[{feature_values.min():.6f}, {feature_values.max():.6f}]")
                    logger.debug(f"  Target stats: mean={target_values.mean():.6f}, std={target_values.std():.6f}, range=[{target_values.min():.6f}, {target_values.max():.6f}]")
                continue
        
        # Create result DataFrame
        if len(correlations) > 0:
            result_df = pd.DataFrame(correlations)
            # 按绝对相关性排序，这样可以更好地展示最重要的特征
            result_df = result_df.sort_values('correlation', key=abs, ascending=False)
            # 使用绝对值来过滤相关性，这样可以捕获正相关和负相关
            result_df = result_df[result_df['correlation'].abs() >= min_corr]
            
            significant_count = len(result_df[result_df['significance'] != 'ns'])
            significant_ratio = significant_count / len(feature_cols) if len(feature_cols) > 0 else 0
        else:
            # 如果没有相关性结果，创建空的DataFrame
            result_df = pd.DataFrame(columns=['feature', 'correlation', 'p_value', 'significance', 'n_samples', 'correlation_type'])
            significant_count = 0
            significant_ratio = 0.0
            logger.warning(f"No valid correlations found for target variable {target_var}")
        
        results[target_var] = {
            'all_results': result_df,
            'top_results': result_df.head(top_n),
            'significant_count': significant_count,
            'total_features': len(feature_cols),
            'significant_ratio': significant_ratio
        }
        
        logger.info(f"Found {len(result_df)} features with |correlation| >= {min_corr}")
        logger.info(f"Significant features: {results[target_var]['significant_count']}")
        
        # 添加调试：显示前几个相关性结果
        if len(result_df) > 0:
            logger.info(f"Top correlations:")
            for _, row in result_df.head(5).iterrows():
                logger.info(f"  {row['feature']}: corr={row['correlation']:.4f}, p={row['p_value']:.4f}")
        else:
            logger.info(f"No features found with |correlation| >= {min_corr}")
            # 显示一些相关性较低的特征作为参考
            if len(correlations) > 0:
                temp_df = pd.DataFrame(correlations)
                temp_df = temp_df.sort_values('correlation', key=abs, ascending=False)
                logger.info(f"Top 3 features by absolute correlation (below threshold):")
                for _, row in temp_df.head(3).iterrows():
                    logger.info(f"  {row['feature']}: corr={row['correlation']:.4f}, p={row['p_value']:.4f}")
        
        # 数据质量总结
        logger.info(f"Data quality summary for {target_var}:")
        logger.info(f"  - Total features analyzed: {len(feature_cols)}")
        logger.info(f"  - Features with valid correlations: {len(correlations)}")
        logger.info(f"  - Features meeting threshold: {len(result_df)}")
        logger.info(f"  - Success rate: {len(correlations)/len(feature_cols)*100:.1f}%")
    
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


def save_correlation_results(results: dict, output_dir: str):
    """Save correlation analysis results with organized file structure"""
    
    # 创建子目录
    data_dir = os.path.join(output_dir, "data")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 保存每个目标变量的相关性结果
    for target_var, result in results.items():
        # 即使没有满足阈值的结果，也创建文件（可能为空）
        result_file = os.path.join(data_dir, f"correlation_{target_var}.csv")
        if len(result['top_results']) > 0:
            result['top_results'].to_csv(result_file, index=False)
        else:
            # 创建空的DataFrame并保存
            empty_df = pd.DataFrame(columns=['feature', 'correlation', 'p_value', 'significance', 'n_samples', 'correlation_type'])
            empty_df.to_csv(result_file, index=False)
            logger.info(f"Created empty correlation file for {target_var} (no features met threshold)")
    
    # 保存汇总结果
    summary_data = []
    for target_var, result in results.items():
        summary_data.append({
            'target_variable': target_var,
            'total_features': result['total_features'],
            'significant_features': result['significant_count'],
            'significant_ratio': result['significant_ratio']
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(data_dir, "correlation_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # 创建结果索引文件
    results_index = {
        "experiment_type": "correlation",
        "files": {
            "summary": "data/correlation_summary.csv"
        },
        "plots": {},
        "summary": {
            "target_variables": list(results.keys()),
            "total_targets": len(results),
            "generated_at": datetime.now().isoformat()
        }
    }
    
    # 添加每个目标变量的文件
    for target_var, result in results.items():
        results_index["files"][f"correlation_{target_var}"] = f"data/correlation_{target_var}.csv"
        results_index["plots"][f"correlation_matrix_{target_var}"] = f"plots/correlation_matrix_{target_var}.png"
        results_index["plots"][f"scatter_plots_{target_var}"] = f"plots/scatter_plots_{target_var}.png"
    
    with open(os.path.join(output_dir, "results_index.json"), "w", encoding='utf-8') as f:
        json.dump(results_index, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Correlation results saved to {output_dir} with organized structure")


def plot_correlation_matrix(results: dict, output_dir: str, min_corr: float = 0.3):
    """Plot correlation matrix for recording-level features"""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    for target_var, result in results.items():
        plt.figure(figsize=(12, 8))
        
        if len(result['top_results']) == 0:
            # 即使没有满足阈值的结果，也生成一个提示图
            plt.text(0.5, 0.5, f'No features met the correlation threshold for {target_var}\n\n'
                     f'Total features analyzed: {result["total_features"]}\n'
                     f'Significant features: {result["significant_count"]}\n'
                     f'Try lowering the min_corr threshold or check data quality.',
                     ha='center', va='center', transform=plt.gca().transAxes,
                     fontsize=14, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            plt.title(f'Correlation Analysis Results for {target_var} - No Significant Correlations')
            plt.axis('off')
        else:
            # 有结果时正常绘制
            top_features = result['top_results']['feature'].tolist()
            correlations = result['top_results']['correlation'].tolist()
            p_values = result['top_results']['p_value'].tolist()
            
            # 为负相关添加不同颜色
            colors = []
            for corr, p_val in zip(correlations, p_values):
                if p_val < 0.05:
                    if corr > 0:
                        colors.append('red')  # 显著正相关
                    else:
                        colors.append('blue')  # 显著负相关
                else:
                    if corr > 0:
                        colors.append('lightcoral')  # 非显著正相关
                    else:
                        colors.append('lightblue')  # 非显著负相关
            
            bars = plt.barh(range(len(top_features)), correlations, color=colors)
            
            for i, (corr, p_val) in enumerate(zip(correlations, p_values)):
                if p_val < 0.05:
                    plt.text(corr + (0.01 if corr > 0 else -0.01), i, 
                            get_significance_level(p_val), 
                            va='center', ha='left' if corr > 0 else 'right')
            
            plt.yticks(range(len(top_features)), top_features)
            plt.xlabel(f'Correlation ({target_var})')
            plt.title(f'Top correlated recording-level features with {target_var} (|r| ≥ {min_corr})')
            plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"correlation_matrix_{target_var}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def plot_scatter_plots(df: pd.DataFrame, results: dict, output_dir: str, top_n: int, min_corr: float = 0.3):
    """Plot scatter plots for recording-level features"""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    for target_var, result in results.items():
        if target_var not in df.columns:
            continue
        
        if len(result['top_results']) == 0:
            # 即使没有满足阈值的结果，也生成一个提示图
            plt.figure(figsize=(10, 6))
            plt.text(0.5, 0.5, f'No features met the correlation threshold for {target_var}\n\n'
                     f'Total features analyzed: {result["total_features"]}\n'
                     f'Significant features: {result["significant_count"]}\n'
                     f'Try lowering the min_corr threshold or check data quality.',
                     ha='center', va='center', transform=plt.gca().transAxes,
                     fontsize=12, bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgray"))
            plt.title(f'Scatter Plot Analysis for {target_var} - No Significant Correlations')
            plt.axis('off')
            plt.tight_layout()
            plt.savefig(os.path.join(plots_dir, f"scatter_plots_{target_var}.png"), 
                       dpi=300, bbox_inches='tight')
            plt.close()
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
        plt.savefig(os.path.join(plots_dir, f"scatter_plots_{target_var}.png"), 
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
