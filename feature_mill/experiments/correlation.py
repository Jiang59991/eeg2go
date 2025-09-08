#!/usr/bin/env python3
"""
Associations Analysis Experiment Module

该模块实现EEG特征与目标变量的关联性分析，包括：
- 相关性分析：Pearson/Spearman相关系数
- 回归效应量：连续目标(β per-SD ± 95%CI)，二分类目标(OR per-SD ± 95%CI)
- 多重比较校正：BH-FDR方法
- 自动目标变量类型检测和相应分析方法选择

Focus: "How do features relate to targets?" - 提供稳健的关联性证据
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import r2_score
import statsmodels.api as sm
from statsmodels.stats.multitest import multipletests
from datetime import datetime
import time
from logging_config import logger
import json
warnings.filterwarnings('ignore')

# Set English font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run(df_feat: pd.DataFrame, df_meta: pd.DataFrame, output_dir: str, **kwargs) -> dict:
    """
    运行关联性分析实验
    
    Args:
        df_feat: 特征矩阵DataFrame
        df_meta: 元数据DataFrame
        output_dir: 输出目录
        **kwargs: 额外参数
            - target_vars: 目标变量列表，默认['age', 'sex']
            - correlation_method: 相关性分析方法，默认'pearson'
            - min_corr: 最小相关系数阈值，默认0.3
            - top_n: 显示前N个最相关特征，默认20
            - fdr_alpha: FDR校正的α水平，默认0.05
            - generate_plots: 是否生成图表，默认True
    
    Returns:
        dict: 实验结果的结构化数据，包含前端Summary Statistics需要的关键信息
    """
    # 获取参数并确保类型正确
    target_vars = kwargs.get('target_vars', ['age', 'sex'])
    correlation_method = kwargs.get('correlation_method', 'pearson')
    min_corr = float(kwargs.get('min_corr', 0.3))
    top_n = int(kwargs.get('top_n', 20))
    fdr_alpha = float(kwargs.get('fdr_alpha', 0.05))
    generate_plots = kwargs.get('generate_plots', True)
    
    logger.info(f"Start associations analysis experiment")
    logger.info(f"Feature matrix shape: {df_feat.shape}")
    logger.info(f"Metadata shape: {df_meta.shape}")
    
    # 创建输出目录
    os.makedirs(output_dir, exist_ok=True)
    
    # 合并特征矩阵和元数据
    df_combined = pd.merge(df_feat, df_meta, on='recording_id', how='inner')
    logger.info(f"Merged data shape: {df_combined.shape}")
    
    # 数据预处理
    df_processed = preprocess_data(df_combined, target_vars)
    logger.info(f"Processed data shape: {df_processed.shape}")
    
    # 执行关联性分析
    results = perform_associations_analysis(df_processed, target_vars, correlation_method, 
                                          min_corr, top_n, fdr_alpha)
    
    # 保存结果
    save_associations_results(results, output_dir)
    
    # 可视化
    if generate_plots:
        plot_associations_matrix(results, output_dir)
        plot_scatter_plots(df_processed, results, output_dir, top_n)
    
    # 生成结构化摘要
    summary_data = generate_structured_summary(results, df_processed, target_vars, 
                                             correlation_method, fdr_alpha)
    
    logger.info(f"Associations analysis completed, results saved to: {output_dir}")
    return summary_data


def preprocess_data(df: pd.DataFrame, target_vars: list) -> pd.DataFrame:
    """数据预处理"""
    df_processed = df.copy()
    
    logger.info(f"Initial data shape: {df_processed.shape}")
    logger.info(f"Target variables: {target_vars}")
    
    # 处理目标变量
    for var in target_vars:
        if var in df_processed.columns:
            logger.info(f"Processing target variable: {var}")
            logger.info(f"  Original type: {df_processed[var].dtype}")
            logger.info(f"  Unique values: {df_processed[var].nunique()}")
            logger.info(f"  Sample values: {df_processed[var].head().tolist()}")
            
            if var == 'sex':
                # 更灵活的性别编码
                sex_mapping = {
                    'M': 1, 'Male': 1, 'male': 1, '1': 1,
                    'F': 0, 'Female': 0, 'female': 0, '0': 0
                }
                df_processed[var] = df_processed[var].astype(str).map(sex_mapping)
                logger.info(f"  Sex encoded: {df_processed[var].value_counts().to_dict()}")
            elif var in ['age', 'age_days']:
                df_processed[var] = pd.to_numeric(df_processed[var], errors='coerce')
                logger.info(f"  Age converted to numeric, missing: {df_processed[var].isnull().sum()}")
    
    # 检查缺失值情况
    missing_ratio = df_processed.isnull().sum() / len(df_processed)
    logger.info(f"Missing value statistics:")
    logger.info(f"  Columns with >50% missing: {len(missing_ratio[missing_ratio > 0.5])}")
    logger.info(f"  Columns with >80% missing: {len(missing_ratio[missing_ratio > 0.8])}")
    logger.info(f"  Columns with >90% missing: {len(missing_ratio[missing_ratio > 0.9])}")
    
    # 移除缺失值过多的列
    columns_to_drop = missing_ratio[missing_ratio > 0.9].index
    df_processed = df_processed.drop(columns=columns_to_drop)
    logger.info(f"Columns with >90% missing values dropped: {len(columns_to_drop)}")
    logger.info(f"Shape after removing high-missing columns: {df_processed.shape}")
    
    # 只保留数值型列
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
    
    # 检查目标变量的最终状态
    for var in target_vars:
        if var in df_processed.columns:
            logger.info(f"Final target variable {var}:")
            logger.info(f"  Type: {df_processed[var].dtype}")
            logger.info(f"  Unique values: {df_processed[var].nunique()}")
            logger.info(f"  Missing: {df_processed[var].isnull().sum()}")
            logger.info(f"  Sample: {df_processed[var].dropna().head().tolist()}")
    
    # 填充缺失值（使用中位数，但保留目标变量的原始缺失情况）
    feature_columns = [col for col in df_processed.columns if col not in ['recording_id'] + target_vars]
    
    # 只对特征列填充缺失值
    for col in feature_columns:
        if df_processed[col].isnull().sum() > 0:
            median_val = df_processed[col].median()
            df_processed[col] = df_processed[col].fillna(median_val)
    
    logger.info(f"Final processed data shape: {df_processed.shape}")
    return df_processed


def detect_target_variable_type(df: pd.DataFrame, target_var: str) -> str:
    """检测目标变量类型"""
    if target_var not in df.columns:
        return 'unknown'
    
    # 获取非空值
    non_null_values = df[target_var].dropna()
    if len(non_null_values) == 0:
        return 'unknown'
    
    # 检查数据类型
    dtype = non_null_values.dtype
    
    # 对于数值类型，检查唯一值数量
    if np.issubdtype(dtype, np.number):
        unique_values = non_null_values.nunique()
        if unique_values == 2:
            return 'binary'
        elif unique_values <= 10:
            return 'categorical'
        else:
            return 'continuous'
    else:
        # 对于非数值类型，检查唯一值数量
        unique_values = non_null_values.nunique()
        if unique_values == 2:
            return 'binary'
        elif unique_values <= 10:
            return 'categorical'
        else:
            # 如果非数值类型有太多唯一值，可能是字符串编码的连续变量
            return 'categorical'


def perform_associations_analysis(df: pd.DataFrame, target_vars: list, correlation_method: str, 
                                min_corr: float, top_n: int, fdr_alpha: float) -> dict:
    """执行关联性分析"""
    results = {}
    
    # 获取特征列
    feature_cols = [col for col in df.columns if col not in ['recording_id'] + target_vars]
    logger.info(f"Analyzing {len(feature_cols)} features")
    
    for target_var in target_vars:
        if target_var not in df.columns:
            logger.warning(f"Target variable {target_var} not in data")
            continue
        
        logger.info(f"Analyzing target variable: {target_var}")
        
        # 检测目标变量类型
        target_type = detect_target_variable_type(df, target_var)
        logger.info(f"Target variable type: {target_type}")
        
        # 对于pearson相关性分析，只处理连续变量
        if correlation_method == 'pearson' and target_type in ['binary', 'categorical']:
            logger.warning(f"Skipping {target_var} (type: {target_type}) for pearson correlation. Use spearman for categorical variables.")
            continue
        
        # 执行相关性分析
        correlation_results = perform_correlation_analysis(df, target_var, feature_cols, 
                                                        correlation_method, min_corr)
        
        # 执行回归效应量分析
        regression_results = perform_regression_analysis(df, target_var, feature_cols, target_type)
        
        # 合并结果并应用FDR校正
        combined_results = combine_and_correct_results(correlation_results, regression_results, 
                                                     feature_cols, fdr_alpha)
        
        # 生成最终结果
        results[target_var] = {
            'target_type': target_type,
            'all_results': combined_results,
            'top_results': combined_results.head(top_n),
            'significant_count': len(combined_results[combined_results['q_value'] < fdr_alpha]),
            'total_features': len(feature_cols),
            'correlation_results': correlation_results,
            'regression_results': regression_results
        }
        
        logger.info(f"Found {results[target_var]['significant_count']} significant associations")
    
    return results


def perform_correlation_analysis(df: pd.DataFrame, target_var: str, feature_cols: list, 
                               method: str, min_corr: float) -> pd.DataFrame:
    """执行相关性分析"""
    correlations = []
    
    logger.info(f"Starting correlation analysis for {target_var}")
    logger.info(f"  Method: {method}")
    logger.info(f"  Target variable type: {df[target_var].dtype}")
    logger.info(f"  Target variable unique values: {df[target_var].nunique()}")
    logger.info(f"  Target variable missing: {df[target_var].isnull().sum()}")
    logger.info(f"  Target variable sample: {df[target_var].dropna().head().tolist()}")
    
    # 检查目标变量是否适合当前的相关性分析方法
    target_type = detect_target_variable_type(df, target_var)
    if method == 'pearson' and target_type in ['binary', 'categorical']:
        logger.warning(f"  Warning: {target_var} is {target_type} but using pearson correlation. Consider using spearman for categorical variables.")
    elif method == 'spearman' and target_type == 'continuous':
        logger.info(f"  Note: {target_var} is continuous, spearman correlation will work but pearson might be more appropriate.")
    
    for i, feature in enumerate(feature_cols):
        if i % 50 == 0:  # 每50个特征输出一次进度
            logger.info(f"  Progress: {i}/{len(feature_cols)} features processed")
        
        valid_data = df[[feature, target_var]].dropna()
        
        if len(valid_data) < 10:
            logger.debug(f"    {feature}: insufficient data ({len(valid_data)} samples)")
            continue
        
        try:
            if method == 'pearson':
                corr, p_value = pearsonr(valid_data[feature], valid_data[target_var])
            elif method == 'spearman':
                corr, p_value = spearmanr(valid_data[feature], valid_data[target_var])
            else:
                corr, p_value = pearsonr(valid_data[feature], valid_data[target_var])
            
            # 检查结果的有效性
            if (not np.isnan(corr) and not np.isnan(p_value) and 
                p_value >= 0 and p_value <= 1 and 
                abs(corr) <= 1):
                
                correlations.append({
                    'feature': feature,
                    'correlation': corr,
                    'p_value': p_value,
                    'n_samples': len(valid_data)
                })
                
                # 记录一些显著的相关性用于调试
                if abs(corr) > 0.3 or p_value < 0.05:
                    logger.debug(f"    {feature}: r={corr:.3f}, p={p_value:.6f}, n={len(valid_data)}")
            else:
                logger.debug(f"    {feature}: invalid correlation result (r={corr}, p={p_value})")
                
        except Exception as e:
            logger.debug(f"    {feature}: correlation failed - {e}")
            continue
    
    logger.info(f"  Correlation analysis completed: {len(correlations)} valid results")
    
    # 确保返回的DataFrame包含所有必要的列
    if correlations:
        result_df = pd.DataFrame(correlations)
        # 确保所有必要的列都存在
        required_columns = ['feature', 'correlation', 'p_value', 'n_samples']
        for col in required_columns:
            if col not in result_df.columns:
                result_df[col] = 0.0 if col in ['correlation', 'p_value'] else 0
        
        # 添加调试信息
        logger.info(f"  Correlation results summary:")
        logger.info(f"    Range: {result_df['correlation'].min():.3f} to {result_df['correlation'].max():.3f}")
        logger.info(f"    P-value range: {result_df['p_value'].min():.6f} to {result_df['p_value'].max():.6f}")
        logger.info(f"    Significant (p<0.05): {(result_df['p_value'] < 0.05).sum()}")
        
        return result_df
    else:
        logger.warning(f"  No valid correlations found for {target_var}")
        # 返回空的DataFrame但包含必要的列
        return pd.DataFrame(columns=['feature', 'correlation', 'p_value', 'n_samples'])


def perform_regression_analysis(df: pd.DataFrame, target_var: str, feature_cols: list, 
                              target_type: str) -> pd.DataFrame:
    """执行回归效应量分析"""
    regression_results = []
    
    # 标准化特征
    scaler = StandardScaler()
    features_scaled = scaler.fit_transform(df[feature_cols])
    features_scaled_df = pd.DataFrame(features_scaled, columns=feature_cols, index=df.index)
    
    for i, feature in enumerate(feature_cols):
        try:
            if target_type == 'continuous':
                # 线性回归：β per-SD ± 95%CI
                X = features_scaled_df[feature].values.reshape(-1, 1)
                y = df[target_var].values
                
                # 移除缺失值
                valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                if np.sum(valid_mask) < 10:
                    continue
                
                X_valid = X[valid_mask]
                y_valid = y[valid_mask]
                
                # 使用statsmodels进行回归分析
                X_with_const = sm.add_constant(X_valid)
                model = sm.OLS(y_valid, X_with_const)
                results = model.fit()
                
                # 安全地访问系数，使用iloc而不是位置索引
                if len(results.params) > 1:
                    beta = results.params.iloc[1]  # 特征系数
                    beta_se = results.bse.iloc[1]  # 标准误
                    p_value = results.pvalues.iloc[1]
                else:
                    # 如果只有常数项，跳过这个特征
                    continue
                
                # 计算95%置信区间
                ci_lower = beta - 1.96 * beta_se
                ci_upper = beta + 1.96 * beta_se
                
                regression_results.append({
                    'feature': feature,
                    'beta_per_sd': beta,
                    'beta_se': beta_se,
                    'ci_lower': ci_lower,
                    'ci_upper': ci_upper,
                    'p_value': p_value,
                    'r_squared': results.rsquared,
                    'n_samples': len(y_valid)
                })
                
            elif target_type == 'binary':
                # 逻辑回归：OR per-SD ± 95%CI
                X = features_scaled_df[feature].values.reshape(-1, 1)
                y = df[target_var].values
                
                # 移除缺失值
                valid_mask = ~(np.isnan(X).any(axis=1) | np.isnan(y))
                if np.sum(valid_mask) < 10:
                    continue
                
                X_valid = X[valid_mask]
                y_valid = y[valid_mask]
                
                # 使用statsmodels进行逻辑回归
                X_with_const = sm.add_constant(X_valid)
                model = sm.Logit(y_valid, X_with_const)
                results = model.fit()
                
                # 安全地访问系数，使用iloc而不是位置索引
                if len(results.params) > 1:
                    beta = results.params.iloc[1]  # 特征系数
                    beta_se = results.bse.iloc[1]  # 标准误
                    p_value = results.pvalues.iloc[1]
                else:
                    # 如果只有常数项，跳过这个特征
                    continue
                
                # 计算OR和95%置信区间
                or_value = np.exp(beta)
                or_ci_lower = np.exp(beta - 1.96 * beta_se)
                or_ci_upper = np.exp(beta + 1.96 * beta_se)
                
                regression_results.append({
                    'feature': feature,
                    'or_per_sd': or_value,
                    'or_ci_lower': or_ci_lower,
                    'or_ci_upper': or_ci_upper,
                    'beta': beta,
                    'beta_se': beta_se,
                    'p_value': p_value,
                    'n_samples': len(y_valid)
                })
                
        except Exception as e:
            logger.warning(f"Regression failed for {feature}: {e}")
            continue
    
    # 确保返回的DataFrame包含必要的列
    if regression_results:
        result_df = pd.DataFrame(regression_results)
        # 确保feature列存在
        if 'feature' not in result_df.columns:
            # 确保feature_cols是列表类型
            if hasattr(feature_cols, 'tolist'):
                feature_cols_list = feature_cols.tolist()
            else:
                feature_cols_list = list(feature_cols)
            result_df['feature'] = feature_cols_list[:len(regression_results)]
        return result_df
    else:
        # 返回空的DataFrame但包含必要的列
        return pd.DataFrame(columns=['feature'])


def combine_and_correct_results(correlation_results: pd.DataFrame, regression_results: pd.DataFrame, 
                              feature_cols: list, fdr_alpha: float) -> pd.DataFrame:
    """合并相关性 and 回归结果，并应用FDR校正"""
    # 确保feature_cols是列表类型
    if hasattr(feature_cols, 'tolist'):
        feature_cols_list = feature_cols.tolist()
    else:
        feature_cols_list = list(feature_cols)
    
    # 创建所有特征的结果框架
    all_features = pd.DataFrame({'feature': feature_cols_list})
    
    # 合并相关性结果
    if not correlation_results.empty:
        # 重命名相关性结果的列以避免冲突
        correlation_renamed = correlation_results.copy()
        correlation_renamed = correlation_renamed.rename(columns={
            'p_value': 'corr_p_value',
            'n_samples': 'corr_n_samples'
        })
        all_features = all_features.merge(correlation_renamed, on='feature', how='left')
        
        # 将相关性p值作为主要p值
        all_features['p_value'] = all_features['corr_p_value']
        all_features['n_samples'] = all_features['corr_n_samples']
    
    # 合并回归结果
    if not regression_results.empty:
        # 重命名回归结果的列以避免冲突
        regression_renamed = regression_results.copy()
        regression_renamed = regression_renamed.rename(columns={
            'p_value': 'reg_p_value',
            'n_samples': 'reg_n_samples'
        })
        all_features = all_features.merge(regression_renamed, on='feature', how='left')
        
        # 如果相关性p值缺失，使用回归p值
        if 'p_value' not in all_features.columns:
            all_features['p_value'] = all_features['reg_p_value']
        else:
            # 优先使用相关性p值，缺失时使用回归p值
            all_features['p_value'] = all_features['p_value'].fillna(all_features['reg_p_value'])
    
    # 确保必要的列存在并填充缺失值
    if 'correlation' not in all_features.columns:
        all_features['correlation'] = 0.0
    
    if 'p_value' not in all_features.columns:
        all_features['p_value'] = 1.0
    
    if 'n_samples' not in all_features.columns:
        all_features['n_samples'] = 0
    
    # 填充缺失值
    all_features = all_features.fillna({
        'correlation': 0.0,
        'p_value': 1.0,
        'n_samples': 0
    })
    
    # 应用FDR校正
    if not all_features.empty:
        # 移除无效的p值
        valid_p_mask = (all_features['p_value'] >= 0) & (all_features['p_value'] <= 1)
        
        if np.sum(valid_p_mask) > 0:
            try:
                valid_p_values = all_features.loc[valid_p_mask, 'p_value']
                
                # 检查是否有足够的变化进行FDR校正
                if len(valid_p_values) > 1 and valid_p_values.std() > 0:
                    _, q_values, _, _ = multipletests(valid_p_values, method='fdr_bh', alpha=fdr_alpha)
                    
                    # 创建q_value列
                    all_features['q_value'] = 1.0  # 默认值
                    all_features.loc[valid_p_mask, 'q_value'] = q_values
                    
                    logger.info(f"FDR correction applied to {len(valid_p_values)} p-values")
                    logger.info(f"P-value range: {valid_p_values.min():.6f} - {valid_p_values.max():.6f}")
                    logger.info(f"Q-value range: {q_values.min():.6f} - {q_values.max():.6f}")
                else:
                    # 如果p值没有变化，直接使用原始p值
                    all_features['q_value'] = all_features['p_value']
                    logger.info("P-values have no variation, using original values for FDR")
                    
            except Exception as e:
                logger.warning(f"FDR correction failed: {e}, using original p-values")
                all_features['q_value'] = all_features['p_value']
        else:
            # 如果没有有效的p值，设置默认q值
            all_features['q_value'] = 1.0
    else:
        # 如果DataFrame为空，创建必要的列
        all_features['q_value'] = pd.Series(dtype=float)
    
    # 计算综合显著性
    all_features['significant'] = all_features['q_value'] < fdr_alpha
    
    # 按相关性绝对值排序
    all_features['abs_correlation'] = all_features['correlation'].abs()
    all_features = all_features.sort_values('abs_correlation', ascending=False)
    
    # 添加调试信息
    logger.info(f"Total features: {len(all_features)}")
    logger.info(f"Features with valid p-values: {valid_p_mask.sum()}")
    logger.info(f"Features with q < {fdr_alpha}: {(all_features['q_value'] < fdr_alpha).sum()}")
    logger.info(f"Final p-value range: {all_features['p_value'].min():.6f} - {all_features['p_value'].max():.6f}")
    logger.info(f"Final q-value range: {all_features['q_value'].min():.6f} - {all_features['q_value'].max():.6f}")
    
    return all_features


def save_associations_results(results: dict, output_dir: str):
    """保存关联性分析结果"""
    data_dir = os.path.join(output_dir, "data")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 保存每个目标变量的结果
    for target_var, result in results.items():
        if not result['all_results'].empty:
            result_file = os.path.join(data_dir, f"associations_{target_var}.csv")
            result['all_results'].to_csv(result_file, index=False)
    
    # 保存汇总结果
    summary_data = []
    for target_var, result in results.items():
        summary_data.append({
            'target_variable': target_var,
            'target_type': result['target_type'],
            'total_features': result['total_features'],
            'significant_features': result['significant_count'],
            'significant_ratio': result['significant_count'] / result['total_features'] if result['total_features'] > 0 else 0
        })
    
    summary_df = pd.DataFrame(summary_data)
    summary_path = os.path.join(data_dir, "associations_summary.csv")
    summary_df.to_csv(summary_path, index=False)
    
    # 创建结果索引文件
    results_index = {
        "experiment_type": "correlation",
        "experiment_purpose": "Associations Analysis (Correlation + Regression Effect Sizes)",
        "files": {
            "summary": "data/associations_summary.csv"
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
        if not result['all_results'].empty:
            results_index["files"][f"associations_{target_var}"] = f"data/associations_{target_var}.csv"
            results_index["plots"][f"associations_matrix_{target_var}"] = f"plots/associations_matrix_{target_var}.png"
            results_index["plots"][f"scatter_plots_{target_var}"] = f"plots/scatter_plots_{target_var}.png"
    
    with open(os.path.join(output_dir, "results_index.json"), "w", encoding='utf-8') as f:
        json.dump(results_index, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Associations results saved to {output_dir}")


def plot_associations_matrix(results: dict, output_dir: str):
    """绘制关联性矩阵图"""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    for target_var, result in results.items():
        if result['all_results'].empty:
            continue
        
        plt.figure(figsize=(12, 8))
        
        top_results = result['all_results'].head(20)  # 显示前20个
        features = top_results['feature'].tolist()
        correlations = top_results['correlation'].tolist()
        q_values = top_results['q_value'].tolist()
        
        # 根据q值设置颜色
        colors = ['red' if q < 0.05 else 'lightcoral' for q in q_values]
        
        bars = plt.barh(range(len(features)), correlations, color=colors)
        
        # 添加显著性标记
        for i, (corr, q_val) in enumerate(zip(correlations, q_values)):
            if q_val < 0.05:
                plt.text(corr + (0.01 if corr > 0 else -0.01), i, 
                        '***' if q_val < 0.001 else '**' if q_val < 0.01 else '*', 
                        va='center', ha='left' if corr > 0 else 'right')
        
        plt.yticks(range(len(features)), features)
        plt.xlabel(f'Correlation ({target_var})')
        plt.title(f'Top associations with {target_var} (FDR-corrected)')
        plt.grid(axis='x', alpha=0.3)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"associations_matrix_{target_var}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def plot_scatter_plots(df: pd.DataFrame, results: dict, output_dir: str, top_n: int):
    """绘制散点图"""
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    for target_var, result in results.items():
        if target_var not in df.columns or result['all_results'].empty:
            continue
        
        top_features = result['all_results'].head(min(top_n, 6))['feature'].tolist()
        
        if not top_features:
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
            
            # 获取关联性信息
            feature_result = result['all_results'][result['all_results']['feature'] == feature]
            if feature_result.empty:
                continue
                
            corr = feature_result['correlation'].iloc[0]
            q_val = feature_result['q_value'].iloc[0]
            
            # 绘制散点图
            ax.scatter(df[target_var], df[feature], alpha=0.6, s=20)
            
            # 添加趋势线
            z = np.polyfit(df[target_var], df[feature], 1)
            p = np.poly1d(z)
            ax.plot(df[target_var], p(df[target_var]), "r--", alpha=0.8)
            
            # 设置标签和标题
            ax.set_xlabel(target_var)
            ax.set_ylabel(feature)
            significance = '***' if q_val < 0.001 else '**' if q_val < 0.01 else '*' if q_val < 0.05 else 'ns'
            ax.set_title(f'{feature}\nr={corr:.3f}, q={q_val:.3f} {significance}')
            ax.grid(True, alpha=0.3)
        
        # 隐藏多余的子图
        for i in range(n_features, len(axes)):
            axes[i].set_visible(False)
        
        plt.tight_layout()
        plt.savefig(os.path.join(plots_dir, f"scatter_plots_{target_var}.png"), 
                   dpi=300, bbox_inches='tight')
        plt.close()


def generate_structured_summary(results: dict, df: pd.DataFrame, target_vars: list, 
                              method: str, fdr_alpha: float) -> dict:
    """生成结构化的实验摘要，供前端显示"""
    
    summary_lines = []
    summary_lines.append("=" * 60)
    summary_lines.append("Associations Analysis Experiment Summary")
    summary_lines.append("=" * 60)
    summary_lines.append(f"Correlation method: {method}")
    summary_lines.append(f"FDR correction: α = {fdr_alpha}")
    summary_lines.append(f"Number of recordings: {len(df)}")
    summary_lines.append(f"Number of features: {len([col for col in df.columns if col not in ['recording_id'] + target_vars])}")
    summary_lines.append("")
    
    # 前端摘要数据
    frontend_summary = {
        "total_features": len([col for col in df.columns if col not in ['recording_id'] + target_vars]),
        "total_samples": len(df),
        "correlation_method": method,
        "fdr_alpha": fdr_alpha,
        "target_variables": {},
        "overall_significant_features": 0,
        "total_significant_associations": 0
    }
    
    # 只处理实际被分析的目标变量
    analyzed_targets = list(results.keys())
    logger.info(f"Actually analyzed target variables: {analyzed_targets}")
    
    for target_var in analyzed_targets:
        result = results[target_var]
        summary_lines.append(f"Target variable: {target_var} ({result['target_type']})")
        summary_lines.append(f"  Total features: {result['total_features']}")
        summary_lines.append(f"  Significant associations: {result['significant_count']}")
        
        if result['total_features'] > 0:
            significant_ratio = result['significant_count'] / result['total_features']
            summary_lines.append(f"  Significant ratio: {significant_ratio:.2%}")
        else:
            significant_ratio = 0.0
            summary_lines.append(f"  Significant ratio: N/A")
        
        # 前端数据
        target_data = {
            "type": result['target_type'],
            "total_features": result['total_features'],
            "significant_count": result['significant_count'],
            "significant_ratio": significant_ratio
        }
        
        # 添加top 5最显著关联
        if not result['all_results'].empty:
            top_5 = result['all_results'].head(5)
            top_associations = []
            for _, row in top_5.iterrows():
                corr = row['correlation']
                q_val = row['q_value']
                significance = '***' if q_val < 0.001 else '**' if q_val < 0.01 else '*' if q_val < 0.05 else 'ns'
                top_associations.append({
                    "feature": row['feature'],
                    "correlation": round(corr, 3),
                    "q_value": round(q_val, 3),
                    "significance": significance,
                    "abs_correlation": round(abs(corr), 3)
                })
            target_data["top_associations"] = top_associations
        
        frontend_summary["target_variables"][target_var] = target_data
        
        frontend_summary["overall_significant_features"] += result['significant_count']
        frontend_summary["total_significant_associations"] += result['significant_count']
        
        # 显示前5个最显著关联
        if not result['all_results'].empty:
            summary_lines.append("  Top 5 associations:")
            top_5 = result['all_results'].head(5)
            for _, row in top_5.iterrows():
                corr = row['correlation']
                q_val = row['q_value']
                significance = '***' if q_val < 0.001 else '**' if q_val < 0.01 else '*' if q_val < 0.05 else 'ns'
                summary_lines.append(f"    {row['feature']}: r={corr:.3f}, q={q_val:.3f} {significance}")
        else:
            summary_lines.append("  No significant associations found")
        summary_lines.append("")
    
    # 如果有被跳过的目标变量，在摘要中说明
    skipped_targets = [var for var in target_vars if var not in analyzed_targets]
    if skipped_targets:
        summary_lines.append("Skipped target variables (unsuitable for current correlation method):")
        for var in skipped_targets:
            summary_lines.append(f"  - {var}: Use spearman correlation for categorical variables")
        summary_lines.append("")
    
    summary_lines.append("Result files:")
    summary_lines.append("  - associations_*.csv: Association analysis results")
    summary_lines.append("  - associations_summary.csv: Summary results")
    summary_lines.append("  - associations_matrix_*.png: Association matrix plots")
    summary_lines.append("  - scatter_plots_*.png: Scatter plots")
    summary_lines.append("")
    summary_lines.append("Note: Results include both correlation coefficients and regression effect sizes")
    summary_lines.append("FDR correction applied for multiple testing")
    
    summary_text = "\n".join(summary_lines)
    
    # 返回结构化数据
    return {
        "status": "success",
        "summary_text": summary_text,
        "frontend_summary": frontend_summary
    }
