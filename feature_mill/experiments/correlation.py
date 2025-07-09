"""
Correlation analysis experiment module

This module analyzes the correlation between EEG features and subject metadata (such as age, sex, etc.).
Supports Pearson, Spearman, and Kendall correlation methods.
Now supports recording-level aggregated features.
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import pearsonr, spearmanr
import warnings
warnings.filterwarnings('ignore')

# Set English font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run(df_feat: pd.DataFrame, df_meta: pd.DataFrame, output_dir: str, **kwargs) -> str:
    """
    Run correlation analysis experiment with recording-level features
    
    Args:
        df_feat: Feature matrix dataframe (recording-level aggregated)
        df_meta: Metadata dataframe
        output_dir: Output directory
        **kwargs: Extra arguments
            - target_vars: List of target variables, default ['age', 'sex']
            - method: Correlation method ('pearson', 'spearman', 'kendall'), default 'pearson'
            - min_corr: Minimum correlation threshold, default 0.3
            - top_n: Show top N most correlated features, default 20
            - plot_corr_matrix: Whether to plot correlation matrix, default True
            - plot_scatter: Whether to plot scatter plots, default True
            - save_detailed_results: Whether to save detailed results, default True
    
    Returns:
        str: Experiment summary
    """
    print(f"[correlation] Start correlation analysis experiment")
    print(f"[correlation] Feature matrix shape: {df_feat.shape}")
    print(f"[correlation] Metadata shape: {df_meta.shape}")
    
    # Get parameters
    target_vars = kwargs.get('target_vars', ['age', 'sex'])
    method = kwargs.get('method', 'pearson')
    min_corr = kwargs.get('min_corr', 0.3)
    top_n = kwargs.get('top_n', 20)
    plot_corr_matrix = kwargs.get('plot_corr_matrix', True)
    plot_scatter = kwargs.get('plot_scatter', True)
    save_detailed_results = kwargs.get('save_detailed_results', True)
    
    # Merge features and metadata
    df_combined = merge_features_and_metadata(df_feat, df_meta)
    
    # Data preprocessing
    df_processed = preprocess_data(df_combined, target_vars)
    
    # Perform correlation analysis
    correlation_results = perform_correlation_analysis(
        df_processed, target_vars, method, min_corr, top_n
    )
    
    # Save results
    save_correlation_results(correlation_results, output_dir, save_detailed_results)
    
    # Visualization
    if plot_corr_matrix:
        plot_correlation_matrix(correlation_results, output_dir)
    
    if plot_scatter:
        plot_scatter_plots(df_processed, correlation_results, output_dir, top_n)
    
    # Generate summary
    summary = generate_summary(correlation_results, df_processed, target_vars, method)
    
    print(f"[correlation] Correlation analysis completed, results saved to: {output_dir}")
    return summary


def merge_features_and_metadata(df_feat: pd.DataFrame, df_meta: pd.DataFrame) -> pd.DataFrame:
    """Merge feature matrix and metadata"""
    if 'recording_id' not in df_feat.columns:
        raise ValueError("Feature matrix missing 'recording_id' column")
    
    df_feat['recording_id'] = df_feat['recording_id'].astype(int)
    df_meta['recording_id'] = df_meta['recording_id'].astype(int)
    
    df_combined = pd.merge(df_feat, df_meta, on='recording_id', how='inner')
    
    print(f"[correlation] Merged data shape: {df_combined.shape}")
    print(f"[correlation] Number of recordings: {len(df_combined)}")
    return df_combined


def preprocess_data(df: pd.DataFrame, target_vars: list) -> pd.DataFrame:
    """Data preprocessing for recording-level analysis"""
    df_processed = df.copy()
    
    print(f"[correlation] Initial data shape: {df_processed.shape}")
    print(f"[correlation] Initial columns: {list(df_processed.columns)}")
    
    # Process target variables
    for var in target_vars:
        if var in df_processed.columns:
            if var == 'sex':
                df_processed[var] = df_processed[var].map({'M': 1, 'F': 0, 'Male': 1, 'Female': 0})
            elif var in ['age', 'age_days']:
                df_processed[var] = pd.to_numeric(df_processed[var], errors='coerce')
    
    # 检查缺失值情况
    missing_ratio = df_processed.isnull().sum() / len(df_processed)
    print(f"[correlation] Missing value statistics:")
    print(f"[correlation] - Columns with >50% missing: {len(missing_ratio[missing_ratio > 0.5])}")
    print(f"[correlation] - Columns with >80% missing: {len(missing_ratio[missing_ratio > 0.8])}")
    print(f"[correlation] - Columns with >90% missing: {len(missing_ratio[missing_ratio > 0.9])}")
    
    # 降低缺失值阈值
    columns_to_drop_missing = missing_ratio[missing_ratio > 0.9].index
    df_processed = df_processed.drop(columns=columns_to_drop_missing)
    print(f"[correlation] Columns with >90% missing values: {len(columns_to_drop_missing)}")
    print(f"[correlation] Shape after removing high-missing columns: {df_processed.shape}")
    
    # Remove non-numeric columns
    numeric_columns = df_processed.select_dtypes(include=[np.number]).columns
    non_numeric_columns = df_processed.select_dtypes(exclude=[np.number]).columns
    
    print(f"[correlation] Numeric columns: {len(numeric_columns)}")
    print(f"[correlation] Non-numeric columns: {len(numeric_columns)}")
    print(f"[correlation] Non-numeric column names: {list(non_numeric_columns)}")
    
    # 保留recording_id和目标变量
    columns_to_keep = ['recording_id'] + target_vars
    columns_to_remove = [col for col in non_numeric_columns if col not in columns_to_keep]
    
    df_processed = df_processed.drop(columns=columns_to_remove)
    print(f"[correlation] Non-numeric columns removed: {len(columns_to_remove)}")
    print(f"[correlation] Shape after removing non-numeric columns: {df_processed.shape}")
    print(f"[correlation] Final columns: {list(df_processed.columns)}")
    
    return df_processed


def perform_correlation_analysis(df: pd.DataFrame, target_vars: list, method: str, 
                                min_corr: float, top_n: int) -> dict:
    """Perform correlation analysis on recording-level features"""
    results = {}
    
    # 添加调试信息
    print(f"[correlation] All columns: {list(df.columns)}")
    print(f"[correlation] Target variables: {target_vars}")
    
    # Get feature columns (exclude recording_id and target_vars)
    feature_cols = [col for col in df.columns if col not in ['recording_id'] + target_vars]
    
    print(f"[correlation] Feature columns: {len(feature_cols)}")
    print(f"[correlation] Sample feature names: {feature_cols[:10] if len(feature_cols) > 10 else feature_cols}")
    
    # 检查是否有特征列
    if len(feature_cols) == 0:
        print("[correlation] WARNING: No feature columns available for analysis!")
        print("[correlation] This means all columns are either recording_id or target variables")
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
            print(f"[correlation] Warning: Target variable {target_var} not in data")
            continue
        
        print(f"[correlation] Analyzing target variable: {target_var}")
        print(f"[correlation] Number of features to analyze: {len(feature_cols)}")
        
        # 添加调试：查看前几个特征的数据
        print(f"[correlation] Sample feature data:")
        for feature in feature_cols[:3]:
            print(f"  {feature}: mean={df[feature].mean():.4f}, std={df[feature].std():.4f}, range=[{df[feature].min():.4f}, {df[feature].max():.4f}]")
        
        correlations = []
        p_values = []
        
        for feature in feature_cols:
            # Calculate correlation
            if method == 'pearson':
                corr, p_val = pearsonr(df[feature], df[target_var])
            elif method == 'spearman':
                corr, p_val = spearmanr(df[feature], df[target_var])
            else:
                corr, p_val = stats.kendalltau(df[feature], df[target_var])
            
            correlations.append(corr)
            p_values.append(p_val)
        
        # Create result DataFrame
        result_df = pd.DataFrame({
            'feature': feature_cols,
            'correlation': correlations,
            'p_value': p_values,
            'abs_correlation': np.abs(correlations)
        })
        
        # Sort and filter
        result_df = result_df.sort_values('abs_correlation', ascending=False)
        result_df = result_df[result_df['abs_correlation'] >= min_corr]
        
        # Add significance mark
        result_df['significant'] = result_df['p_value'] < 0.05
        result_df['significance_level'] = result_df['p_value'].apply(get_significance_level)
        
        results[target_var] = {
            'all_results': result_df,
            'top_results': result_df.head(top_n),
            'significant_count': len(result_df[result_df['significant']]),
            'total_features': len(feature_cols)
        }
        
        print(f"[correlation] Found {len(result_df)} features with correlation >= {min_corr}")
        print(f"[correlation] Significant features: {results[target_var]['significant_count']}")
        
        # 添加调试：显示前几个相关性结果
        if len(result_df) > 0:
            print(f"[correlation] Top correlations:")
            for _, row in result_df.head(5).iterrows():
                print(f"  {row['feature']}: corr={row['correlation']:.4f}, p={row['p_value']:.4f}")
    
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
