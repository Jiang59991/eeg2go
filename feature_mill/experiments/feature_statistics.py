"""
Feature Statistics Analysis Experiment Module

This module performs comprehensive statistical analysis of EEG features including:
- Basic statistics (mean, std, min, max, skewness, kurtosis)
- Distribution analysis
- Feature quality assessment
- Outlier detection
- Feature importance ranking
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
warnings.filterwarnings('ignore')

# Set English font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run(df_feat: pd.DataFrame, df_meta: pd.DataFrame, output_dir: str, **kwargs) -> str:
    """
    Run feature statistics analysis experiment
    
    Args:
        df_feat: Feature matrix dataframe
        df_meta: Metadata dataframe (optional for this experiment)
        output_dir: Output directory
        **kwargs: Extra arguments
            - outlier_method: Outlier detection method ('iqr', 'zscore', 'isolation'), default 'iqr'
            - outlier_threshold: Outlier threshold, default 1.5
            - plot_distributions: Whether to plot feature distributions, default True
            - plot_correlation_heatmap: Whether to plot feature correlation heatmap, default True
            - plot_outliers: Whether to plot outlier analysis, default True
            - top_n_features: Number of top features to analyze in detail, default 20
    
    Returns:
        str: Experiment summary
    """
    print(f"[feature_statistics] Start feature statistics analysis experiment")
    print(f"[feature_statistics] Feature matrix shape: {df_feat.shape}")
    
    # Get parameters
    outlier_method = kwargs.get('outlier_method', 'iqr')
    outlier_threshold = kwargs.get('outlier_threshold', 1.5)
    plot_distributions = kwargs.get('plot_distributions', True)
    plot_correlation_heatmap = kwargs.get('plot_correlation_heatmap', True)
    plot_outliers = kwargs.get('plot_outliers', True)
    top_n_features = kwargs.get('top_n_features', 20)
    
    # Data preprocessing
    df_processed = preprocess_features(df_feat)
    
    # Basic statistics
    basic_stats = calculate_basic_statistics(df_processed)
    
    # Distribution analysis
    distribution_analysis = analyze_distributions(df_processed)
    
    # Outlier analysis
    outlier_analysis = detect_outliers(df_processed, outlier_method, outlier_threshold)
    
    # Feature importance ranking
    feature_importance = rank_features(df_processed)
    
    # Save results
    save_statistics_results(basic_stats, distribution_analysis, outlier_analysis, 
                           feature_importance, output_dir)
    
    # Visualizations
    if plot_distributions:
        plot_feature_distributions(df_processed, distribution_analysis, output_dir, top_n_features)
    
    if plot_correlation_heatmap:
        plot_feature_correlation_heatmap(df_processed, output_dir)
    
    if plot_outliers:
        plot_outlier_analysis(df_processed, outlier_analysis, output_dir, top_n_features)
    
    # Generate summary
    summary = generate_summary(basic_stats, distribution_analysis, outlier_analysis, 
                              feature_importance, df_processed)
    
    print(f"[feature_statistics] Feature statistics analysis completed, results saved to: {output_dir}")
    return summary


def preprocess_features(df_feat: pd.DataFrame) -> pd.DataFrame:
    """Preprocess feature matrix"""
    df_processed = df_feat.copy()
    
    # Remove recording_id column for analysis
    if 'recording_id' in df_processed.columns:
        df_processed = df_processed.drop(columns=['recording_id'])
    
    # Remove columns with too many missing values
    missing_threshold = 0.5
    missing_ratio = df_processed.isnull().sum() / len(df_processed)
    columns_to_drop = missing_ratio[missing_ratio > missing_threshold].index
    df_processed = df_processed.drop(columns=columns_to_drop)
    
    # Remove non-numeric columns
    df_processed = df_processed.select_dtypes(include=[np.number])
    
    # Fill missing values with median
    df_processed = df_processed.fillna(df_processed.median())
    
    print(f"[feature_statistics] Processed data shape: {df_processed.shape}")
    return df_processed


def calculate_basic_statistics(df: pd.DataFrame) -> dict:
    """Calculate basic statistics for all features"""
    print("[feature_statistics] Calculating basic statistics...")
    
    stats_dict = {}
    
    for column in df.columns:
        values = df[column].dropna()
        if len(values) == 0:
            continue
            
        stats_dict[column] = {
            'count': len(values),
            'mean': np.mean(values),
            'std': np.std(values),
            'min': np.min(values),
            'max': np.max(values),
            'median': np.median(values),
            'skewness': skew(values),
            'kurtosis': kurtosis(values),
            'q25': np.percentile(values, 25),
            'q75': np.percentile(values, 75),
            'iqr': np.percentile(values, 75) - np.percentile(values, 25),
            'cv': np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf
        }
    
    return stats_dict


def analyze_distributions(df: pd.DataFrame) -> dict:
    """Analyze feature distributions"""
    print("[feature_statistics] Analyzing distributions...")
    
    distribution_analysis = {}
    
    for column in df.columns:
        values = df[column].dropna()
        if len(values) == 0:
            continue
        
        # Normality test
        try:
            _, p_value = stats.normaltest(values)
            is_normal = p_value > 0.05
        except:
            is_normal = False
            p_value = np.nan
        
        # Distribution type classification
        skewness = skew(values)
        kurt = kurtosis(values)
        
        if abs(skewness) < 0.5 and abs(kurt) < 0.5:
            dist_type = "Normal-like"
        elif abs(skewness) > 1:
            dist_type = "Skewed"
        elif abs(kurt) > 1:
            dist_type = "Heavy-tailed"
        else:
            dist_type = "Other"
        
        distribution_analysis[column] = {
            'is_normal': is_normal,
            'normality_p_value': p_value,
            'distribution_type': dist_type,
            'skewness': skewness,
            'kurtosis': kurt
        }
    
    return distribution_analysis


def detect_outliers(df: pd.DataFrame, method: str, threshold: float) -> dict:
    """Detect outliers in features"""
    print(f"[feature_statistics] Detecting outliers using {method} method...")
    
    outlier_analysis = {}
    
    for column in df.columns:
        values = df[column].dropna()
        if len(values) == 0:
            continue
        
        outliers = []
        
        if method == 'iqr':
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - threshold * iqr
            upper_bound = q3 + threshold * iqr
            outliers = values[(values < lower_bound) | (values > upper_bound)]
        
        elif method == 'zscore':
            z_scores = np.abs(stats.zscore(values))
            outliers = values[z_scores > threshold]
        
        elif method == 'isolation':
            # Simple isolation forest-like approach
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 2 * iqr
            upper_bound = q3 + 2 * iqr
            outliers = values[(values < lower_bound) | (values > upper_bound)]
        
        outlier_analysis[column] = {
            'outlier_count': len(outliers),
            'outlier_percentage': len(outliers) / len(values) * 100,
            'outlier_values': outliers.tolist() if len(outliers) <= 10 else outliers.head(10).tolist()
        }
    
    return outlier_analysis


def rank_features(df: pd.DataFrame) -> pd.DataFrame:
    """Rank features by importance/variability"""
    print("[feature_statistics] Ranking features by importance...")
    
    feature_scores = []
    
    for column in df.columns:
        values = df[column].dropna()
        if len(values) == 0:
            continue
        
        # Calculate various importance metrics
        variance = np.var(values)
        cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
        range_val = np.max(values) - np.min(values)
        
        # Combined score (weighted average)
        score = (0.4 * variance + 0.3 * abs(cv) + 0.3 * range_val)
        
        feature_scores.append({
            'feature': column,
            'variance': variance,
            'coefficient_of_variation': cv,
            'range': range_val,
            'importance_score': score
        })
    
    # Create DataFrame and sort by importance score
    feature_importance_df = pd.DataFrame(feature_scores)
    feature_importance_df = feature_importance_df.sort_values('importance_score', ascending=False)
    
    return feature_importance_df


def save_statistics_results(basic_stats: dict, distribution_analysis: dict, 
                           outlier_analysis: dict, feature_importance: pd.DataFrame, 
                           output_dir: str):
    """Save all analysis results"""
    
    # Save basic statistics
    basic_stats_df = pd.DataFrame(basic_stats).T
    basic_stats_df.to_csv(os.path.join(output_dir, "feature_basic_statistics.csv"))
    
    # Save distribution analysis
    dist_df = pd.DataFrame(distribution_analysis).T
    dist_df.to_csv(os.path.join(output_dir, "feature_distribution_analysis.csv"))
    
    # Save outlier analysis
    outlier_df = pd.DataFrame(outlier_analysis).T
    outlier_df.to_csv(os.path.join(output_dir, "feature_outlier_analysis.csv"))
    
    # Save feature importance
    feature_importance.to_csv(os.path.join(output_dir, "feature_importance_ranking.csv"), index=False)
    
    print(f"[feature_statistics] Results saved to {output_dir}")


def plot_feature_distributions(df: pd.DataFrame, distribution_analysis: dict, 
                              output_dir: str, top_n: int):
    """Plot feature distributions"""
    print("[feature_statistics] Plotting feature distributions...")
    
    # Get top features by importance
    top_features = list(distribution_analysis.keys())[:top_n]
    
    # Create subplots
    n_cols = 4
    n_rows = (len(top_features) + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(15, 3 * n_rows))
    if n_rows == 1:
        axes = axes.reshape(1, -1)
    
    for i, feature in enumerate(top_features):
        row = i // n_cols
        col = i % n_cols
        
        values = df[feature].dropna()
        dist_info = distribution_analysis[feature]
        
        # Plot histogram
        axes[row, col].hist(values, bins=30, alpha=0.7, edgecolor='black')
        axes[row, col].set_title(f'{feature}\n{dist_info["distribution_type"]}')
        axes[row, col].set_xlabel('Value')
        axes[row, col].set_ylabel('Frequency')
        
        # Add statistics text
        stats_text = f'μ={np.mean(values):.3f}\nσ={np.std(values):.3f}\nSkew={dist_info["skewness"]:.3f}'
        axes[row, col].text(0.02, 0.98, stats_text, transform=axes[row, col].transAxes, 
                           verticalalignment='top', fontsize=8,
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    # Hide empty subplots
    for i in range(len(top_features), n_rows * n_cols):
        row = i // n_cols
        col = i % n_cols
        axes[row, col].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_distributions.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_correlation_heatmap(df: pd.DataFrame, output_dir: str):
    """Plot feature correlation heatmap"""
    print("[feature_statistics] Plotting correlation heatmap...")
    
    # Calculate correlation matrix
    corr_matrix = df.corr()
    
    # Plot heatmap
    plt.figure(figsize=(12, 10))
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap='coolwarm', center=0,
                square=True, linewidths=0.5, cbar_kws={"shrink": .8})
    plt.title('Feature Correlation Matrix')
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "feature_correlation_heatmap.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_outlier_analysis(df: pd.DataFrame, outlier_analysis: dict, 
                         output_dir: str, top_n: int):
    """Plot outlier analysis"""
    print("[feature_statistics] Plotting outlier analysis...")
    
    # Get top features by outlier percentage
    outlier_percentages = [(feature, outlier_analysis[feature]['outlier_percentage']) 
                          for feature in outlier_analysis.keys()]
    outlier_percentages.sort(key=lambda x: x[1], reverse=True)
    top_features = [feature for feature, _ in outlier_percentages[:top_n]]
    
    # Create box plots
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    
    for i, feature in enumerate(top_features[:4]):
        values = df[feature].dropna()
        outlier_info = outlier_analysis[feature]
        
        # Box plot
        axes[i].boxplot(values)
        axes[i].set_title(f'{feature}\nOutliers: {outlier_info["outlier_count"]} ({outlier_info["outlier_percentage"]:.1f}%)')
        axes[i].set_ylabel('Value')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "outlier_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary(basic_stats: dict, distribution_analysis: dict, 
                    outlier_analysis: dict, feature_importance: pd.DataFrame, 
                    df: pd.DataFrame) -> str:
    """Generate experiment summary"""
    
    total_features = len(basic_stats)
    normal_features = sum(1 for info in distribution_analysis.values() if info['is_normal'])
    high_outlier_features = sum(1 for info in outlier_analysis.values() 
                               if info['outlier_percentage'] > 10)
    
    top_features = feature_importance.head(10)['feature'].tolist()
    
    summary = f"""
Feature Statistics Analysis Summary
==================================

Dataset Overview:
- Total features analyzed: {total_features}
- Total samples: {len(df)}

Distribution Analysis:
- Normal-like distributions: {normal_features} ({normal_features/total_features*100:.1f}%)
- Non-normal distributions: {total_features - normal_features} ({(total_features-normal_features)/total_features*100:.1f}%)

Outlier Analysis:
- Features with >10% outliers: {high_outlier_features} ({high_outlier_features/total_features*100:.1f}%)
- Average outlier percentage: {np.mean([info['outlier_percentage'] for info in outlier_analysis.values()]):.1f}%

Top 10 Most Important Features:
{chr(10).join([f"{i+1}. {feature}" for i, feature in enumerate(top_features)])}

Key Findings:
- Feature variability analysis completed
- Distribution characteristics identified
- Outlier patterns analyzed
- Feature importance ranking generated

Results saved to output directory.
"""
    
    return summary 