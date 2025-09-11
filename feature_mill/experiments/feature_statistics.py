import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
from scipy.stats import skew, kurtosis
import warnings
from logging_config import logger
import json
from datetime import datetime
warnings.filterwarnings('ignore')

try:
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Liberation Sans']
    plt.rcParams['axes.unicode_minus'] = False
except Exception as e:
    logger.warning(f"Font setting failed: {e}")

def run(
    df_feat: pd.DataFrame,
    df_meta: pd.DataFrame,
    output_dir: str,
    **kwargs
) -> dict:
    """
    Run the feature profiling and quality assessment experiment.

    Args:
        df_feat (pd.DataFrame): Feature matrix dataframe.
        df_meta (pd.DataFrame): Metadata dataframe (optional).
        output_dir (str): Output directory.
        **kwargs: Extra arguments for configuration.

    Returns:
        dict: Experiment result with structured data for frontend display.
    """
    logger.info(f"Start feature profiling and quality assessment experiment")
    logger.info(f"Feature matrix shape: {df_feat.shape}")

    outlier_method = kwargs.get('outlier_method', 'iqr')
    outlier_threshold = float(kwargs.get('outlier_threshold', 1.5))
    top_n_features = int(kwargs.get('top_n_features', 20))
    generate_plots = kwargs.get('generate_plots', True)
    quality_thresholds = kwargs.get('quality_thresholds', {
        'missing': 0.9,
        'zero_var': 0.0,
        'extreme': 0.1
    })

    df_processed = preprocess_features(df_feat)
    quality_metrics = assess_feature_quality(df_processed, quality_thresholds)
    basic_stats = calculate_basic_statistics(df_processed)
    distribution_analysis = analyze_distributions(df_processed)
    outlier_analysis = detect_outliers(df_processed, outlier_method, outlier_threshold)
    feature_variability = rank_features_by_variability(df_processed)
    data_health_summary = generate_data_health_summary(quality_metrics, df_processed)

    save_quality_assessment_results(
        quality_metrics, basic_stats, distribution_analysis,
        outlier_analysis, feature_variability, data_health_summary, output_dir
    )

    if generate_plots:
        try:
            logger.info("Starting visualization generation...")
            plot_quality_metrics(quality_metrics, output_dir)
            plot_feature_distributions(df_processed, distribution_analysis, output_dir, top_n_features)
            plot_outlier_analysis(df_processed, outlier_analysis, output_dir, top_n_features)
            logger.info("All visualizations completed successfully")
        except Exception as e:
            logger.error(f"Error during visualization generation: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")

    summary_data = generate_structured_summary(
        quality_metrics, basic_stats, distribution_analysis,
        outlier_analysis, feature_variability, data_health_summary, df_processed
    )

    logger.info(f"Feature profiling and quality assessment completed, results saved to: {output_dir}")
    return summary_data

def preprocess_features(df_feat: pd.DataFrame) -> pd.DataFrame:
    """
    Preprocess the feature matrix for quality assessment.

    Args:
        df_feat (pd.DataFrame): Feature matrix dataframe.

    Returns:
        pd.DataFrame: Processed feature dataframe (numeric columns only).
    """
    df_processed = df_feat.copy()
    if 'recording_id' in df_processed.columns:
        df_processed = df_processed.drop(columns=['recording_id'])
    df_processed = df_processed.select_dtypes(include=[np.number])
    logger.info(f"Processed data shape: {df_processed.shape}")
    return df_processed

def assess_feature_quality(df: pd.DataFrame, thresholds: dict) -> dict:
    """
    Assess feature quality and data health.

    Args:
        df (pd.DataFrame): Feature dataframe.
        thresholds (dict): Quality thresholds.

    Returns:
        dict: Quality metrics for each feature.
    """
    logger.info("Assessing feature quality and data health...")
    quality_metrics = {}
    for column in df.columns:
        values = df[column].dropna()
        total_count = len(df[column])
        if total_count == 0:
            continue
        missing_count = df[column].isnull().sum()
        missing_rate = missing_count / total_count
        if len(values) > 0:
            variance = np.var(values)
            zero_variance = variance == 0
            coefficient_variation = np.std(values) / np.mean(values) if np.mean(values) != 0 else np.inf
        else:
            variance = np.nan
            zero_variance = True
            coefficient_variation = np.nan
        if len(values) > 0:
            q1, q3 = np.percentile(values, [25, 75])
            iqr = q3 - q1
            lower_bound = q1 - 1.5 * iqr
            upper_bound = q3 + 1.5 * iqr
            extreme_count = len(values[(values < lower_bound) | (values > upper_bound)])
            extreme_rate = extreme_count / len(values)
        else:
            extreme_count = 0
            extreme_rate = 0
        high_missing = missing_rate > thresholds['missing']
        has_zero_variance = zero_variance
        high_extreme = extreme_rate > thresholds['extreme']
        quality_score = 100
        if high_missing:
            quality_score -= 30
        if has_zero_variance:
            quality_score -= 25
        if high_extreme:
            quality_score -= 20
        if missing_rate > 0.5:
            quality_score -= 15
        quality_score = max(0, quality_score)
        quality_metrics[column] = {
            'total_count': total_count,
            'missing_count': missing_count,
            'missing_rate': missing_rate,
            'zero_variance': zero_variance,
            'variance': variance,
            'coefficient_variation': coefficient_variation,
            'extreme_count': extreme_count,
            'extreme_rate': extreme_rate,
            'quality_flags': {
                'high_missing': high_missing,
                'zero_variance': has_zero_variance,
                'high_extreme': high_extreme
            },
            'quality_score': quality_score,
            'quality_grade': get_quality_grade(quality_score)
        }
    return quality_metrics

def get_quality_grade(score: float) -> str:
    """
    Convert quality score to letter grade.

    Args:
        score (float): Quality score.

    Returns:
        str: Letter grade.
    """
    if score >= 90:
        return 'A'
    elif score >= 80:
        return 'B'
    elif score >= 70:
        return 'C'
    elif score >= 60:
        return 'D'
    else:
        return 'F'

def calculate_basic_statistics(df: pd.DataFrame) -> dict:
    """
    Calculate basic statistics for all features.

    Args:
        df (pd.DataFrame): Feature dataframe.

    Returns:
        dict: Basic statistics for each feature.
    """
    logger.info("Calculating basic statistics...")
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
    """
    Analyze feature distributions for quality assessment.

    Args:
        df (pd.DataFrame): Feature dataframe.

    Returns:
        dict: Distribution analysis for each feature.
    """
    logger.info("Analyzing distributions...")
    distribution_analysis = {}
    for column in df.columns:
        values = df[column].dropna()
        if len(values) == 0:
            continue
        try:
            _, p_value = stats.normaltest(values)
            is_normal = p_value > 0.05
        except:
            is_normal = False
            p_value = np.nan
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
    """
    Detect outliers in features for data integrity assessment.

    Args:
        df (pd.DataFrame): Feature dataframe.
        method (str): Outlier detection method.
        threshold (float): Outlier threshold.

    Returns:
        dict: Outlier analysis for each feature.
    """
    logger.info(f"Detecting outliers using {method} method...")
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

def rank_features_by_variability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Rank features by variability for quality prioritization.

    Args:
        df (pd.DataFrame): Feature dataframe.

    Returns:
        pd.DataFrame: DataFrame of features ranked by variability score.
    """
    logger.info("Ranking features by variability...")
    feature_scores = []
    for column in df.columns:
        values = df[column].dropna()
        if len(values) == 0:
            continue
        variance = np.var(values)
        cv = np.std(values) / np.mean(values) if np.mean(values) != 0 else 0
        range_val = np.max(values) - np.min(values)
        score = (0.4 * variance + 0.3 * abs(cv) + 0.3 * range_val)
        feature_scores.append({
            'feature': column,
            'variance': variance,
            'coefficient_of_variation': cv,
            'range': range_val,
            'variability_score': score
        })
    feature_variability_df = pd.DataFrame(feature_scores)
    feature_variability_df = feature_variability_df.sort_values('variability_score', ascending=False)
    return feature_variability_df

def generate_data_health_summary(quality_metrics: dict, df: pd.DataFrame) -> dict:
    """
    Generate comprehensive data health summary.

    Args:
        quality_metrics (dict): Feature quality metrics.
        df (pd.DataFrame): Feature dataframe.

    Returns:
        dict: Data health summary.
    """
    logger.info("Generating data health summary...")
    total_features = len(quality_metrics)
    if total_features == 0:
        return {}
    quality_scores = [metrics['quality_score'] for metrics in quality_metrics.values()]
    quality_grades = [metrics['quality_grade'] for metrics in quality_metrics.values()]
    grade_counts = {}
    for grade in ['A', 'B', 'C', 'D', 'F']:
        grade_counts[grade] = quality_grades.count(grade)
    high_missing_count = sum(1 for metrics in quality_metrics.values()
                            if metrics['quality_flags']['high_missing'])
    zero_variance_count = sum(1 for metrics in quality_metrics.values()
                             if metrics['quality_flags']['zero_variance'])
    high_extreme_count = sum(1 for metrics in quality_metrics.values()
                            if metrics['quality_flags']['high_extreme'])
    missing_rates = [metrics['missing_rate'] for metrics in quality_metrics.values()]
    avg_missing_rate = np.mean(missing_rates) if missing_rates else 0
    overall_health_score = np.mean(quality_scores) if quality_scores else 0
    overall_health_grade = get_quality_grade(overall_health_score)
    health_summary = {
        'total_features': total_features,
        'overall_health_score': overall_health_score,
        'overall_health_grade': overall_health_grade,
        'quality_distribution': {
            'grade_counts': grade_counts,
            'avg_quality_score': np.mean(quality_scores) if quality_scores else 0,
            'min_quality_score': np.min(quality_scores) if quality_scores else 0,
            'max_quality_score': np.max(quality_scores) if quality_scores else 0
        },
        'data_quality_issues': {
            'high_missing_features': high_missing_count,
            'zero_variance_features': zero_variance_count,
            'high_extreme_features': high_extreme_count,
            'problematic_features_ratio': (high_missing_count + zero_variance_count + high_extreme_count) / total_features
        },
        'missing_data_summary': {
            'average_missing_rate': avg_missing_rate,
            'features_with_missing': sum(1 for metrics in quality_metrics.values() if metrics['missing_rate'] > 0),
            'severely_missing_features': high_missing_count
        },
        'recommendations': generate_quality_recommendations(quality_metrics, overall_health_score)
    }
    return health_summary

def generate_quality_recommendations(quality_metrics: dict, overall_score: float) -> list:
    """
    Generate quality improvement recommendations.

    Args:
        quality_metrics (dict): Feature quality metrics.
        overall_score (float): Overall health score.

    Returns:
        list: List of recommendations.
    """
    recommendations = []
    if overall_score < 70:
        recommendations.append("Overall data quality is poor. Consider data collection improvements.")
    high_missing_features = [name for name, metrics in quality_metrics.items()
                            if metrics['quality_flags']['high_missing']]
    if high_missing_features:
        recommendations.append(f"Remove or impute {len(high_missing_features)} features with >90% missing values")
    zero_var_features = [name for name, metrics in quality_metrics.items()
                        if metrics['quality_flags']['zero_variance']]
    if zero_var_features:
        recommendations.append(f"Remove {len(zero_var_features)} features with zero variance")
    if overall_score >= 80:
        recommendations.append("Data quality is good. Minor improvements may be needed.")
    return recommendations

def save_quality_assessment_results(
    quality_metrics: dict,
    basic_stats: dict,
    distribution_analysis: dict,
    outlier_analysis: dict,
    feature_variability: pd.DataFrame,
    data_health_summary: dict,
    output_dir: str
) -> None:
    """
    Save all quality assessment results with organized file structure.

    Args:
        quality_metrics (dict): Feature quality metrics.
        basic_stats (dict): Basic statistics.
        distribution_analysis (dict): Distribution analysis.
        outlier_analysis (dict): Outlier analysis.
        feature_variability (pd.DataFrame): Feature variability ranking.
        data_health_summary (dict): Data health summary.
        output_dir (str): Output directory.
    """
    data_dir = os.path.join(output_dir, "data")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    quality_metrics_df = pd.DataFrame(quality_metrics).T
    quality_metrics_df.to_csv(os.path.join(data_dir, "feature_quality_metrics.csv"))
    health_summary_df = pd.DataFrame([data_health_summary])
    health_summary_df.to_csv(os.path.join(data_dir, "data_health_summary.csv"), index=False)
    basic_stats_df = pd.DataFrame(basic_stats).T
    basic_stats_df.to_csv(os.path.join(data_dir, "feature_basic_statistics.csv"))
    dist_df = pd.DataFrame(distribution_analysis).T
    dist_df.to_csv(os.path.join(data_dir, "feature_distribution_analysis.csv"))
    outlier_df = pd.DataFrame(outlier_analysis).T
    outlier_df.to_csv(os.path.join(data_dir, "feature_outlier_analysis.csv"))
    feature_variability.to_csv(os.path.join(data_dir, "feature_variability_ranking.csv"), index=False)
    results_index = {
        "experiment_type": "feature_statistics",
        "experiment_purpose": "Feature Profiling / Quality Assessment",
        "files": {
            "quality_metrics": "data/feature_quality_metrics.csv",
            "data_health_summary": "data/data_health_summary.csv",
            "basic_statistics": "data/feature_basic_statistics.csv",
            "distribution_analysis": "data/feature_distribution_analysis.csv",
            "outlier_analysis": "data/feature_outlier_analysis.csv",
            "feature_variability": "data/feature_variability_ranking.csv"
        },
        "plots": {
            "quality_metrics": "plots/quality_metrics.png",
            "distributions": "plots/feature_distributions.png",
            "outlier_analysis": "plots/outlier_analysis.png"
        },
        "summary": {
            "total_features": len(quality_metrics),
            "overall_health_score": data_health_summary.get('overall_health_score', 0),
            "overall_health_grade": data_health_summary.get('overall_health_grade', 'N/A'),
            "generated_at": datetime.now().isoformat()
        }
    }
    with open(os.path.join(output_dir, "results_index.json"), "w", encoding='utf-8') as f:
        json.dump(results_index, f, indent=2, ensure_ascii=False)
    logger.info(f"Quality assessment results saved to {output_dir} with organized structure")

def plot_quality_metrics(quality_metrics: dict, output_dir: str) -> None:
    """
    Plot quality metrics overview.

    Args:
        quality_metrics (dict): Feature quality metrics.
        output_dir (str): Output directory.
    """
    try:
        logger.info("Plotting quality metrics...")
        features = list(quality_metrics.keys())
        quality_scores = [metrics['quality_score'] for metrics in quality_metrics.values()]
        missing_rates = [metrics['missing_rate'] for metrics in quality_metrics.values()]
        fig, axes = plt.subplots(2, 2, figsize=(15, 12))
        axes[0, 0].hist(quality_scores, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 0].set_title('Feature Quality Score Distribution')
        axes[0, 0].set_xlabel('Quality Score')
        axes[0, 0].set_ylabel('Number of Features')
        axes[0, 0].axvline(np.mean(quality_scores), color='red', linestyle='--',
                           label=f'Mean: {np.mean(quality_scores):.1f}')
        axes[0, 0].legend()
        axes[0, 1].hist(missing_rates, bins=20, alpha=0.7, edgecolor='black')
        axes[0, 1].set_title('Feature Missing Rate Distribution')
        axes[0, 1].set_xlabel('Missing Rate')
        axes[0, 1].set_ylabel('Number of Features')
        axes[0, 1].axvline(np.mean(missing_rates), color='red', linestyle='--',
                           label=f'Mean: {np.mean(missing_rates):.1%}')
        axes[0, 1].legend()
        axes[1, 0].scatter(missing_rates, quality_scores, alpha=0.6)
        axes[1, 0].set_title('Quality Score vs Missing Rate')
        axes[1, 0].set_xlabel('Missing Rate')
        axes[1, 0].set_ylabel('Quality Score')
        grade_counts = {}
        for metrics in quality_metrics.values():
            grade = metrics['quality_grade']
            grade_counts[grade] = grade_counts.get(grade, 0) + 1
        grades = list(grade_counts.keys())
        counts = list(grade_counts.values())
        axes[1, 1].bar(grades, counts, alpha=0.7)
        axes[1, 1].set_title('Feature Quality Grade Distribution')
        axes[1, 1].set_xlabel('Quality Grade')
        axes[1, 1].set_ylabel('Number of Features')
        plt.tight_layout()
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        output_path = os.path.join(plots_dir, "quality_metrics.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Quality metrics plot saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error in plot_quality_metrics: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        try:
            plt.close('all')
        except:
            pass

def plot_feature_distributions(
    df: pd.DataFrame,
    distribution_analysis: dict,
    output_dir: str,
    top_n: int
) -> None:
    """
    Plot feature distributions.

    Args:
        df (pd.DataFrame): Feature dataframe.
        distribution_analysis (dict): Distribution analysis.
        output_dir (str): Output directory.
        top_n (int): Number of top features to plot.
    """
    try:
        logger.info("Plotting feature distributions...")
        top_features = list(distribution_analysis.keys())[:top_n]
        logger.info(f"Plotting distributions for {len(top_features)} features")
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
            axes[row, col].hist(values, bins=30, alpha=0.7, edgecolor='black')
            axes[row, col].set_title(f'{feature}\n{dist_info["distribution_type"]}')
            axes[row, col].set_xlabel('Value')
            axes[row, col].set_ylabel('Frequency')
            stats_text = f'μ={np.mean(values):.3f}\nσ={np.std(values):.3f}\nSkew={dist_info["skewness"]:.3f}'
            axes[row, col].text(0.02, 0.98, stats_text, transform=axes[row, col].transAxes,
                               verticalalignment='top', fontsize=8,
                               bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
        for i in range(len(top_features), n_rows * n_cols):
            row = i // n_cols
            col = i % n_cols
            axes[row, col].set_visible(False)
        plt.tight_layout()
        plots_dir = os.path.join(output_dir, "plots")
        os.makedirs(plots_dir, exist_ok=True)
        output_path = os.path.join(plots_dir, "feature_distributions.png")
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        plt.close()
        logger.info(f"Feature distributions plot saved to: {output_path}")
    except Exception as e:
        logger.error(f"Error in plot_feature_distributions: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        try:
            plt.close('all')
        except:
            pass

def plot_outlier_analysis(
    df: pd.DataFrame,
    outlier_analysis: dict,
    output_dir: str,
    top_n: int
) -> None:
    """
    Plot outlier analysis.

    Args:
        df (pd.DataFrame): Feature dataframe.
        outlier_analysis (dict): Outlier analysis.
        output_dir (str): Output directory.
        top_n (int): Number of top features to plot.
    """
    logger.info("Plotting outlier analysis...")
    outlier_percentages = [(feature, outlier_analysis[feature]['outlier_percentage'])
                          for feature in outlier_analysis.keys()]
    outlier_percentages.sort(key=lambda x: x[1], reverse=True)
    top_features = [feature for feature, _ in outlier_percentages[:top_n]]
    fig, axes = plt.subplots(2, 2, figsize=(15, 10))
    axes = axes.flatten()
    for i, feature in enumerate(top_features[:4]):
        values = df[feature].dropna()
        outlier_info = outlier_analysis[feature]
        axes[i].boxplot(values)
        axes[i].set_title(f'{feature}\nOutliers: {outlier_info["outlier_count"]} ({outlier_info["outlier_percentage"]:.1f}%)')
        axes[i].set_ylabel('Value')
    plt.tight_layout()
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    plt.savefig(os.path.join(plots_dir, "outlier_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()

def generate_structured_summary(
    quality_metrics: dict,
    basic_stats: dict,
    distribution_analysis: dict,
    outlier_analysis: dict,
    feature_variability: pd.DataFrame,
    data_health_summary: dict,
    df: pd.DataFrame
) -> dict:
    """
    Generate structured summary data for frontend display.

    Args:
        quality_metrics (dict): Feature quality metrics.
        basic_stats (dict): Basic statistics.
        distribution_analysis (dict): Distribution analysis.
        outlier_analysis (dict): Outlier analysis.
        feature_variability (pd.DataFrame): Feature variability ranking.
        data_health_summary (dict): Data health summary.
        df (pd.DataFrame): Feature dataframe.

    Returns:
        dict: Structured summary for frontend.
    """
    total_features = len(quality_metrics)
    if total_features == 0:
        return {
            "status": "error",
            "message": "No features to analyze",
            "summary_text": "No features to analyze"
        }
    quality_scores = [metrics['quality_score'] for metrics in quality_metrics.values()]
    avg_quality = np.mean(quality_scores)
    high_missing_count = sum(1 for metrics in quality_metrics.values()
                            if metrics['quality_flags']['high_missing'])
    zero_variance_count = sum(1 for metrics in quality_metrics.values()
                             if metrics['quality_flags']['zero_variance'])
    high_extreme_count = sum(1 for metrics in quality_metrics.values()
                            if metrics['quality_flags']['high_extreme'])
    normal_features = sum(1 for info in distribution_analysis.values() if info['is_normal'])
    top_features = feature_variability.head(10)['feature'].tolist()
    worst_features = []
    for feature_name, metrics in quality_metrics.items():
        worst_features.append({
            'feature': feature_name,
            'quality_score': metrics['quality_score'],
            'quality_grade': metrics['quality_grade'],
            'missing_rate': metrics['missing_rate'],
            'zero_variance': metrics['zero_variance'],
            'extreme_rate': metrics['extreme_rate']
        })
    worst_features.sort(key=lambda x: x['quality_score'])
    top_worst_features = worst_features[:5]
    summary_text = f"""
Feature Profiling / Quality Assessment Summary
==============================================

Dataset Overview:
- Total features analyzed: {total_features}
- Total samples: {len(df)}

Data Health Assessment:
- Overall Health Score: {data_health_summary.get('overall_health_score', 0):.1f}/100
- Overall Health Grade: {data_health_summary.get('overall_health_grade', 'N/A')}
- Average Feature Quality: {avg_quality:.1f}/100

Quality Issues Identified:
- Features with >90% missing values: {high_missing_count} ({high_missing_count/total_features*100:.1f}%)
- Features with zero variance: {zero_variance_count} ({zero_variance_count/total_features*100:.1f}%)
- Features with >10% extreme values: {high_extreme_count} ({high_extreme_count/total_features*100:.1f}%)

Distribution Characteristics:
- Normal-like distributions: {normal_features} ({normal_features/total_features*100:.1f}%)
- Non-normal distributions: {total_features - normal_features} ({(total_features-normal_features)/total_features*100:.1f}%)

Top 10 Most Variable Features:
{chr(10).join([f"{i+1}. {feature}" for i, feature in enumerate(top_features)])}

Quality Recommendations:
{chr(10).join([f"- {rec}" for rec in data_health_summary.get('recommendations', [])])}

Key Findings:
- Feature quality profiling completed
- Data health assessment performed
- Quality improvement recommendations generated
- Distribution characteristics analyzed for quality context

Results saved to output directory.
"""
    return {
        "status": "success",
        "summary_text": summary_text,
        "frontend_summary": {
            "total_features": total_features,
            "total_samples": len(df),
            "overall_health_score": data_health_summary.get('overall_health_score', 0),
            "overall_health_grade": data_health_summary.get('overall_health_grade', 'N/A'),
            "average_quality_score": avg_quality,
            "quality_distribution": {
                "grade_a": sum(1 for metrics in quality_metrics.values() if metrics['quality_grade'] == 'A'),
                "grade_b": sum(1 for metrics in quality_metrics.values() if metrics['quality_grade'] == 'B'),
                "grade_c": sum(1 for metrics in quality_metrics.values() if metrics['quality_grade'] == 'C'),
                "grade_d": sum(1 for metrics in quality_metrics.values() if metrics['quality_grade'] == 'D'),
                "grade_f": sum(1 for metrics in quality_metrics.values() if metrics['quality_grade'] == 'F')
            },
            "quality_issues": {
                "high_missing_features": high_missing_count,
                "zero_variance_features": zero_variance_count,
                "high_extreme_features": high_extreme_count,
                "problematic_features_ratio": (high_missing_count + zero_variance_count + high_extreme_count) / total_features
            },
            "distribution_characteristics": {
                "normal_features": normal_features,
                "non_normal_features": total_features - normal_features,
                "normal_percentage": normal_features / total_features * 100
            },
            "top_variable_features": top_features[:5],
            "top_worst_features": top_worst_features,
            "recommendations": data_health_summary.get('recommendations', [])
        }
    }