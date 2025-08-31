"""
Feature Selection Experiment Module

This module performs feature selection using multiple methods:
- Variance-based selection
- Correlation-based selection
- Mutual information
- Recursive feature elimination
- L1-based selection (Lasso)
- Principal component analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_selection import (
    VarianceThreshold, SelectKBest, f_regression, mutual_info_regression,
    RFE, SelectFromModel
)
from sklearn.linear_model import Lasso, LinearRegression
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import cross_val_score
import warnings
from logging_config import logger  # 使用全局logger
import json
from datetime import datetime
warnings.filterwarnings('ignore')

# Set English font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def run(df_feat: pd.DataFrame, df_meta: pd.DataFrame, output_dir: str, **kwargs) -> str:
    """
    Run feature selection experiment
    
    Args:
        df_feat: Feature matrix dataframe
        df_meta: Metadata dataframe
        output_dir: Output directory
        **kwargs: Extra arguments
            - target_var: Target variable for supervised selection, default 'age'
            - n_features: Number of features to select, default 20
            - variance_threshold: Variance threshold for variance-based selection, default 0.01
            - correlation_threshold: Correlation threshold for correlation-based selection, default 0.95
            - generate_plots: Whether to generate plots, default True
    
    Returns:
        str: Experiment summary
    """
    logger.info(f"Start feature selection experiment")
    logger.info(f"Feature matrix shape: {df_feat.shape}")
    logger.info(f"Metadata shape: {df_meta.shape}")
    
    # Get parameters
    target_var = kwargs.get('target_var', 'age')
    n_features = kwargs.get('n_features', 20)
    variance_threshold = kwargs.get('variance_threshold', 0.01)
    correlation_threshold = kwargs.get('correlation_threshold', 0.95)
    generate_plots = kwargs.get('generate_plots', True)
    
    # Data preprocessing
    df_processed, X, y = preprocess_data(df_feat, df_meta, target_var)
    
    # Perform different feature selection methods
    selection_results = {}
    
    # 1. Variance-based selection
    selection_results['variance'] = variance_based_selection(X, variance_threshold)
    
    # 2. Correlation-based selection
    selection_results['correlation'] = correlation_based_selection(X, correlation_threshold)
    
    # 3. Univariate selection (F-test)
    selection_results['univariate_f'] = univariate_selection(X, y, n_features, 'f_regression')
    
    # 4. Mutual information
    selection_results['mutual_info'] = univariate_selection(X, y, n_features, 'mutual_info')
    
    # 5. L1-based selection (Lasso)
    selection_results['lasso'] = lasso_based_selection(X, y, n_features)
    
    # 6. Recursive feature elimination
    selection_results['rfe'] = recursive_feature_elimination(X, y, n_features)
    
    # 7. Principal component analysis
    selection_results['pca'] = principal_component_analysis(X, n_features)
    
    # Compare methods
    comparison_results = compare_selection_methods(selection_results, X, y)
    
    # Save results
    save_selection_results(selection_results, comparison_results, output_dir)
    
    # Visualizations
    if generate_plots:
        plot_selection_comparison(selection_results, output_dir)
        plot_feature_importance_analysis(selection_results, X, y, output_dir)
    
    # Generate summary
    summary = generate_summary(selection_results, comparison_results, df_processed, target_var)
    
    logger.info(f"Feature selection experiment completed, results saved to: {output_dir}")
    return summary


def preprocess_data(df_feat: pd.DataFrame, df_meta: pd.DataFrame, target_var: str) -> tuple:
    """Preprocess data for feature selection"""
    # Merge features and metadata
    df_combined = pd.merge(df_feat, df_meta, on='recording_id', how='inner')
    
    # Prepare features (X) and target (y)
    feature_cols = [col for col in df_feat.columns if col != 'recording_id']
    X = df_combined[feature_cols].copy()
    y = df_combined[target_var].copy()
    
    # Remove rows with missing target values
    mask = ~y.isna()
    X = X[mask]
    y = y[mask]
    
    # Remove features with too many missing values
    missing_threshold = 0.5
    missing_ratio = X.isnull().sum() / len(X)
    columns_to_drop = missing_ratio[missing_ratio > missing_threshold].index
    X = X.drop(columns=columns_to_drop)
    
    # Fill missing values
    X = X.fillna(X.median())
    
    # Remove non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    logger.info(f"Final data shape: X={X.shape}, y={y.shape}")
    return df_combined, X, y


def variance_based_selection(X: pd.DataFrame, threshold: float) -> dict:
    """Variance-based feature selection"""
    logger.info("Performing variance-based selection...")
    
    selector = VarianceThreshold(threshold=threshold)
    X_selected = selector.fit_transform(X)
    selected_features = X.columns[selector.get_support()].tolist()
    
    return {
        'method': 'Variance-based',
        'selected_features': selected_features,
        'n_selected': len(selected_features),
        'threshold': threshold,
        'removed_features': [col for col in X.columns if col not in selected_features]
    }


def correlation_based_selection(X: pd.DataFrame, threshold: float) -> dict:
    """Correlation-based feature selection (remove highly correlated features)"""
    logger.info("Performing correlation-based selection...")
    
    corr_matrix = X.corr().abs()
    upper_tri = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    
    # Find features to drop
    to_drop = [column for column in upper_tri.columns if any(upper_tri[column] > threshold)]
    
    selected_features = [col for col in X.columns if col not in to_drop]
    
    return {
        'method': 'Correlation-based',
        'selected_features': selected_features,
        'n_selected': len(selected_features),
        'threshold': threshold,
        'removed_features': to_drop
    }


def univariate_selection(X: pd.DataFrame, y: pd.Series, n_features: int, method: str) -> dict:
    """Univariate feature selection"""
    logger.info(f"Performing {method} univariate selection...")
    
    if method == 'f_regression':
        selector = SelectKBest(score_func=f_regression, k=min(n_features, X.shape[1]))
    else:  # mutual_info
        selector = SelectKBest(score_func=mutual_info_regression, k=min(n_features, X.shape[1]))
    
    X_selected = selector.fit_transform(X, y)
    selected_features = X.columns[selector.get_support()].tolist()
    scores = selector.scores_[selector.get_support()]
    
    return {
        'method': f'Univariate ({method})',
        'selected_features': selected_features,
        'n_selected': len(selected_features),
        'scores': scores,
        'feature_scores': dict(zip(selected_features, scores))
    }


def lasso_based_selection(X: pd.DataFrame, y: pd.Series, n_features: int) -> dict:
    """L1-based feature selection using Lasso"""
    logger.info("Performing Lasso-based selection...")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Use Lasso with different alphas to get desired number of features
    alpha = 0.01
    while True:
        lasso = Lasso(alpha=alpha, random_state=42)
        lasso.fit(X_scaled, y)
        n_selected = np.sum(lasso.coef_ != 0)
        
        if n_selected <= n_features or alpha > 10:
            break
        alpha *= 1.5
    
    selected_features = X.columns[lasso.coef_ != 0].tolist()
    coefficients = lasso.coef_[lasso.coef_ != 0]
    
    return {
        'method': 'Lasso-based',
        'selected_features': selected_features,
        'n_selected': len(selected_features),
        'alpha': alpha,
        'coefficients': coefficients,
        'feature_importance': dict(zip(selected_features, np.abs(coefficients)))
    }


def recursive_feature_elimination(X: pd.DataFrame, y: pd.Series, n_features: int) -> dict:
    """Recursive feature elimination"""
    logger.info("Performing recursive feature elimination...")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    estimator = LinearRegression()
    rfe = RFE(estimator=estimator, n_features_to_select=min(n_features, X.shape[1]))
    rfe.fit(X_scaled, y)
    
    selected_features = X.columns[rfe.support_].tolist()
    rankings = rfe.ranking_
    
    return {
        'method': 'Recursive Feature Elimination',
        'selected_features': selected_features,
        'n_selected': len(selected_features),
        'feature_rankings': dict(zip(X.columns, rankings))
    }


def principal_component_analysis(X: pd.DataFrame, n_features: int) -> dict:
    """Principal component analysis"""
    logger.info("Performing principal component analysis...")
    
    # Standardize features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # Fit PCA
    n_components = min(n_features, X.shape[1])
    pca = PCA(n_components=n_components)
    pca.fit(X_scaled)
    
    # Get feature importance based on PCA loadings
    loadings = pca.components_
    feature_importance = np.sum(np.abs(loadings), axis=0)
    
    # Select top features based on PCA importance
    top_indices = np.argsort(feature_importance)[-n_features:]
    selected_features = X.columns[top_indices].tolist()
    
    return {
        'method': 'Principal Component Analysis',
        'selected_features': selected_features,
        'n_selected': len(selected_features),
        'explained_variance_ratio': pca.explained_variance_ratio_,
        'cumulative_variance': np.cumsum(pca.explained_variance_ratio_),
        'feature_importance': dict(zip(X.columns, feature_importance))
    }


def compare_selection_methods(selection_results: dict, X: pd.DataFrame, y: pd.Series) -> pd.DataFrame:
    """Compare different feature selection methods"""
    logger.info("Comparing selection methods...")
    
    comparison_data = []
    
    for method_name, result in selection_results.items():
        if 'selected_features' not in result:
            continue
            
        selected_features = result['selected_features']
        if len(selected_features) == 0:
            continue
        
        # Use selected features for simple linear regression
        X_selected = X[selected_features]
        
        # Cross-validation score
        try:
            from sklearn.linear_model import LinearRegression
            from sklearn.model_selection import cross_val_score
            
            model = LinearRegression()
            scores = cross_val_score(model, X_selected, y, cv=5, scoring='r2')
            
            comparison_data.append({
                'Method': method_name,
                'N_Features': len(selected_features),
                'Mean_CV_Score': np.mean(scores),
                'Std_CV_Score': np.std(scores),
                'Selected_Features': ', '.join(selected_features)
            })
        except Exception as e:
            logger.error(f"Error evaluating {method_name}: {e}")
            comparison_data.append({
                'Method': method_name,
                'N_Features': len(selected_features),
                'Mean_CV_Score': np.nan,
                'Std_CV_Score': np.nan,
                'Selected_Features': ', '.join(selected_features)
            })
    
    return pd.DataFrame(comparison_data)


def save_selection_results(selection_results: dict, comparison_results: pd.DataFrame, output_dir: str):
    """Save feature selection results with organized file structure"""
    
    # 创建子目录
    data_dir = os.path.join(output_dir, "data")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 保存每种方法的结果
    for method_name, result in selection_results.items():
        if result is not None and len(result) > 0:
            result_file = os.path.join(data_dir, f"selection_{method_name}.csv")
            # 将字典转换为DataFrame
            if isinstance(result, dict):
                # 创建包含选择特征的DataFrame
                selected_features = result.get('selected_features', [])
                removed_features = result.get('removed_features', [])
                
                # 创建结果DataFrame
                result_data = {
                    'method': [result.get('method', method_name)],
                    'n_selected': [result.get('n_selected', len(selected_features))],
                    'threshold': [result.get('threshold', 'N/A')],
                    'selected_features': [', '.join(selected_features)],
                    'removed_features': [', '.join(removed_features)]
                }
                
                # 如果有特征重要性信息，也保存
                if 'feature_importance' in result:
                    importance_data = result['feature_importance']
                    if isinstance(importance_data, dict):
                        result_data['feature_importance'] = [json.dumps(importance_data)]
                    else:
                        result_data['feature_importance'] = [str(importance_data)]
                
                result_df = pd.DataFrame(result_data)
                result_df.to_csv(result_file, index=False)
            else:
                # 如果result已经是DataFrame，直接保存
                result.to_csv(result_file, index=False)
    
    # 保存方法比较结果
    if comparison_results is not None and len(comparison_results) > 0:
        comparison_file = os.path.join(data_dir, "selection_methods_comparison.csv")
        comparison_results.to_csv(comparison_file, index=False)
    
    # 创建结果索引文件
    results_index = {
        "experiment_type": "feature_selection",
        "files": {
            "methods_comparison": "data/selection_methods_comparison.csv"
        },
        "plots": {
            "selection_comparison": "plots/selection_methods_comparison.png",
            "feature_importance": "plots/feature_importance_comparison.png"
        },
        "summary": {
            "methods_used": list(selection_results.keys()),
            "total_methods": len(selection_results),
            "generated_at": datetime.now().isoformat()
        }
    }
    
    # 添加每种方法的文件
    for method_name, result in selection_results.items():
        if result is not None and len(result) > 0:
            results_index["files"][f"selection_{method_name}"] = f"data/selection_{method_name}.csv"
    
    with open(os.path.join(output_dir, "results_index.json"), "w", encoding='utf-8') as f:
        json.dump(results_index, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Feature selection results saved to {output_dir} with organized structure")


def plot_selection_comparison(selection_results: dict, output_dir: str):
    """Plot comparison of different selection methods"""
    logger.info("Plotting selection comparison...")
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Compare number of selected features
    methods = []
    n_features = []
    
    for method_name, result in selection_results.items():
        if 'selected_features' in result:
            methods.append(result['method'])
            n_features.append(result['n_selected'])
    
    plt.figure(figsize=(10, 6))
    bars = plt.bar(methods, n_features)
    plt.title('Number of Features Selected by Each Method')
    plt.xlabel('Selection Method')
    plt.ylabel('Number of Features')
    plt.xticks(rotation=45, ha='right')
    
    # Add value labels on bars
    for bar, value in zip(bars, n_features):
        plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5, 
                str(value), ha='center', va='bottom')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "selection_methods_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance_analysis(selection_results: dict, X: pd.DataFrame, y: pd.Series, output_dir: str):
    """Plot feature importance analysis"""
    logger.info("Plotting feature importance analysis...")
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Collect feature importance from different methods
    importance_data = {}
    
    for method_name, result in selection_results.items():
        if 'feature_importance' in result:
            importance_data[method_name] = result['feature_importance']
        elif 'feature_scores' in result:
            importance_data[method_name] = result['feature_scores']
    
    if not importance_data:
        return
    
    # Plot top features from each method
    n_methods = len(importance_data)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (method_name, importance_dict) in enumerate(importance_data.items()):
        if i >= 4:
            break
            
        # Sort by importance
        sorted_items = sorted(importance_dict.items(), key=lambda x: x[1], reverse=True)
        top_features = sorted_items[:10]
        
        features, scores = zip(*top_features)
        
        axes[i].barh(range(len(features)), scores)
        axes[i].set_yticks(range(len(features)))
        axes[i].set_yticklabels(features)
        axes[i].set_title(f'Top Features - {method_name}')
        axes[i].set_xlabel('Importance Score')
    
    # Hide empty subplots
    for i in range(len(importance_data), 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_importance_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary(selection_results: dict, comparison_results: pd.DataFrame, 
                    df_processed: pd.DataFrame, target_var: str) -> str:
    """Generate experiment summary"""
    
    total_features = len([col for col in df_processed.columns if col not in ['recording_id', target_var]])
    
    # Count features selected by each method
    method_summary = []
    for method_name, result in selection_results.items():
        if 'selected_features' in result:
            method_summary.append({
                'method': result['method'],
                'n_selected': result['n_selected'],
                'percentage': result['n_selected'] / total_features * 100
            })
    
    # Find most commonly selected features
    feature_counts = {}
    for result in selection_results.values():
        if 'selected_features' in result:
            for feature in result['selected_features']:
                feature_counts[feature] = feature_counts.get(feature, 0) + 1
    
    top_common_features = sorted(feature_counts.items(), key=lambda x: x[1], reverse=True)[:10]
    
    summary = f"""
Feature Selection Analysis Summary
==================================

Dataset Overview:
- Total features: {total_features}
- Target variable: {target_var}

Selection Methods Applied:
"""
    
    for method_info in method_summary:
        summary += f"- {method_info['method']}: {method_info['n_selected']} features ({method_info['percentage']:.1f}%)\n"
    
    summary += f"""
Most Commonly Selected Features (across methods):
"""
    
    for feature, count in top_common_features:
        summary += f"- {feature}: selected by {count} methods\n"
    
    if not comparison_results.empty:
        # 找到最佳方法
        best_idx = comparison_results['Mean_CV_Score'].idxmax()
        best_method = comparison_results.loc[best_idx]
        summary += f"""
Model Performance Comparison:
- Best performing method: {best_method['Method']} (CV R² = {best_method['Mean_CV_Score']:.3f} ± {best_method['Std_CV_Score']:.3f})
"""
    
    summary += """
Key Findings:
- Multiple feature selection methods applied
- Feature importance rankings generated
- Cross-validation performance evaluated
- Common features across methods identified

Results saved to output directory.
"""
    
    return summary 