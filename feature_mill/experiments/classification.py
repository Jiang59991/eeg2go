"""
Classification Experiment Module

This module performs classification experiments using EEG features:
- Age group classification (young vs old)
- Multi-class age classification
- Sex classification
- Cross-validation and model comparison
- Feature importance analysis
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_recall_fscore_support, roc_auc_score, roc_curve
)
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
    Run classification experiment
    
    Args:
        df_feat: Feature matrix dataframe
        df_meta: Metadata dataframe
        output_dir: Output directory
        **kwargs: Extra arguments
            - target_var: Target variable ('age_group', 'sex', 'age_class'), default 'age_group'
            - age_threshold: Age threshold for binary classification, default 65 (only used for age-related targets)
            - test_size: Test set size, default 0.2
            - n_splits: Number of CV folds, default 5
            - generate_plots: Whether to generate plots, default True
    
    Returns:
        str: Experiment summary
    """
    logger.info(f"Start classification experiment")
    logger.info(f"Feature matrix shape: {df_feat.shape}")
    logger.info(f"Metadata shape: {df_meta.shape}")
    
    # Get parameters
    target_var = kwargs.get('target_var', 'age_group')
    test_size = kwargs.get('test_size', 0.2)
    random_state = 42  # 固定为42，不再作为用户输入参数
    n_splits = kwargs.get('n_splits', 5)
    generate_plots = kwargs.get('generate_plots', True)
    
    # 只有当target_var是年龄相关时才使用age_threshold
    age_threshold = None
    if target_var in ['age_group', 'age_class']:
        age_threshold = kwargs.get('age_threshold', 65)
        logger.info(f"Using age threshold: {age_threshold} for target variable: {target_var}")
    else:
        logger.info(f"Age threshold not applicable for target variable: {target_var}")
    
    # Data preprocessing
    df_processed, X, y = preprocess_data(df_feat, df_meta, target_var, age_threshold)
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Define models
    models = {
        'Logistic Regression': LogisticRegression(random_state=random_state, max_iter=1000),
        'Random Forest': RandomForestClassifier(n_estimators=100, random_state=random_state),
        'Gradient Boosting': GradientBoostingClassifier(n_estimators=100, random_state=random_state),
        'SVM': SVC(random_state=random_state, probability=True)
    }
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        results[name] = train_and_evaluate_model(
            model, X_train_scaled, X_test_scaled, y_train, y_test, 
            X_train, y_train, n_splits, random_state
        )
    
    # Compare models
    comparison_results = compare_models(results)
    
    # Save results
    save_classification_results(results, comparison_results, output_dir)
    
    # Visualizations
    if generate_plots:
        plot_classification_results(results, output_dir)
        plot_feature_importance_analysis(results, X_train, output_dir)
    
    # Generate summary
    summary = generate_summary(results, comparison_results, df_processed, target_var)
    
    logger.info(f"Classification experiment completed, results saved to: {output_dir}")
    return summary


def preprocess_data(df_feat: pd.DataFrame, df_meta: pd.DataFrame, 
                   target_var: str, age_threshold: int) -> tuple:
    """Preprocess data for classification"""
    # Merge features and metadata
    df_combined = pd.merge(df_feat, df_meta, on='recording_id', how='inner')
    
    # Prepare features (X)
    feature_cols = [col for col in df_feat.columns if col != 'recording_id']
    X = df_combined[feature_cols].copy()
    
    # Prepare target (y) based on target_var
    if target_var == 'age_group':
        # Binary classification: young vs old
        age_col = 'age' if 'age' in df_combined.columns else 'age_days'
        if age_col in df_combined.columns:
            y = (df_combined[age_col] >= age_threshold).astype(int)
            y = y.map({0: 'Young', 1: 'Old'})
        else:
            raise ValueError(f"Age column not found for age_group classification")
    
    elif target_var == 'sex':
        # Binary classification: male vs female
        if 'sex' in df_combined.columns:
            y = df_combined['sex'].map({'M': 'Male', 'F': 'Female', 'Male': 'Male', 'Female': 'Female'})
        else:
            raise ValueError("Sex column not found")
    
    elif target_var == 'age_class':
        # Multi-class age classification
        age_col = 'age' if 'age' in df_combined.columns else 'age_days'
        if age_col in df_combined.columns:
            ages = df_combined[age_col]
            y = pd.cut(ages, bins=[0, 30, 50, 70, 100], labels=['Young', 'Middle', 'Senior', 'Elderly'])
        else:
            raise ValueError(f"Age column not found for age_class classification")
    
    else:
        raise ValueError(f"Unsupported target variable: {target_var}")
    
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
    
    # Encode target variable
    le = LabelEncoder()
    y_encoded = le.fit_transform(y)
    
    logger.info(f"Final data shape: X={X.shape}, y={y.shape}")
    logger.info(f"Target classes: {le.classes_}")
    logger.info(f"Class distribution: {pd.Series(y).value_counts().to_dict()}")
    
    return df_combined, X, y_encoded


def train_and_evaluate_model(model, X_train_scaled, X_test_scaled, y_train, y_test, 
                           X_train_orig, y_train_orig, n_splits, random_state):
    """Train and evaluate a single model"""
    
    # Train model
    model.fit(X_train_scaled, y_train)
    
    # Predictions
    y_pred = model.predict(X_test_scaled)
    y_pred_proba = model.predict_proba(X_test_scaled) if hasattr(model, 'predict_proba') else None
    
    # Cross-validation
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    cv_scores = cross_val_score(model, X_train_scaled, y_train, cv=cv, scoring='accuracy')
    
    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(y_test, y_pred, average='weighted')
    
    # ROC AUC (for binary classification)
    roc_auc = None
    if len(np.unique(y_test)) == 2 and y_pred_proba is not None:
        roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
    
    # Feature importance
    feature_importance = None
    if hasattr(model, 'feature_importances_'):
        feature_importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        feature_importance = np.abs(model.coef_[0])
    
    return {
        'model': model,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'roc_auc': roc_auc,
        'cv_scores': cv_scores,
        'cv_mean': np.mean(cv_scores),
        'cv_std': np.std(cv_scores),
        'y_pred': y_pred,
        'y_pred_proba': y_pred_proba,
        'feature_importance': feature_importance,
        'confusion_matrix': confusion_matrix(y_test, y_pred)
    }


def compare_models(results: dict) -> pd.DataFrame:
    """Compare different models"""
    comparison_data = []
    
    for model_name, result in results.items():
        comparison_data.append({
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1-Score': result['f1_score'],
            'ROC-AUC': result['roc_auc'],
            'CV-Mean': result['cv_mean'],
            'CV-Std': result['cv_std']
        })
    
    return pd.DataFrame(comparison_data)


def save_classification_results(results: dict, comparison_results: pd.DataFrame, output_dir: str):
    """Save classification results with organized file structure"""
    
    # 创建子目录
    data_dir = os.path.join(output_dir, "data")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 保存每个模型的结果
    for model_name, result in results.items():
        if result is not None:
            # 保存模型性能指标
            performance_file = os.path.join(data_dir, f"performance_{model_name.lower().replace(' ', '_')}.csv")
            # 从result中提取性能指标
            performance_data = {
                'accuracy': result['accuracy'],
                'precision': result['precision'],
                'recall': result['recall'],
                'f1_score': result['f1_score'],
                'roc_auc': result['roc_auc'],
                'cv_mean': result['cv_mean'],
                'cv_std': result['cv_std']
            }
            performance_df = pd.DataFrame([performance_data])
            performance_df.to_csv(performance_file, index=False)
            
            # 保存特征重要性（如果有）
            if 'feature_importance' in result and result['feature_importance'] is not None:
                importance_file = os.path.join(data_dir, f"importance_{model_name.lower().replace(' ', '_')}.csv")
                # feature_importance是numpy数组，需要转换为DataFrame
                importance_array = result['feature_importance']
                importance_df = pd.DataFrame({
                    'feature': [f'feature_{i}' for i in range(len(importance_array))],
                    'importance': importance_array
                })
                importance_df = importance_df.sort_values('importance', ascending=False)
                importance_df.to_csv(importance_file, index=False)
    
    # 保存模型比较结果
    if comparison_results is not None and len(comparison_results) > 0:
        comparison_file = os.path.join(data_dir, "models_comparison.csv")
        comparison_results.to_csv(comparison_file, index=False)
    
    # 创建结果索引文件
    results_index = {
        "experiment_type": "classification",
        "files": {
            "models_comparison": "data/models_comparison.csv"
        },
        "plots": {
            "models_comparison": "plots/models_comparison.png",
            "feature_importance": "plots/feature_importance_comparison.png",
            "confusion_matrices": "plots/confusion_matrices.png"
        },
        "summary": {
            "models_used": list(results.keys()),
            "total_models": len(results),
            "generated_at": datetime.now().isoformat()
        }
    }
    
    # 添加每个模型的文件
    for model_name, result in results.items():
        if result is not None:
            model_key = model_name.lower().replace(' ', '_')
            results_index["files"][f"performance_{model_key}"] = f"data/performance_{model_key}.csv"
            if 'feature_importance' in result and result['feature_importance'] is not None:
                results_index["files"][f"importance_{model_key}"] = f"data/importance_{model_key}.csv"
    
    with open(os.path.join(output_dir, "results_index.json"), "w", encoding='utf-8') as f:
        json.dump(results_index, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Classification results saved to {output_dir} with organized structure")


def plot_classification_results(results: dict, output_dir: str):
    """Plot classification results"""
    logger.info("[classification] Plotting classification results...")
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # 1. Model comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    
    # Accuracy comparison
    models = list(results.keys())
    accuracies = [results[model]['accuracy'] for model in models]
    axes[0, 0].bar(models, accuracies)
    axes[0, 0].set_title('Model Accuracy Comparison')
    axes[0, 0].set_ylabel('Accuracy')
    axes[0, 0].tick_params(axis='x', rotation=45)
    
    # F1-Score comparison
    f1_scores = [results[model]['f1_score'] for model in models]
    axes[0, 1].bar(models, f1_scores)
    axes[0, 1].set_title('Model F1-Score Comparison')
    axes[0, 1].set_ylabel('F1-Score')
    axes[0, 1].tick_params(axis='x', rotation=45)
    
    # Cross-validation scores
    cv_means = [results[model]['cv_mean'] for model in models]
    cv_stds = [results[model]['cv_std'] for model in models]
    axes[1, 0].bar(models, cv_means, yerr=cv_stds, capsize=5)
    axes[1, 0].set_title('Cross-Validation Scores')
    axes[1, 0].set_ylabel('CV Accuracy')
    axes[1, 0].tick_params(axis='x', rotation=45)
    
    # ROC-AUC comparison (if available)
    roc_aucs = [results[model]['roc_auc'] for model in models]
    roc_aucs = [auc for auc in roc_aucs if auc is not None]
    if roc_aucs:
        models_with_auc = [model for model, auc in zip(models, [results[model]['roc_auc'] for model in models]) if auc is not None]
        axes[1, 1].bar(models_with_auc, roc_aucs)
        axes[1, 1].set_title('ROC-AUC Comparison')
        axes[1, 1].set_ylabel('ROC-AUC')
        axes[1, 1].tick_params(axis='x', rotation=45)
    else:
        axes[1, 1].text(0.5, 0.5, 'ROC-AUC not available\nfor multi-class problems', 
                       ha='center', va='center', transform=axes[1, 1].transAxes)
        axes[1, 1].set_title('ROC-AUC Comparison')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "models_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()
    
    # 2. Confusion matrices
    n_models = len(results)
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    axes = axes.flatten()
    
    for i, (model_name, result) in enumerate(results.items()):
        if i >= 4:
            break
            
        cm = result['confusion_matrix']
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix - {model_name}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    # Hide empty subplots
    for i in range(n_models, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrices.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance_analysis(results: dict, X_train: pd.DataFrame, output_dir: str):
    """Plot feature importance analysis"""
    logger.info("[classification] Plotting feature importance analysis...")
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Collect feature importance from models that have it
    importance_data = {}
    
    for model_name, result in results.items():
        if result['feature_importance'] is not None:
            importance_data[model_name] = result['feature_importance']
    
    if not importance_data:
        return
    
    # Plot top features from each model
    n_models = len(importance_data)
    fig, axes = plt.subplots(2, 2, figsize=(15, 12))
    axes = axes.flatten()
    
    for i, (model_name, importance) in enumerate(importance_data.items()):
        if i >= 4:
            break
            
        # Get top 10 features
        top_indices = np.argsort(importance)[-10:]
        top_features = [f"Feature_{idx}" for idx in top_indices]
        top_importance = importance[top_indices]
        
        axes[i].barh(range(len(top_features)), top_importance)
        axes[i].set_yticks(range(len(top_features)))
        axes[i].set_yticklabels(top_features)
        axes[i].set_title(f'Top Features - {model_name}')
        axes[i].set_xlabel('Importance Score')
        axes[i].invert_yaxis()
    
    # Hide empty subplots
    for i in range(n_models, 4):
        axes[i].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_importance_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()


def generate_summary(results: dict, comparison_results: pd.DataFrame, 
                    df_processed: pd.DataFrame, target_var: str) -> str:
    """Generate experiment summary"""
    
    # Find best model
    best_model = comparison_results.loc[comparison_results['Accuracy'].idxmax()]
    
    # Calculate average performance
    avg_accuracy = comparison_results['Accuracy'].mean()
    avg_f1 = comparison_results['F1-Score'].mean()
    
    # Count total samples and features
    total_samples = len(df_processed)
    total_features = len([col for col in df_processed.columns if col not in ['recording_id', 'age', 'age_days', 'sex']])
    
    summary = f"""
Classification Experiment Summary
================================

Dataset Overview:
- Total samples: {total_samples}
- Total features: {total_features}
- Target variable: {target_var}

Model Performance:
- Best model: {best_model['Model']} (Accuracy: {best_model['Accuracy']:.3f})
- Average accuracy across models: {avg_accuracy:.3f}
- Average F1-score across models: {avg_f1:.3f}

Individual Model Results:
"""
    
    for _, row in comparison_results.iterrows():
        summary += f"- {row['Model']}: Accuracy={row['Accuracy']:.3f}, F1={row['F1-Score']:.3f}, CV={row['CV-Mean']:.3f}±{row['CV-Std']:.3f}\n"
    
    summary += f"""
Key Findings:
- {len(results)} different classification models evaluated
- Cross-validation performed with 5 folds
- Feature importance analysis completed
- Confusion matrices generated

Best performing model: {best_model['Model']}
- Test Accuracy: {best_model['Accuracy']:.3f}
- Test F1-Score: {best_model['F1-Score']:.3f}
- Cross-validation Accuracy: {best_model['CV-Mean']:.3f} ± {best_model['CV-Std']:.3f}

Results saved to output directory.
"""
    
    return summary 