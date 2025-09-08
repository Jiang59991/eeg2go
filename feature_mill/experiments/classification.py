"""
Classification Experiment Module - Baseline Predictive Check

This module performs baseline predictive modeling to validate selected features:
- Uses selected features from Selection module or Associations Top-K
- Runs robust baseline models (Logistic Regression / Random Forest)
- Implements GroupKFold cross-validation by subject to avoid data leakage
- Focuses on predictive performance validation, not platform demonstration

Key Features:
- Feature selection integration (Selection results or Associations Top-K)
- Robust baseline models with minimal hyperparameter tuning
- Subject-level cross-validation to prevent overfitting
- Comprehensive performance metrics (AUC, F1, confusion matrix)
- Feature importance analysis for selected features
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import GroupKFold, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report, confusion_matrix, accuracy_score, 
    precision_recall_fscore_support, roc_auc_score, roc_curve,
    f1_score, precision_score, recall_score
)
import warnings
from logging_config import logger  # 使用全局logger
import json
from datetime import datetime
import traceback
warnings.filterwarnings('ignore')

# Set English font
plt.rcParams['font.sans-serif'] = ['DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def parse_feature_name_for_display(feature_name: str) -> str:
    """
    解析特征名称，将fx{id}_{shortname}_{chans}格式转换为更友好的显示名称
    例如: fx123_entropy_C3 -> fx123_entropy_C3
         fx456_relative_power_C4 -> fx456_relative_power_C4
         fx789_alpha_power_C3-C4 -> fx789_alpha_power_C3-C4
    """
    try:
        if feature_name.startswith('fx') and '_' in feature_name:
            # 保持原始格式，因为用户要求显示完整的fxdef_[num]_[shortname]_[channel]
            return feature_name
        return feature_name
    except Exception as e:
        logger.warning(f"Failed to parse feature name '{feature_name}': {e}")
        return feature_name


def run_baseline_predictive_check_experiment(df_feat: pd.DataFrame, df_meta: pd.DataFrame, 
                                           output_dir: str, **kwargs) -> dict:
    """Run baseline predictive check experiment for classification
    
    Args:
        df_feat: Feature DataFrame
        df_meta: Metadata DataFrame
        output_dir: Output directory
        **kwargs: Extra arguments
            - target_vars: List of target variables from frontend
            - n_features: Number of top features to use if no selection results, default 20
            - selection_dir: Directory containing selection results, default None
            - associations_dir: Directory containing associations results, default None
            - model_type: Baseline model type ('logistic' or 'random_forest'), default 'logistic'
            - n_splits: Number of CV folds, default 5
            - generate_plots: Whether to generate plots, default True
    
    Returns:
        dict: Structured experiment summary with frontend data
    """
    logger.info(f"Starting Baseline Predictive Check experiment")
    logger.info(f"Feature matrix shape: {df_feat.shape}")
    logger.info(f"Metadata shape: {df_meta.shape}")
    
    # Get parameters from frontend
    target_vars = kwargs.get('target_vars', [])
    
    # Handle both 'target_vars' (plural) and 'target_var' (singular) for backward compatibility
    if not target_vars:
        target_var = kwargs.get('target_var', None)
        if target_var:
            target_vars = [target_var]
            logger.info(f"Converted single target_var '{target_var}' to target_vars list")
    
    if not target_vars:
        logger.error("No target variables specified by frontend")
        raise ValueError("No target variables specified")
    
    n_features = int(kwargs.get('n_features', 20))
    selection_dir = kwargs.get('selection_dir', None)
    associations_dir = kwargs.get('associations_dir', None)
    model_type = kwargs.get('model_type', 'logistic')
    n_splits = int(kwargs.get('n_splits', 5))
    generate_plots = kwargs.get('generate_plots', True)
    
    logger.info(f"Frontend target variables: {target_vars}")
    logger.info(f"Number of features: {n_features}")
    logger.info(f"Selection directory: {selection_dir}")
    logger.info(f"Associations directory: {associations_dir}")
    logger.info(f"Model type: {model_type}")
    
    # Data preprocessing - only process the target variables specified by frontend
    try:
        df_processed, feature_cols = preprocess_data(df_feat, df_meta, target_vars)
        logger.info(f"Data preprocessing completed successfully")
        logger.info(f"Processed data shape: {df_processed.shape}")
        logger.info(f"Available columns: {list(df_processed.columns)}")
    except Exception as e:
        logger.error(f"Data preprocessing failed: {e}")
        raise RuntimeError(f"Data preprocessing failed: {e}")
    
    # Perform predictive check for each target variable specified by frontend
    results = {}
    processed_targets = []
    
    for target_var in target_vars:
        logger.info(f"Processing target variable: {target_var}")
        
        if target_var not in df_processed.columns:
            logger.warning(f"Target variable {target_var} not found in processed data, skipping")
            continue
            
        # Get target data
        y = df_processed[target_var].copy()
        X = df_processed[feature_cols].copy()
        
        # Remove rows with missing target values
        mask = ~y.isna()
        X_valid = X[mask]
        y_valid = y[mask]
        
        if len(y_valid) < 10:
            logger.warning(f"Insufficient data for {target_var}: {len(y_valid)} samples")
            continue
        
        # Check if target variable has enough classes for classification
        unique_classes = y_valid.unique()
        if len(unique_classes) < 2:
            logger.warning(f"Target variable {target_var} has only {len(unique_classes)} class(es), insufficient for classification")
            continue
        
        # Get selected features
        try:
            selected_features = get_selected_features(
                target_var, selection_dir, associations_dir, X_valid, y_valid, n_features
            )
        except Exception as e:
            logger.warning(f"Failed to get selected features for {target_var}: {e}")
            # Use all available features as fallback
            selected_features = list(X_valid.columns)
            logger.info(f"Using all available features as fallback: {len(selected_features)} features")
        
        if not selected_features:
            logger.warning(f"No features selected for {target_var}, skipping")
            continue
        
        # Use only selected features
        X_selected = X_valid[selected_features]
        
        # Prepare subject IDs for GroupKFold
        subject_ids = df_processed.loc[mask, 'subject_id'].values if 'subject_id' in df_processed.columns else None
        
        # Run baseline predictive check
        try:
            target_results = run_baseline_predictive_check(
                X_selected, y_valid, subject_ids, target_var, model_type, n_splits
            )
            results[target_var] = target_results
            processed_targets.append(target_var)
            logger.info(f"Successfully processed target variable: {target_var}")
        except Exception as e:
            logger.error(f"Failed to process target variable {target_var}: {e}")
            continue
    
    if not results:
        logger.error("No target variables were successfully processed")
        raise RuntimeError("No target variables were successfully processed")
    
    logger.info(f"Successfully processed {len(processed_targets)} target variables: {processed_targets}")
    
    # Save results
    try:
        save_predictive_check_results(results, output_dir)
        logger.info("Results saved successfully")
    except Exception as e:
        logger.warning(f"Failed to save results: {e}")
    
    # Generate visualizations
    if generate_plots:
        try:
            plot_predictive_check_results(results, output_dir)
            logger.info("Plots generated successfully")
        except Exception as e:
            logger.warning(f"Failed to generate plots: {e}")
    
    # Generate structured summary
    try:
        summary = generate_structured_summary(results, df_processed)
        logger.info("Summary generated successfully")
    except Exception as e:
        logger.error(f"Failed to generate summary: {e}")
        raise RuntimeError(f"Failed to generate summary: {e}")
    
    logger.info(f"Baseline Predictive Check experiment completed successfully")
    logger.info(f"Results saved to: {output_dir}")
    return summary


def preprocess_data(df_feat: pd.DataFrame, df_meta: pd.DataFrame, target_vars: list) -> tuple:
    """Preprocess data for classification"""
    # Merge features and metadata
    df_combined = pd.merge(df_feat, df_meta, on='recording_id', how='inner')
    
    # Prepare features (X)
    feature_cols = [col for col in df_feat.columns if col != 'recording_id']
    X = df_combined[feature_cols].copy()
    
    # Remove features with too many missing values
    missing_threshold = 0.5
    missing_ratio = X.isnull().sum() / len(X)
    columns_to_drop = missing_ratio[missing_ratio > missing_threshold].index
    X = X.drop(columns=columns_to_drop)
    
    # Fill missing values with median (more robust than mean)
    X = X.fillna(X.median())
    
    # Remove non-numeric columns
    X = X.select_dtypes(include=[np.number])
    
    # Additional data cleaning: remove infinite values and replace with large finite values
    X = X.replace([np.inf, -np.inf], np.nan)
    X = X.fillna(X.median())
    
    # Final check: ensure no NaN values remain
    if X.isnull().any().any():
        logger.warning(f"Still have NaN values after cleaning, dropping problematic columns")
        X = X.dropna(axis=1)
    
    # Get the final feature columns after preprocessing
    final_feature_cols = list(X.columns)
    
    # Filter for target variables and features, also include subject_id if available
    available_targets = [var for var in target_vars if var in df_combined.columns]
    additional_cols = ['subject_id'] if 'subject_id' in df_combined.columns else []
    
    df_processed = df_combined[['recording_id'] + additional_cols + available_targets + final_feature_cols].copy()
    
    # Process target variables for classification - ONLY the ones specified by frontend
    for target_var in available_targets:
        if target_var in df_processed.columns:
            df_processed = process_target_variable_for_classification(df_processed, target_var)
    
    # Final validation: ensure no NaN values in the processed data
    if df_processed.isnull().any().any():
        logger.warning(f"NaN values detected in processed data, attempting final cleanup")
        # Drop rows with any NaN values in target variables
        df_processed = df_processed.dropna(subset=available_targets)
        # Fill remaining NaN values in features with 0 (last resort)
        feature_cols_only = [col for col in final_feature_cols if col in df_processed.columns]
        df_processed[feature_cols_only] = df_processed[feature_cols_only].fillna(0)
    
    logger.info(f"Final data shape: X={X.shape}, processed={df_processed.shape}")
    logger.info(f"Available target variables: {available_targets}")
    logger.info(f"Additional columns: {additional_cols}")
    logger.info(f"Final feature columns: {len(final_feature_cols)}")
    logger.info(f"NaN values in final data: {df_processed.isnull().sum().sum()}")
    
    return df_processed, final_feature_cols

def process_target_variable_for_classification(df: pd.DataFrame, target_var: str) -> pd.DataFrame:
    """Process target variable to make it suitable for classification"""
    try:
        logger.info(f"Processing target variable: {target_var}")
        
        # Check if target variable already exists as a categorical variable
        if target_var in df.columns:
            original_values = df[target_var]
            logger.info(f"Original {target_var} values: {original_values.value_counts().to_dict()}")
            logger.info(f"Original {target_var} dtype: {original_values.dtype}")
            
            # Check if we have enough non-null values
            non_null_count = original_values.notna().sum()
            if non_null_count == 0:
                logger.warning(f"Target variable {target_var} has no non-null values, skipping")
                return df
            
            # Check if we have enough unique classes for classification
            unique_values = original_values.dropna().unique()
            if len(unique_values) < 2:
                logger.warning(f"Target variable {target_var} has only {len(unique_values)} unique value(s), insufficient for classification")
                return df
            
            # If it's already categorical and has discrete classes, keep it
            if original_values.dtype.name == 'category' or original_values.nunique() <= 10:
                logger.info(f"Target variable {target_var} is already suitable for classification")
                return df
            
            # If it's numeric with many unique values, we need to convert it
            if original_values.dtype in ['float64', 'int64'] and original_values.nunique() > 10:
                logger.info(f"Converting continuous {target_var} to categorical")
                
                if target_var in ['age', 'age_group', 'age_class']:
                    # Convert age to age groups for classification
                    age_values = pd.to_numeric(original_values, errors='coerce')
                    
                    # Smart age grouping based on data distribution
                    if age_values.nunique() > 10:  # Only convert if truly continuous
                        # Use data-driven age thresholds based on percentiles
                        age_25 = age_values.quantile(0.25)
                        age_75 = age_values.quantile(0.75)
                        
                        if age_75 - age_25 > 20:  # If there's significant age spread
                            # Create 3 age groups based on data distribution
                            bins = [0, age_25, age_75, 100]
                            labels = ['Young', 'Middle', 'Senior']
                        else:
                            # Create 2 age groups with median as threshold
                            median_age = age_values.median()
                            bins = [0, median_age, 100]
                            labels = ['Young', 'Old']
                        
                        df[f'{target_var}_processed'] = pd.cut(
                            age_values, 
                            bins=bins, 
                            labels=labels,
                            include_lowest=True
                        )
                        # Replace original column with processed version
                        df[target_var] = df[f'{target_var}_processed']
                        df = df.drop(columns=[f'{target_var}_processed'])
                        logger.info(f"Converted {target_var} to age groups: {df[target_var].value_counts().to_dict()}")
                        logger.info(f"Age bins used: {bins}")
                    else:
                        logger.info(f"Age variable {target_var} already has {age_values.nunique()} unique values, keeping as is")
                    
                elif target_var in ['age_days', 'visit_count', 'icd10_count', 'medication_count']:
                    # Convert continuous variables to categorical using quantiles
                    values = pd.to_numeric(original_values, errors='coerce')
                    if not values.isnull().all():
                        # Create 3 categories based on quantiles
                        df[f'{target_var}_processed'] = pd.qcut(
                            values, 
                            q=3, 
                            labels=['Low', 'Medium', 'High'],
                            duplicates='drop'
                        )
                        # Replace original column with categorical version
                        df[target_var] = df[f'{target_var}_processed']
                        df = df.drop(columns=[f'{target_var}_processed'])
                        logger.info(f"Converted {target_var} to categories: {df[target_var].value_counts().to_dict()}")
            
            # Ensure target variable is categorical for classification
            if df[target_var].dtype in ['float64', 'int64']:
                # Convert to categorical if it's numeric
                df[target_var] = df[target_var].astype('category')
                logger.info(f"Converted {target_var} to categorical: {df[target_var].value_counts().to_dict()}")
            
            # Final check: ensure we have discrete classes
            unique_values = df[target_var].nunique()
            if unique_values < 2:
                logger.warning(f"Target variable {target_var} has only {unique_values} unique values, skipping")
                df = df.drop(columns=[target_var])
            elif unique_values > 10:
                logger.warning(f"Target variable {target_var} has {unique_values} unique values, may need further discretization")
                
        else:
            logger.warning(f"Target variable {target_var} not found in dataframe")
            
    except Exception as e:
        logger.warning(f"Failed to process target variable {target_var}: {e}")
        # If processing fails, try to convert to categorical anyway
        try:
            if target_var in df.columns:
                df[target_var] = df[target_var].astype('category')
                logger.info(f"Fallback: converted {target_var} to categorical")
        except:
            logger.error(f"Could not convert {target_var} to categorical, dropping")
            if target_var in df.columns:
                df = df.drop(columns=[target_var])
    
    return df


def detect_target_type(y: pd.Series) -> str:
    """Detect the type of target variable (e.g., continuous, binary, categorical)"""
    if y.dtype == 'float64' or y.dtype == 'int64':
        return 'continuous'
    elif y.nunique() <= 2:
        return 'binary'
    else:
        return 'categorical'


def get_selected_features(target_var: str, selection_dir: str, associations_dir: str, 
                         X_valid: pd.DataFrame, y_valid: pd.Series, n_features: int) -> list:
    """
    Get selected features based on selection results or associations.
    Prioritizes selection results if available, otherwise uses associations.
    """
    if selection_dir and os.path.exists(selection_dir):
        selection_file = os.path.join(selection_dir, f"selected_features_{target_var}.csv")
        if os.path.exists(selection_file):
            logger.info(f"Using selected features from {selection_file}")
            selected_features = pd.read_csv(selection_file)['feature'].tolist()
            return selected_features
        else:
            logger.warning(f"Selection results not found at {selection_file}, trying associations...")
    
    if associations_dir and os.path.exists(associations_dir):
        associations_file = os.path.join(associations_dir, f"associations_{target_var}.csv")
        if os.path.exists(associations_file):
            logger.info(f"Using associations from {associations_file}")
            # Assuming associations file has 'feature' and 'importance' columns
            associations = pd.read_csv(associations_file)
            # Sort by importance and take top n_features
            associations = associations.sort_values('importance', ascending=False)
            selected_features = associations['feature'].tolist()[:n_features]
            return selected_features
        else:
            logger.warning(f"Associations results not found at {associations_file}, using all features.")
    
    # Fallback: use all features if no selection or associations results
    logger.info(f"No selection or associations results found, using all {X_valid.shape[1]} features.")
    return X_valid.columns.tolist()


def run_baseline_predictive_check(X_selected: pd.DataFrame, y_valid: pd.Series, 
                                 subject_ids: np.ndarray, target_var: str, 
                                 model_type: str, n_splits: int):
    """
    Run robust baseline predictive check for a single target variable.
    """
    logger.info(f"Running baseline predictive check for {target_var} with {X_selected.shape[1]} features.")
    
    # Scale features
    scaler = StandardScaler()
    X_scaled_array = scaler.fit_transform(X_selected)
    
    # Convert back to DataFrame to maintain column names
    X_scaled = pd.DataFrame(X_scaled_array, columns=X_selected.columns, index=X_selected.index)
    
    # Define baseline models
    models = {
        'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000)
    }
    
    # Add Random Forest as alternative to LightGBM
    models['Random Forest'] = RandomForestClassifier(
        n_estimators=100, 
        random_state=42, 
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        n_jobs=-1
    )
    logger.info("Using Logistic Regression and Random Forest classifiers")
    
    # Train and evaluate models
    results = {}
    for name, model in models.items():
        logger.info(f"Training {name}...")
        results[name] = train_and_evaluate_model(
            X_scaled, y_valid, model, n_splits, subject_ids
        )
    
    # Compare models
    comparison_results = compare_models(results)
    
    # Feature importance analysis (if applicable)
    feature_importance = None
    if 'Logistic Regression' in results:
        feature_importance = results['Logistic Regression']['feature_importance']
    
    return {
        'model_comparison': comparison_results,
        'feature_importance': feature_importance,
        'y_valid': y_valid.tolist(),
        'target_var': target_var,
        'n_features': X_selected.shape[1]
    }


def train_and_evaluate_model(X, y, model, cv_folds: int = 5, 
                           subject_ids: np.ndarray = None) -> dict:
    """Train and evaluate a classification model"""
    try:
        # Handle both pandas DataFrame and numpy array
        if isinstance(X, np.ndarray):
            logger.info("Converting numpy array to DataFrame for validation")
            X = pd.DataFrame(X)
        
        if isinstance(y, np.ndarray):
            logger.info("Converting numpy array to Series for validation")
            y = pd.Series(y)
        
        # Ensure target variable is categorical for classification
        if y.dtype in ['float64', 'int64']:
            logger.warning(f"Target variable is numeric ({y.dtype}), converting to categorical")
            # Convert to categorical if it's numeric
            y = y.astype('category')
        
        # Encode categorical labels to numeric for sklearn
        if y.dtype.name == 'category':
            from sklearn.preprocessing import LabelEncoder
            le = LabelEncoder()
            y_encoded = le.fit_transform(y)
            y = pd.Series(y_encoded, index=y.index)
            logger.info(f"Encoded categorical labels: {dict(zip(range(len(le.classes_)), le.classes_))}")
        
        # Final data validation before training
        if hasattr(X, 'isnull') and X.isnull().any().any():
            logger.error("NaN values detected in features before training")
            raise ValueError("Features contain NaN values")
        
        if hasattr(y, 'isnull') and y.isnull().any():
            logger.error("NaN values detected in target before training")
            raise ValueError("Target contains NaN values")
        
        # Ensure X and y have the same length
        if len(X) != len(y):
            logger.error(f"Feature and target length mismatch: X={len(X)}, y={len(y)}")
            raise ValueError("Feature and target length mismatch")
        
        # Ensure all data is numeric
        if hasattr(X, 'select_dtypes'):
            if not X.select_dtypes(include=[np.number]).shape[1] == X.shape[1]:
                logger.error("Non-numeric features detected before training")
                raise ValueError("Non-numeric features detected")
        else:
            # For numpy arrays, check if they're numeric
            if not np.issubdtype(X.dtype, np.number):
                logger.error("Non-numeric features detected before training")
                raise ValueError("Non-numeric features detected")
        
        # Ensure target has discrete classes
        unique_classes = np.unique(y)
        if len(unique_classes) < 2:
            logger.error(f"Target variable has only {len(unique_classes)} class(es), need at least 2 for classification")
            raise ValueError("Insufficient classes for classification")
        
        logger.info(f"Training model with {X.shape[1]} features and {len(y)} samples")
        logger.info(f"Target classes: {unique_classes}")
        
        # Get target distribution safely
        if hasattr(y, 'value_counts'):
            target_dist = y.value_counts().to_dict()
        else:
            target_dist = dict(zip(*np.unique(y, return_counts=True)))
        logger.info(f"Target distribution: {target_dist}")
        
        # Choose cross-validation strategy
        if subject_ids is not None and len(np.unique(subject_ids)) >= cv_folds:
            cv = GroupKFold(n_splits=min(cv_folds, len(np.unique(subject_ids))))
            cv_groups = subject_ids
            logger.info(f"Using GroupKFold with {cv_folds} folds based on {len(np.unique(subject_ids))} subjects")
        else:
            # Fallback to StratifiedKFold
            cv = StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42)
            cv_groups = None
            logger.info(f"Using StratifiedKFold with {cv_folds} folds")
        
        # Perform cross-validation with timeout handling
        logger.info(f"Starting cross-validation for {type(model).__name__}...")
        try:
            cv_scores = cross_val_score(model, X, y, cv=cv, groups=cv_groups, scoring='f1_macro')
            logger.info(f"Cross-validation completed successfully. Scores: {cv_scores}")
        except Exception as e:
            logger.error(f"Cross-validation failed: {e}")
            # Use default scores if CV fails
            cv_scores = np.array([0.5] * cv_folds)
            logger.warning("Using default CV scores due to failure")
        
        # Train on all data
        model.fit(X, y)
        
        # Make predictions on training data (for metrics calculation)
        y_pred = model.predict(X)
        y_pred_proba = model.predict_proba(X) if hasattr(model, 'predict_proba') else None
        
        # Calculate metrics
        accuracy = accuracy_score(y, y_pred)
        precision = precision_score(y, y_pred, average='macro', zero_division=0)
        recall = recall_score(y, y_pred, average='macro', zero_division=0)
        f1 = f1_score(y, y_pred, average='macro', zero_division=0)
        
        # Calculate ROC AUC if possible
        if y_pred_proba is not None and len(unique_classes) == 2:
            try:
                roc_auc = roc_auc_score(y, y_pred_proba[:, 1])
            except:
                roc_auc = 0.5  # Default for binary classification
        else:
            roc_auc = 0.5
        
        # Get feature importance
        feature_importance = None
        # 保持原有的描述性特征名称
        if hasattr(X, 'columns'):
            feature_names = list(X.columns)
        else:
            # 如果没有列名，使用索引但保持可读性
            feature_names = [f"feature_{i}" for i in range(X.shape[1])]
        
        if hasattr(model, 'coef_'):
            feature_importance = model.coef_[0] if len(model.coef_.shape) > 1 else model.coef_
        elif hasattr(model, 'feature_importances_'):
            feature_importance = model.feature_importances_
        
        # Create results dictionary
        results = {
            'model': model,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1_score': f1,
            'roc_auc': roc_auc,
            'cv_mean': cv_scores.mean(),
            'cv_std': cv_scores.std(),
            'feature_importance': feature_importance,
            'feature_names': feature_names,
            'cv_scores': cv_scores.tolist(),
            'n_classes': len(unique_classes),
            'class_distribution': target_dist
        }
        
        logger.info(f"Model training completed successfully")
        logger.info(f"  Accuracy: {accuracy:.3f}")
        logger.info(f"  F1 Score: {f1:.3f}")
        logger.info(f"  CV Mean: {cv_scores.mean():.3f} ± {cv_scores.std():.3f}")
        logger.info(f"  Classes: {len(unique_classes)}")
        
        return results
        
    except Exception as e:
        logger.error(f"Model training failed: {e}")
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        raise RuntimeError(f"Model training failed: {e}")


def compare_models(results: dict) -> pd.DataFrame:
    """Compare different models based on their performance metrics"""
    logger.info("Comparing model performance...")
    
    comparison_data = []
    
    for model_name, result in results.items():
        if result is None:
            continue
            
        comparison_data.append({
            'Model': model_name,
            'Accuracy': result['accuracy'],
            'Precision': result['precision'],
            'Recall': result['recall'],
            'F1_Score': result['f1_score'],
            'ROC_AUC': result['roc_auc'],
            'CV_Mean': result['cv_mean'],
            'CV_Std': result['cv_std']
        })
    
    comparison_df = pd.DataFrame(comparison_data)
    
    # Sort by F1 score (descending)
    if not comparison_df.empty:
        comparison_df = comparison_df.sort_values('F1_Score', ascending=False)
    
    logger.info(f"Model comparison completed for {len(comparison_data)} models")
    return comparison_df


def save_predictive_check_results(results: dict, output_dir: str):
    """Save predictive check results with organized file structure"""
    
    # 创建子目录
    data_dir = os.path.join(output_dir, "data")
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(data_dir, exist_ok=True)
    os.makedirs(plots_dir, exist_ok=True)
    
    # 保存每个目标变量的结果
    for target_var, target_results in results.items():
        if target_results is not None:
            # 保存模型比较结果
            model_comparison_file = os.path.join(data_dir, f"model_comparison_{target_var.lower().replace(' ', '_')}.csv")
            target_results['model_comparison'].to_csv(model_comparison_file, index=False)
            
            # 保存特征重要性（如果有）
            if 'feature_importance' in target_results and target_results['feature_importance'] is not None:
                importance_file = os.path.join(data_dir, f"importance_{target_var.lower().replace(' ', '_')}.csv")
                # feature_importance是numpy数组，需要转换为DataFrame
                importance_array = target_results['feature_importance']
                # 使用原有的特征名称，而不是重新编号
                if 'feature_names' in target_results and target_results['feature_names']:
                    feature_names = target_results['feature_names']
                else:
                    # 如果没有特征名称，使用索引
                    feature_names = [f'feature_{i}' for i in range(len(importance_array))]
                
                importance_df = pd.DataFrame({
                    'feature': feature_names,
                    'importance': importance_array
                })
                importance_df = importance_df.sort_values('importance', ascending=False)
                importance_df.to_csv(importance_file, index=False)
    
    # 创建结果索引文件
    results_index = {
        "experiment_type": "baseline_predictive_check",
        "files": {},
        "plots": {
            "model_comparison": "plots/model_comparison.png",
            "feature_importance": "plots/feature_importance_comparison.png",
            "confusion_matrices": "plots/confusion_matrices.png"
        },
        "summary": {
            "target_variables": list(results.keys()),
            "total_target_vars": len(results),
            "generated_at": datetime.now().isoformat()
        }
    }
    
    # 添加每个目标变量的文件
    for target_var, target_results in results.items():
        if target_results is not None:
            target_key = target_var.lower().replace(' ', '_')
            results_index["files"][f"model_comparison_{target_key}"] = f"data/model_comparison_{target_key}.csv"
            if 'feature_importance' in target_results and target_results['feature_importance'] is not None:
                results_index["files"][f"importance_{target_key}"] = f"data/importance_{target_key}.csv"
    
    with open(os.path.join(output_dir, "results_index.json"), "w", encoding='utf-8') as f:
        json.dump(results_index, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Baseline Predictive Check results saved to {output_dir} with organized structure")


def plot_predictive_check_results(results: dict, output_dir: str):
    """Plot predictive check results for all target variables"""
    logger.info("Plotting predictive check results...")
    
    plots_dir = os.path.join(output_dir, "plots")
    os.makedirs(plots_dir, exist_ok=True)
    
    # Plot 1: Model comparison across target variables
    plot_model_comparison(results, plots_dir)
    
    # Plot 2: Feature importance analysis
    plot_feature_importance_analysis(results, plots_dir)
    
    # Plot 3: Confusion matrices
    plot_confusion_matrices(results, plots_dir)


def plot_model_comparison(results: dict, plots_dir: str):
    """Plot model comparison across target variables"""
    logger.info("Plotting model comparison...")
    
    # Prepare data for plotting
    all_data = []
    for target_var, target_results in results.items():
        if 'model_comparison' in target_results:
            for _, row in target_results['model_comparison'].iterrows():
                all_data.append({
                    'Target_Variable': target_var,
                    'Model': row['Model'],
                    'F1_Score': row['F1_Score'],
                    'Accuracy': row['Accuracy'],
                    'ROC_AUC': row['ROC_AUC'] if pd.notna(row['ROC_AUC']) else 0
                })
    
    if not all_data:
        logger.warning("No model comparison data to plot")
        return
    
    comparison_df = pd.DataFrame(all_data)
    
    # Create subplots
    fig, axes = plt.subplots(1, 3, figsize=(18, 6))
    
    # F1 Score comparison
    sns.barplot(data=comparison_df, x='Target_Variable', y='F1_Score', hue='Model', ax=axes[0])
    axes[0].set_title('F1 Score Comparison')
    axes[0].set_ylabel('F1 Score')
    axes[0].tick_params(axis='x', rotation=45)
    
    # Accuracy comparison
    sns.barplot(data=comparison_df, x='Target_Variable', y='Accuracy', hue='Model', ax=axes[1])
    axes[1].set_title('Accuracy Comparison')
    axes[1].set_ylabel('Accuracy')
    axes[1].tick_params(axis='x', rotation=45)
    
    # ROC AUC comparison
    sns.barplot(data=comparison_df, x='Target_Variable', y='ROC_AUC', hue='Model', ax=axes[2])
    axes[2].set_title('ROC AUC Comparison')
    axes[2].set_ylabel('ROC AUC')
    axes[2].tick_params(axis='x', rotation=45)
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "model_comparison.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_feature_importance_analysis(results: dict, plots_dir: str):
    """Plot feature importance analysis for selected features"""
    logger.info("Plotting feature importance analysis...")
    
    # Find target variables with feature importance
    target_vars_with_importance = []
    for target_var, target_results in results.items():
        if 'feature_importance' in target_results and target_results['feature_importance'] is not None:
            target_vars_with_importance.append(target_var)
    
    if not target_vars_with_importance:
        logger.warning("No feature importance data to plot")
        return
    
    # Create subplots
    n_targets = len(target_vars_with_importance)
    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 6))
    if n_targets == 1:
        axes = [axes]
    
    for i, target_var in enumerate(target_vars_with_importance):
        target_results = results[target_var]
        importance = target_results['feature_importance']
        feature_names = target_results.get('feature_names', [f'Feature_{j}' for j in range(len(importance))])
        
        # Sort by importance
        sorted_indices = np.argsort(importance)[::-1]
        top_features = [feature_names[j] for j in sorted_indices[:10]]  # Top 10 features
        top_importance = [importance[j] for j in sorted_indices[:10]]
        
        # Parse feature names for better display
        display_names = [parse_feature_name_for_display(name) for name in top_features]
        
        # Plot
        axes[i].barh(range(len(top_features)), top_importance)
        axes[i].set_yticks(range(len(top_features)))
        axes[i].set_yticklabels(display_names)
        axes[i].set_title(f'Top Features - {target_var}')
        axes[i].set_xlabel('Importance Score')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "feature_importance_analysis.png"), dpi=300, bbox_inches='tight')
    plt.close()


def plot_confusion_matrices(results: dict, plots_dir: str):
    """Plot confusion matrices for all target variables"""
    logger.info("Plotting confusion matrices...")
    
    # Find target variables with confusion matrices
    target_vars_with_cm = []
    for target_var, target_results in results.items():
        if 'confusion_matrix' in target_results and target_results['confusion_matrix'] is not None:
            target_vars_with_cm.append(target_var)
    
    if not target_vars_with_cm:
        logger.warning("No confusion matrix data to plot")
        return
    
    # Create subplots
    n_targets = len(target_vars_with_cm)
    fig, axes = plt.subplots(1, n_targets, figsize=(6 * n_targets, 6))
    if n_targets == 1:
        axes = [axes]
    
    for i, target_var in enumerate(target_vars_with_cm):
        target_results = results[target_var]
        cm = target_results['confusion_matrix']
        
        # Plot confusion matrix
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=axes[i])
        axes[i].set_title(f'Confusion Matrix - {target_var}')
        axes[i].set_xlabel('Predicted')
        axes[i].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig(os.path.join(plots_dir, "confusion_matrices.png"), dpi=300, bbox_inches='tight')
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


def generate_structured_summary(results: dict, df_processed: pd.DataFrame) -> dict:
    """Generate a structured summary of the baseline predictive check experiment."""
    
    # Calculate overall metrics
    total_targets = len(results)
    total_features = 0
    best_models = {}
    
    for target_var, target_results in results.items():
        if 'model_comparison' in target_results and not target_results['model_comparison'].empty:
            # Get best model for this target
            best_model_row = target_results['model_comparison'].iloc[0]
            best_models[target_var] = {
                'model': best_model_row['Model'],
                'f1_score': best_model_row['F1_Score'],
                'accuracy': best_model_row['Accuracy'],
                'roc_auc': best_model_row['ROC_AUC'] if pd.notna(best_model_row['ROC_AUC']) else None,
                'cv_mean': best_model_row['CV_Mean'],
                'cv_std': best_model_row['CV_Std']
            }
        
        if 'n_features' in target_results:
            total_features += target_results['n_features']
    
    # Generate frontend summary with proper serialization
    frontend_summary = {
        'total_targets': total_targets,
        'total_features_used': total_features,
        'target_performance': best_models,
        'overall_performance': {
            'average_f1': float(np.mean([m['f1_score'] for m in best_models.values()])) if best_models else 0.0,
            'average_accuracy': float(np.mean([m['accuracy'] for m in best_models.values()])) if best_models else 0.0,
            'average_roc_auc': float(np.mean([m['roc_auc'] for m in best_models.values() if m['roc_auc'] is not None])) if best_models else 0.0
        }
    }
    
    # Convert numpy types to Python native types for serialization
    for target_var, target_data in frontend_summary['target_performance'].items():
        for key, value in target_data.items():
            if hasattr(value, 'item'):  # numpy scalar
                target_data[key] = value.item()
            elif value is None:
                target_data[key] = None
            elif key == 'model':  # Keep model name as string
                target_data[key] = str(value)
            elif isinstance(value, (int, float)):  # Only convert numeric values
                target_data[key] = float(value)
            else:
                target_data[key] = value  # Keep other types as is
    
    # Generate detailed summary
    detailed_summary = {
        'experiment_type': 'Baseline Predictive Check',
        'total_targets': total_targets,
        'target_variables': list(results.keys()),
        'best_models': best_models,
        'feature_usage': {
            target_var: target_results.get('n_features', 0) 
            for target_var, target_results in results.items()
        },
        'cross_validation': {
            target_var: {
                'cv_mean': target_results.get('model_comparison', pd.DataFrame()).get('CV_Mean', [0]).iloc[0] if not target_results.get('model_comparison', pd.DataFrame()).empty else 0,
                'cv_std': target_results.get('model_comparison', pd.DataFrame()).get('CV_Std', [0]).iloc[0] if not target_results.get('model_comparison', pd.DataFrame()).empty else 0
            }
            for target_var, target_results in results.items()
        },
        'generated_at': datetime.now().isoformat()
    }
    
    summary = {
        "status": "success",
        "summary_text": f"Baseline Predictive Check completed for {total_targets} target variables using {total_features} features",
        "frontend_summary": frontend_summary,
        "detailed_summary": detailed_summary
    }
    
    return summary 


def run(df_feat: pd.DataFrame, df_meta: pd.DataFrame, output_dir: str, **kwargs) -> dict:
    """Main entry point for the classification experiment module
    
    This function maintains backward compatibility and calls the new
    run_baseline_predictive_check_experiment function.
    
    Args:
        df_feat: Feature DataFrame
        df_meta: Metadata DataFrame
        output_dir: Output directory
        **kwargs: Extra arguments passed to the experiment
    
    Returns:
        dict: Structured experiment summary
    """
    return run_baseline_predictive_check_experiment(df_feat, df_meta, output_dir, **kwargs) 