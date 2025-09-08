"""
Experiment modules for EEG2Go

This package contains various experiment modules for analyzing EEG features.
Each experiment module should implement a `run()` function with the signature:
    run(df_feat: pd.DataFrame, df_meta: pd.DataFrame, output_dir: str, **kwargs) -> str
"""

from . import correlation
from . import feature_statistics
from . import classification

# Available experiments with parameter definitions
AVAILABLE_EXPERIMENTS = {
    'correlation': {
        'name': 'Correlation Analysis',
        'description': 'Analyze correlation between EEG features and subject metadata (age, sex, etc.)',
        'module': 'feature_mill.experiments.correlation',
        'function': 'run',
        'parameters': {
            'target_vars': {
                'type': 'multi_select',
                'label': 'Target Variables',
                'description': 'Select target variables for correlation analysis',
                'options': ['age', 'sex', 'age_days', 'race', 'ethnicity', 'visit_count', 'icd10_count', 'medication_count', 'seizure', 'spindles', 'status', 'normal', 'abnormal'],
                'default': ['age', 'sex'],
                'required': True
            },
            'method': {
                'type': 'select',
                'label': 'Correlation Method',
                'description': 'Statistical method for correlation analysis',
                'options': ['pearson', 'spearman', 'kendall'],
                'default': 'pearson',
                'required': True
            },
            'min_corr': {
                'type': 'number',
                'label': 'Minimum Correlation',
                'description': 'Minimum correlation coefficient to include in results',
                'min': 0.0,
                'max': 1.0,
                'step': 0.1,
                'default': 0.3,
                'required': False
            },
            'top_n': {
                'type': 'number',
                'label': 'Top N Features',
                'description': 'Number of top correlated features to display',
                'min': 1,
                'max': 100,
                'step': 1,
                'default': 20,
                'required': False
            },
            'generate_plots': {
                'type': 'checkbox',
                'label': 'Generate Plots',
                'description': 'Generate correlation matrix and scatter plot visualizations',
                'default': True,
                'required': False
            }
        }
    },
    'feature_statistics': {
        'name': 'Feature Statistics Analysis',
        'description': 'Comprehensive statistical analysis of EEG features including distributions, outliers, and quality assessment',
        'module': 'feature_mill.experiments.feature_statistics',
        'function': 'run',
        'parameters': {
            'outlier_method': {
                'type': 'select',
                'label': 'Outlier Detection Method',
                'description': 'Method for detecting outliers in features',
                'options': ['iqr', 'zscore', 'isolation_forest'],
                'default': 'iqr',
                'required': False
            },
            'outlier_threshold': {
                'type': 'number',
                'label': 'Outlier Threshold',
                'description': 'Threshold for outlier detection',
                'min': 0.5,
                'max': 5.0,
                'step': 0.1,
                'default': 1.5,
                'required': False
            },
            'top_n_features': {
                'type': 'number',
                'label': 'Top N Features',
                'description': 'Number of top features to analyze',
                'min': 1,
                'max': 100,
                'step': 1,
                'default': 20,
                'required': False
            },
            'generate_plots': {
                'type': 'checkbox',
                'label': 'Generate Plots',
                'description': 'Generate statistical plots and visualizations',
                'default': True,
                'required': False
            }
        }
    },
    'classification': {
        'name': 'Classification Analysis',
        'description': 'Perform classification tasks using EEG features (age groups, sex, etc.)',
        'module': 'feature_mill.experiments.classification',
        'function': 'run',
        'parameters': {
            'target_var': {
                'type': 'select',
                'label': 'Target Variable',
                'description': 'Target variable for classification',
                'options': ['age_group', 'sex', 'age_class'],
                'default': 'age_group',
                'required': True
            },
            'test_size': {
                'type': 'number',
                'label': 'Test Set Size',
                'description': 'Proportion of data for testing',
                'min': 0.1,
                'max': 0.5,
                'step': 0.05,
                'default': 0.2,
                'required': False
            },
            'n_splits': {
                'type': 'number',
                'label': 'Cross-validation Folds',
                'description': 'Number of folds for cross-validation',
                'min': 2,
                'max': 10,
                'step': 1,
                'default': 5,
                'required': False
            },
            'generate_plots': {
                'type': 'checkbox',
                'label': 'Generate Plots',
                'description': 'Generate classification result plots and feature importance',
                'default': True,
                'required': False
            }
        }
    }
}

def get_experiment_info(experiment_name: str) -> dict:
    """Get information about a specific experiment"""
    if experiment_name not in AVAILABLE_EXPERIMENTS:
        raise ValueError(f"Experiment '{experiment_name}' not found")
    
    return AVAILABLE_EXPERIMENTS[experiment_name]

def list_experiments() -> list:
    """List all available experiments"""
    return list(AVAILABLE_EXPERIMENTS.keys())

def get_experiment_description(experiment_name: str) -> str:
    """Get description of a specific experiment"""
    if experiment_name not in AVAILABLE_EXPERIMENTS:
        return f"Experiment '{experiment_name}' not found"
    
    return AVAILABLE_EXPERIMENTS[experiment_name]['description']

def get_experiment_parameters(experiment_name: str) -> dict:
    """Get parameter definitions for a specific experiment"""
    if experiment_name not in AVAILABLE_EXPERIMENTS:
        return {}
    
    return AVAILABLE_EXPERIMENTS[experiment_name].get('parameters', {})

def get_experiment_default_params(experiment_name: str) -> dict:
    """Get default parameters for a specific experiment"""
    if experiment_name not in AVAILABLE_EXPERIMENTS:
        return {}
    
    params = AVAILABLE_EXPERIMENTS[experiment_name].get('parameters', {})
    defaults = {}
    for param_name, param_config in params.items():
        if 'default' in param_config:
            defaults[param_name] = param_config['default']
    
    return defaults 