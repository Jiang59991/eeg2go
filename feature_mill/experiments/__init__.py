"""
Experiment modules for EEG2Go

This package contains various experiment modules for analyzing EEG features.
Each experiment module should implement a `run()` function with the signature:
    run(df_feat: pd.DataFrame, df_meta: pd.DataFrame, output_dir: str, **kwargs) -> str
"""

from . import correlation
from . import feature_statistics
from . import feature_selection
from . import classification

# Available experiments
AVAILABLE_EXPERIMENTS = {
    'correlation': {
        'name': 'Correlation Analysis',
        'description': 'Analyze correlation between EEG features and subject metadata (age, sex, etc.)',
        'module': 'feature_mill.experiments.correlation',
        'function': 'run',
        'default_params': {
            'target_vars': ['age', 'sex'],
            'method': 'pearson',
            'min_corr': 0.3,
            'top_n': 20
        }
    },
    'feature_statistics': {
        'name': 'Feature Statistics Analysis',
        'description': 'Comprehensive statistical analysis of EEG features including distributions, outliers, and quality assessment',
        'module': 'feature_mill.experiments.feature_statistics',
        'function': 'run',
        'default_params': {
            'outlier_method': 'iqr',
            'outlier_threshold': 1.5,
            'top_n_features': 20
        }
    },
    'feature_selection': {
        'name': 'Feature Selection',
        'description': 'Select most important features using multiple methods (variance, correlation, mutual info, Lasso, etc.)',
        'module': 'feature_mill.experiments.feature_selection',
        'function': 'run',
        'default_params': {
            'target_var': 'age',
            'n_features': 20,
            'variance_threshold': 0.01,
            'correlation_threshold': 0.95
        }
    },
    'classification': {
        'name': 'Classification Analysis',
        'description': 'Perform classification tasks using EEG features (age groups, sex, etc.)',
        'module': 'feature_mill.experiments.classification',
        'function': 'run',
        'default_params': {
            'target_var': 'age_group',
            'age_threshold': 65,
            'test_size': 0.2,
            'n_splits': 5
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

def get_experiment_default_params(experiment_name: str) -> dict:
    """Get default parameters for a specific experiment"""
    if experiment_name not in AVAILABLE_EXPERIMENTS:
        return {}
    
    return AVAILABLE_EXPERIMENTS[experiment_name].get('default_params', {}) 