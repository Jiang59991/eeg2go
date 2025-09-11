import os
from typing import Type, Optional

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)
DATABASE_PATH = os.path.join(PROJECT_ROOT, "database", "eeg2go.db")

class Config:
    """Base configuration class for Flask app."""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000
    STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
    TEMPLATE_FOLDER = os.path.join(BASE_DIR, 'templates')
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'uploads')
    LOG_LEVEL = 'INFO'
    LOG_FILE = os.path.join(PROJECT_ROOT, 'logs', 'web_interface.log')

class DevelopmentConfig(Config):
    """Configuration for development environment."""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """Configuration for production environment."""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    HOST = '127.0.0.1'

config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config(config_name: Optional[str] = None) -> Type[Config]:
    """
    Get the configuration class according to the config_name.
    If config_name is None, use the FLASK_ENV environment variable or 'default'.
    """
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    return config.get(config_name, config['default'])