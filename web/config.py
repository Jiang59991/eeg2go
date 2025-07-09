"""
Web应用配置文件
包含数据库路径、应用设置等配置项
"""

import os

# 基础路径配置
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(BASE_DIR)

# 数据库配置
DATABASE_PATH = os.path.join(PROJECT_ROOT, "database", "eeg2go.db")

# Flask应用配置
class Config:
    """基础配置类"""
    SECRET_KEY = os.environ.get('SECRET_KEY') or 'dev-secret-key-change-in-production'
    DEBUG = True
    HOST = '0.0.0.0'
    PORT = 5000
    
    # 静态文件配置
    STATIC_FOLDER = os.path.join(BASE_DIR, 'static')
    TEMPLATE_FOLDER = os.path.join(BASE_DIR, 'templates')
    
    # 文件上传配置
    MAX_CONTENT_LENGTH = 16 * 1024 * 1024  # 16MB
    UPLOAD_FOLDER = os.path.join(PROJECT_ROOT, 'data', 'uploads')
    
    # 日志配置
    LOG_LEVEL = 'INFO'
    LOG_FILE = os.path.join(PROJECT_ROOT, 'logs', 'web_interface.log')

class DevelopmentConfig(Config):
    """开发环境配置"""
    DEBUG = True
    LOG_LEVEL = 'DEBUG'

class ProductionConfig(Config):
    """生产环境配置"""
    DEBUG = False
    LOG_LEVEL = 'WARNING'
    HOST = '127.0.0.1'

# 配置映射
config = {
    'development': DevelopmentConfig,
    'production': ProductionConfig,
    'default': DevelopmentConfig
}

def get_config(config_name=None):
    """获取配置对象"""
    if config_name is None:
        config_name = os.environ.get('FLASK_ENV', 'default')
    return config.get(config_name, config['default']) 