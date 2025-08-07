import logging
import os
from datetime import datetime

# 日志目录和文件名
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# 默认使用固定的日志文件名
DEFAULT_LOG_FILE = os.path.join(LOG_DIR, 'eeg2go_default.log')

# 日志格式
LOG_FORMAT = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# 创建logger
logger = logging.getLogger('eeg2go')
logger.setLevel(logging.INFO)

# 文件Handler - 使用固定文件名
file_handler = logging.FileHandler(DEFAULT_LOG_FILE, encoding='utf-8')
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
logger.addHandler(file_handler)

# 控制台Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
logger.addHandler(console_handler)

# 可选：抑制mne和其他第三方库的冗余输出
try:
    import mne
    mne.utils.set_log_level('WARNING')
except ImportError:
    pass

def create_new_log_file(suffix=None):
    """
    创建新的日志文件
    
    Args:
        suffix (str, optional): 文件名后缀，如果不提供则使用时间戳
    
    Returns:
        str: 新日志文件的路径
    """
    if suffix is None:
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix = time_str
    
    new_log_file = os.path.join(LOG_DIR, f'eeg2go_{suffix}.log')
    
    # 移除所有现有的文件handler
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()
    
    # 添加新的文件handler
    new_file_handler = logging.FileHandler(new_log_file, encoding='utf-8')
    new_file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(new_file_handler)
    
    logger.info(f"日志输出已切换到新文件: {new_log_file}")
    return new_log_file

def reset_to_default_log():
    """
    重置回默认的日志文件
    """
    # 移除所有文件handler
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()
    
    # 重新添加默认文件handler
    file_handler = logging.FileHandler(DEFAULT_LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(file_handler)
    
    logger.info(f"日志输出已重置到默认文件: {DEFAULT_LOG_FILE}")

# 用法示例：
# from logging_config import logger
# logger.info('任务开始')
# logger.error('发生错误')
# 
# # 创建新的日志文件
# from logging_config import create_new_log_file
# create_new_log_file('experiment_001')
# 
# # 重置回默认日志文件
# from logging_config import reset_to_default_log
# reset_to_default_log()
