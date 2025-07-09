import logging
import os
from datetime import datetime

# 日志目录和文件名
LOG_DIR = os.path.join(os.path.dirname(__file__), 'logs')
os.makedirs(LOG_DIR, exist_ok=True)

# 自动生成带时间戳的日志文件名
time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
LOG_FILE = os.path.join(LOG_DIR, f'eeg2go_{time_str}.log')

# 日志格式
LOG_FORMAT = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

# 创建logger
logger = logging.getLogger('eeg2go')
logger.setLevel(logging.INFO)

# 文件Handler
file_handler = logging.FileHandler(LOG_FILE, encoding='utf-8')
file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
logger.addHandler(file_handler)

# 控制台Handler
console_handler = logging.StreamHandler()
console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
logger.addHandler(console_handler)

# 可选：抑制mne和其他第三方库的冗余输出
try:
    import mne
    mne.set_log_level('WARNING')
except ImportError:
    pass

# 用法示例：
# from logging_config import logger
# logger.info('任务开始')
# logger.error('发生错误')
