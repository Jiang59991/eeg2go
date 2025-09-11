import logging
import os
from datetime import datetime

LOG_DIR = '/rds/general/user/zj724/home/eeg2go/logs'
os.makedirs(LOG_DIR, exist_ok=True)

forced_log_file = os.environ.get('EEG2GO_LOG_FILE')
if forced_log_file:
    DEFAULT_LOG_FILE = forced_log_file
else:
    DEFAULT_LOG_FILE = os.path.join(LOG_DIR, 'eeg2go_default.log')

LOG_FORMAT = '[%(asctime)s] [%(levelname)s] [%(name)s] %(message)s'
DATE_FORMAT = '%Y-%m-%d %H:%M:%S'

logger = logging.getLogger('eeg2go')
logger.setLevel(logging.INFO)

DISABLE_FILE_LOG = os.environ.get('EEG2GO_NO_FILE_LOG', '').lower() in ('1', 'true', 'yes')

# Avoid duplicate handlers
if not logger.handlers:
    if not DISABLE_FILE_LOG:
        file_handler = logging.FileHandler(DEFAULT_LOG_FILE, encoding='utf-8')
        file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
        logger.addHandler(file_handler)

    console_handler = logging.StreamHandler()
    console_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(console_handler)

try:
    import mne
    mne.utils.set_log_level('WARNING')
except ImportError:
    pass

def create_new_log_file(suffix: str = None) -> str:
    """
    Create a new log file and switch logger output to it.

    Args:
        suffix (str, optional): Suffix for the log file name. If not provided, use timestamp.

    Returns:
        str: The path to the new log file.
    """
    if suffix is None:
        time_str = datetime.now().strftime('%Y%m%d_%H%M%S')
        suffix = time_str

    new_log_file = os.path.join(LOG_DIR, f'eeg2go_{suffix}.log')

    # Remove all existing file handlers
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    new_file_handler = logging.FileHandler(new_log_file, encoding='utf-8')
    new_file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(new_file_handler)

    logger.info(f"Log output switched to new file: {new_log_file}")
    return new_log_file

def reset_to_default_log() -> None:
    """
    Reset logger output to the default log file.
    """
    for handler in logger.handlers[:]:
        if isinstance(handler, logging.FileHandler):
            logger.removeHandler(handler)
            handler.close()

    file_handler = logging.FileHandler(DEFAULT_LOG_FILE, encoding='utf-8')
    file_handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(file_handler)

    logger.info(f"Log output reset to default file: {DEFAULT_LOG_FILE}")
