from logging_config import logger, create_new_log_file
import subprocess
import os
from datetime import datetime

# 为main.py执行创建新的日志文件
timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
create_new_log_file(f'main_{timestamp}')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# ====== 修复：清空Redis队列任务 ======
import redis

# 获取Redis配置（与celery_app.py保持一致）
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_DB = int(os.getenv('REDIS_DB', '0'))

def clear_redis_queues():
    """
    清空Celery相关的Redis队列
    """
    logger.info("正在清空Redis队列中的所有任务...")
    r = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    
    # Celery使用的Redis键名格式
    queue_names = [
        "celery",  # 默认队列
        "feature_extraction",
        "experiments", 
        "recordings",
        "default"
    ]
    
    # 清理队列 - 使用正确的键名格式
    for queue in queue_names:
        # Celery的队列键名就是队列名本身
        deleted = r.delete(queue)
        logger.info(f"队列 {queue} 已清空（删除键数: {deleted}）")
        
        # 同时清理相关的键
        # 清理未确认的任务
        unacked_key = f"{queue}.unacked"
        unacked_deleted = r.delete(unacked_key)
        if unacked_deleted > 0:
            logger.info(f"未确认任务队列 {unacked_key} 已清空（删除键数: {unacked_deleted}）")
        
        # 清理延迟任务
        delayed_key = f"{queue}.delayed"
        delayed_deleted = r.delete(delayed_key)
        if delayed_deleted > 0:
            logger.info(f"延迟任务队列 {delayed_key} 已清空（删除键数: {delayed_deleted}）")
    
    # 清理其他Celery相关的键
    celery_keys = [
        "celery",  # 默认队列
        "celery.pidbox",  # 控制命令队列
        "celeryev",  # 事件队列
    ]
    
    for key in celery_keys:
        deleted = r.delete(key)
        if deleted > 0:
            logger.info(f"Celery键 {key} 已清空（删除键数: {deleted}）")
    
    # 清理所有以celery开头的键（更彻底的方法）
    pattern = "celery*"
    keys = r.keys(pattern)
    if keys:
        deleted = r.delete(*keys)
        logger.info(f"清理了 {deleted} 个以celery开头的键")
    
    # 清理所有队列相关的键
    queue_pattern = "*queue*"
    queue_keys = r.keys(queue_pattern)
    if queue_keys:
        deleted = r.delete(*queue_keys)
        logger.info(f"清理了 {deleted} 个队列相关的键")

try:
    clear_redis_queues()
    logger.info("Redis队列清空完成。")
except Exception as e:
    logger.error(f"清空Redis队列时发生错误: {e}")
    raise

# ====== 继续执行初始化脚本 ======
module_scripts = [
    "database.init_db",
    # "database.import_harvard_demo",
    "database.import_sleep_edfx",
    "database.default_pipelines",
    "database.default_fxdefs",
    "database.default_featuresets",
    # "setup_age_correlation",
    "database.create_experiment1_featureset",
    "database.create_experiment2_featureset"
]

for module in module_scripts:
    logger.info(f"Running {module} ...")
    result = subprocess.run(["python", "-m", module], capture_output=True, text=True)
    if result.returncode == 0:
        logger.info(f"{module} completed successfully.")
        if result.stdout.strip():
            logger.info(f"Output: {result.stdout.strip()}")
    else:
        logger.error(f"Errors in {module}: {result.stderr.strip()}")
        logger.error(f"{module} failed.")
        break 
logger.info("All setup scripts executed.")
