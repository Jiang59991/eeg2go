from logging_config import logger, create_new_log_file
import subprocess
import os
from datetime import datetime
import redis

timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
create_new_log_file(f'main_{timestamp}')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = int(os.getenv('REDIS_PORT', '6379'))
REDIS_DB = int(os.getenv('REDIS_DB', '0'))

def clear_redis_queues() -> None:
    """
    Clear all Celery-related Redis queues and keys.

    Returns:
        None
    """
    logger.info("Clearing all tasks from Redis queues...")
    r = redis.StrictRedis(host=REDIS_HOST, port=REDIS_PORT, db=REDIS_DB)
    queue_names = [
        "celery",
        "feature_extraction",
        "experiments", 
        "recordings",
        "default"
    ]
    for queue in queue_names:
        deleted = r.delete(queue)
        logger.info(f"Queue {queue} cleared (deleted keys: {deleted})")
        # Clear unacked tasks
        unacked_key = f"{queue}.unacked"
        unacked_deleted = r.delete(unacked_key)
        if unacked_deleted > 0:
            logger.info(f"Unacked queue {unacked_key} cleared (deleted keys: {unacked_deleted})")
        # Clear delayed tasks
        delayed_key = f"{queue}.delayed"
        delayed_deleted = r.delete(delayed_key)
        if delayed_deleted > 0:
            logger.info(f"Delayed queue {delayed_key} cleared (deleted keys: {delayed_deleted})")
    celery_keys = [
        "celery",
        "celery.pidbox",
        "celeryev",
    ]
    for key in celery_keys:
        deleted = r.delete(key)
        if deleted > 0:
            logger.info(f"Celery key {key} cleared (deleted keys: {deleted})")
    # Delete all keys starting with 'celery'
    pattern = "celery*"
    keys = r.keys(pattern)
    if keys:
        deleted = r.delete(*keys)
        logger.info(f"Deleted {deleted} keys starting with 'celery'")
    # Delete all keys related to queues
    queue_pattern = "*queue*"
    queue_keys = r.keys(queue_pattern)
    if queue_keys:
        deleted = r.delete(*queue_keys)
        logger.info(f"Deleted {deleted} queue-related keys")

try:
    clear_redis_queues()
    logger.info("Redis queues cleared.")
except Exception as e:
    logger.error(f"Error clearing Redis queues: {e}")
    raise

module_scripts = [
    "database.init_db",
    # "database.import_harvard_demo",
    "database.import_sleep_edfx",
    "database.import_tuab_data",
    "database.default_pipelines",
    "database.default_fxdefs",
    "database.default_featuresets",
    # "setup_age_correlation",
    "database.create_experiment1_featureset",
    "database.create_experiment2_featureset"
]

def run_module_scripts(scripts: list[str]) -> None:
    """
    Run a list of Python modules as subprocesses.

    Args:
        scripts (list[str]): List of module names to run.

    Returns:
        None
    """
    for module in scripts:
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

run_module_scripts(module_scripts)
