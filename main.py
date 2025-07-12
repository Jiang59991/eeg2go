from logging_config import logger
import subprocess
import os

from logging_config import create_new_log_file
create_new_log_file('init_db')

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

module_scripts = [
    "database.init_db",
    "database.import_harvard_demo",
    "database.default_pipelines",
    "database.default_fxdefs",
    "database.default_featuresets",
    "setup_age_correlation"
]

for module in module_scripts:
    logger.info(f"Running {module} ...")
    result = subprocess.run(["python", "-m", module], capture_output=True, text=True)
    if result.returncode == 0:
        logger.info(f"{module} completed successfully.")
    else:
        logger.error(f"Errors in {module}: {result.stderr.strip()}")
        logger.error(f"{module} failed.")
        break 
logger.info("All setup scripts executed.")
