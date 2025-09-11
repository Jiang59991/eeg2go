#!/usr/bin/env python3
import os
import sys
import subprocess
from logging_config import logger

def start_recordings_worker() -> None:
    """
    Start a Celery worker dedicated to the 'recordings' queue.

    This function sets up the environment and launches a Celery worker process
    with specific configuration for handling run_feature_set tasks for individual recordings.
    """
    try:
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        os.environ['PYTHONPATH'] = project_root

        queue_name = 'recordings'
        concurrency = 4  # Higher concurrency for large single recording tasks
        hostname = f'recordings_worker@{os.uname().nodename}'

        # Build the Celery worker command
        cmd = [
            'celery',
            '--app=task_queue.celery_app:celery_app',
            'worker',
            '--loglevel=INFO',
            f'--concurrency={concurrency}',
            '--pool=prefork',
            f'--queues={queue_name}',
            f'--hostname={hostname}',
            '--without-gossip',
            '--without-mingle',
            '--without-heartbeat'
        ]

        logger.info(f"Starting Celery Worker: {' '.join(cmd)}")

        process = subprocess.Popen(cmd, cwd=project_root)

        logger.info(f"Recordings Worker started with PID: {process.pid}")
        logger.info(f"Worker 3 (recordings) started, PID: {process.pid}")

        # Wait for the worker process to finish
        process.wait()

    except KeyboardInterrupt:
        logger.info("Recordings Worker stopped by user")
    except Exception as e:
        logger.error(f"Failed to start recordings worker: {e}")
        raise

if __name__ == "__main__":
    start_recordings_worker() 