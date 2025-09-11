#!/usr/bin/env python3
import os
import sys
import argparse
from typing import Optional
from celery.bin.celery import main as celery_main

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logging_config import logger
import task_queue.tasks

def start_worker(queue_name: str = 'default', concurrency: int = 1, hostname: Optional[str] = None) -> None:
    """
    Start a Celery Worker process with the specified queue, concurrency, and hostname.

    Args:
        queue_name (str): The name of the queue to listen to.
        concurrency (int): Number of worker processes.
        hostname (Optional[str]): Custom worker hostname.

    Returns:
        None
    """
    os.environ.setdefault('CELERY_CONFIG_MODULE', 'task_queue.celery_app')
    cmd = [
        'celery', '--app=task_queue.celery_app:celery_app', 'worker',
        '--loglevel=INFO',
        f'--concurrency={concurrency}',
        '--pool=prefork',  # Use prefork pool for CPU-bound tasks
    ]
    if queue_name != 'default':
        cmd.extend(['--queues', queue_name])
    if hostname:
        cmd.extend(['--hostname', hostname])
    cmd.extend([
        '--without-gossip',
        '--without-mingle',
        '--without-heartbeat',
    ])
    logger.info(f"Starting Celery Worker: {' '.join(cmd)}")
    sys.argv = cmd
    celery_main()

def main() -> None:
    """
    Main entry point for starting a Celery Worker via command line arguments.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='Start Celery Worker')
    parser.add_argument('--queue', default='default',
                       help='Queue name (default, feature_extraction, experiments)')
    parser.add_argument('--concurrency', type=int, default=1,
                       help='Number of worker processes')
    parser.add_argument('--hostname',
                       help='Worker hostname (optional)')
    args = parser.parse_args()
    valid_queues = ['default', 'feature_extraction', 'experiments', 'recordings']
    if args.queue not in valid_queues:
        logger.error(f"Error: Invalid queue name '{args.queue}'")
        logger.error(f"Valid queue names: {', '.join(valid_queues)}")
        sys.exit(1)
    try:
        start_worker(
            queue_name=args.queue,
            concurrency=args.concurrency,
            hostname=args.hostname
        )
    except KeyboardInterrupt:
        logger.info("Worker stopped")
    except Exception as e:
        logger.error(f"Failed to start Worker: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()