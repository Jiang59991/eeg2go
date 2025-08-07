#!/usr/bin/env python3
"""
Celery Worker for recordings queue
专门处理单个recording的run_feature_set任务
"""
import os
import sys
import subprocess
from logging_config import logger

def start_recordings_worker():
    """启动recordings队列的Worker"""
    try:
        # 获取当前目录
        current_dir = os.path.dirname(os.path.abspath(__file__))
        project_root = os.path.dirname(current_dir)
        
        # 设置环境变量
        os.environ['PYTHONPATH'] = project_root
        
        # Worker配置
        queue_name = 'recordings'
        concurrency = 4  # 可以设置更高的并发数，因为单个recording任务相对较大
        hostname = f'recordings_worker@{os.uname().nodename}'
        
        # 构建命令
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
        
        # 启动Worker
        process = subprocess.Popen(cmd, cwd=project_root)
        
        logger.info(f"Recordings Worker started with PID: {process.pid}")
        logger.info(f"Worker 3 (recordings) started, PID: {process.pid}")
        
        # 等待进程结束
        process.wait()
        
    except KeyboardInterrupt:
        logger.info("Recordings Worker stopped by user")
    except Exception as e:
        logger.error(f"Failed to start recordings worker: {e}")
        raise

if __name__ == "__main__":
    start_recordings_worker() 