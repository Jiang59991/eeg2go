#!/usr/bin/env python3
"""
Celery Worker启动脚本
支持启动不同类型的worker进程
"""
import os
import sys
import argparse
from celery.bin.celery import main as celery_main

# 添加项目根目录到Python路径
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logging_config import logger

# 确保任务模块被导入
import task_queue.tasks

def start_worker(queue_name='default', concurrency=1, hostname=None):
    """启动Celery Worker"""
    
    # 设置环境变量
    os.environ.setdefault('CELERY_CONFIG_MODULE', 'task_queue.celery_app')
    
    # 构建celery worker命令参数
    cmd = [
        'celery', '--app=task_queue.celery_app:celery_app', 'worker',
        '--loglevel=INFO',
        f'--concurrency={concurrency}',
        '--pool=prefork',  # 使用prefork池，适合CPU密集型任务
    ]
    
    # 添加队列参数
    if queue_name != 'default':
        cmd.extend(['--queues', queue_name])
    
    # 添加hostname参数
    if hostname:
        cmd.extend(['--hostname', hostname])
    
    # 添加其他有用的参数
    cmd.extend([
        '--without-gossip',  # 禁用gossip协议，减少网络开销
        '--without-mingle',   # 禁用mingle，减少启动时间
        '--without-heartbeat', # 禁用心跳，减少网络开销
    ])
    
    logger.info(f"Starting Celery Worker: {' '.join(cmd)}")
    
    # 执行celery worker命令
    sys.argv = cmd
    celery_main()

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='启动Celery Worker')
    parser.add_argument('--queue', default='default', 
                       help='队列名称 (default, feature_extraction, experiments)')
    parser.add_argument('--concurrency', type=int, default=1,
                       help='并发worker数量')
    parser.add_argument('--hostname', 
                       help='Worker主机名 (可选)')
    
    args = parser.parse_args()
    
    # 验证队列名称
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