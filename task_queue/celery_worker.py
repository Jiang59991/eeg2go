#!/usr/bin/env python3
"""
Celery Worker启动脚本
支持启动不同类型的worker进程
"""
import os
import sys
import argparse
from celery.bin.celery import main as celery_main

def start_worker(queue_name='default', concurrency=1, hostname=None):
    """启动Celery Worker"""
    
    # 设置环境变量
    os.environ.setdefault('CELERY_CONFIG_MODULE', 'task_queue.celery_app')
    
    # 构建celery worker命令参数
    cmd = [
        'celery', 'worker',
        '--app=task_queue.celery_app:celery_app',
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
    
    print(f"启动Celery Worker: {' '.join(cmd)}")
    
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
    valid_queues = ['default', 'feature_extraction', 'experiments']
    if args.queue not in valid_queues:
        print(f"错误: 无效的队列名称 '{args.queue}'")
        print(f"有效的队列名称: {', '.join(valid_queues)}")
        sys.exit(1)
    
    try:
        start_worker(
            queue_name=args.queue,
            concurrency=args.concurrency,
            hostname=args.hostname
        )
    except KeyboardInterrupt:
        print("\nWorker已停止")
    except Exception as e:
        print(f"启动Worker失败: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 