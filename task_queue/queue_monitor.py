#!/usr/bin/env python3
"""
队列监控脚本
检查Redis队列状态和Celery worker状态
"""
import os
import sys
import time
import argparse
import redis
from celery import Celery
from task_queue.celery_app import celery_app

def check_redis_queues():
    """检查Redis队列状态"""
    print("=" * 50)
    print("        Redis队列状态检查")
    print("=" * 50)
    
    try:
        # 连接到Redis
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', '6379'))
        redis_db = int(os.getenv('REDIS_DB', '0'))
        
        r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        
        # 检查队列长度
        queues = ['celery', 'feature_extraction', 'experiments', 'recordings']
        
        print(f"\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("Redis队列状态:")
        
        for queue in queues:
            length = r.llen(queue)
            print(f"  {queue}: {length} 个任务")
            
            # 显示队列中的前几个任务
            if length > 0:
                tasks = r.lrange(queue, 0, 2)  # 显示前3个任务
                print(f"    前{len(tasks)}个任务:")
                for i, task in enumerate(tasks):
                    print(f"      {i+1}. {task[:100]}...")  # 只显示前100个字符
        
        r.close()
        
    except Exception as e:
        print(f"Redis连接失败: {e}")

def check_celery_workers():
    """检查Celery worker状态"""
    print("=" * 50)
    print("        Celery Worker状态检查")
    print("=" * 50)
    
    try:
        inspect = celery_app.control.inspect()
        
        # 检查注册的worker
        registered = inspect.registered()
        active = inspect.active()
        stats = inspect.stats()
        
        print(f"\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        
        if registered:
            print("注册的Worker:")
            for worker_name, tasks in registered.items():
                print(f"  {worker_name}")
                if tasks:
                    print(f"    支持的任务: {', '.join(tasks)}")
        else:
            print("没有注册的Worker")
        
        if active:
            print("\n活跃的Worker:")
            for worker_name, tasks in active.items():
                print(f"  {worker_name}: {len(tasks)} 个活跃任务")
                for task in tasks:
                    print(f"    - {task['name']} (ID: {task['id']})")
        else:
            print("\n没有活跃的Worker")
        
        if stats:
            print("\nWorker统计信息:")
            for worker_name, stat in stats.items():
                print(f"  {worker_name}:")
                print(f"    池大小: {stat.get('pool', {}).get('max-concurrency', 'N/A')}")
                print(f"    处理的任务数: {stat.get('total', {}).get('task_queue.tasks.run_feature_set_task', 0)}")
        
    except Exception as e:
        print(f"Celery检查失败: {e}")

def monitor_queues():
    """持续监控队列状态"""
    print("=" * 50)
    print("        队列状态持续监控")
    print("=" * 50)
    
    try:
        while True:
            check_redis_queues()
            check_celery_workers()
            print("\n" + "="*50)
            time.sleep(10)  # 每10秒检查一次
            
    except KeyboardInterrupt:
        print("\n监控已停止")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='队列监控工具')
    parser.add_argument('--mode', choices=['redis', 'workers', 'monitor'], 
                       default='monitor', help='监控模式')
    
    args = parser.parse_args()
    
    if args.mode == 'redis':
        check_redis_queues()
    elif args.mode == 'workers':
        check_celery_workers()
    elif args.mode == 'monitor':
        monitor_queues()

if __name__ == "__main__":
    main()
