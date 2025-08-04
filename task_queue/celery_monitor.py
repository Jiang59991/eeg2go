#!/usr/bin/env python3
"""
Celery监控脚本
监控任务状态和worker状态
"""
import os
import sys
import time
import argparse
from celery import Celery
from celery.result import AsyncResult
from task_queue.celery_app import celery_app
from task_queue.models import TaskManager

def monitor_tasks():
    """监控任务状态"""
    print("=" * 50)
    print("        Celery任务监控")
    print("=" * 50)
    
    task_manager = TaskManager()
    
    try:
        while True:
            # 获取所有任务
            tasks = task_manager.get_all_tasks()
            
            # 按状态分组
            status_counts = {}
            for task in tasks:
                status = task['status']
                status_counts[status] = status_counts.get(status, 0) + 1
            
            # 显示统计信息
            print(f"\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("任务状态统计:")
            for status, count in status_counts.items():
                print(f"  {status}: {count}")
            
            # 显示最近的任务
            recent_tasks = tasks[-5:]  # 最近5个任务
            if recent_tasks:
                print("\n最近的任务:")
                for task in recent_tasks:
                    print(f"  ID: {task['id']}, 类型: {task['task_type']}, 状态: {task['status']}")
            
            time.sleep(5)  # 每5秒更新一次
            
    except KeyboardInterrupt:
        print("\n监控已停止")

def monitor_workers():
    """监控worker状态"""
    print("=" * 50)
    print("        Celery Worker监控")
    print("=" * 50)
    
    try:
        while True:
            # 获取活跃的worker
            inspect = celery_app.control.inspect()
            active_workers = inspect.active()
            registered_workers = inspect.registered()
            stats = inspect.stats()
            
            if active_workers:
                print(f"\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print("活跃的Worker:")
                
                for worker_name, tasks in active_workers.items():
                    print(f"  {worker_name}: {len(tasks)} 个活跃任务")
                    
                    # 显示任务详情
                    for task in tasks[:3]:  # 只显示前3个任务
                        print(f"    - {task['name']} (ID: {task['id']})")
                    
                    if len(tasks) > 3:
                        print(f"    ... 还有 {len(tasks) - 3} 个任务")
            else:
                print(f"\n时间: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print("没有活跃的Worker")
            
            time.sleep(5)  # 每5秒更新一次
            
    except KeyboardInterrupt:
        print("\n监控已停止")

def get_task_result(task_id):
    """获取任务结果"""
    try:
        # 尝试从Celery获取结果
        result = AsyncResult(task_id, app=celery_app)
        
        print(f"任务ID: {task_id}")
        print(f"状态: {result.state}")
        
        if result.ready():
            if result.successful():
                print("结果:")
                print(result.result)
            else:
                print("错误:")
                print(result.info)
        else:
            print("任务仍在运行中...")
            
    except Exception as e:
        print(f"获取任务结果失败: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='Celery监控工具')
    parser.add_argument('--mode', choices=['tasks', 'workers', 'result'], 
                       default='tasks', help='监控模式')
    parser.add_argument('--task-id', help='任务ID (仅在result模式下使用)')
    
    args = parser.parse_args()
    
    if args.mode == 'tasks':
        monitor_tasks()
    elif args.mode == 'workers':
        monitor_workers()
    elif args.mode == 'result':
        if not args.task_id:
            print("错误: result模式需要指定--task-id参数")
            sys.exit(1)
        get_task_result(args.task_id)

if __name__ == "__main__":
    main() 