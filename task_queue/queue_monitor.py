#!/usr/bin/env python3
import os
import sys
import time
import argparse
import redis
from celery import Celery
from task_queue.celery_app import celery_app

def check_redis_queues() -> None:
    """
    Check the status of Redis queues and print their lengths and a preview of tasks.
    """
    print("=" * 50)
    print("        Redis Queue Status Check")
    print("=" * 50)
    try:
        redis_host = os.getenv('REDIS_HOST', 'localhost')
        redis_port = int(os.getenv('REDIS_PORT', '6379'))
        redis_db = int(os.getenv('REDIS_DB', '0'))
        r = redis.Redis(host=redis_host, port=redis_port, db=redis_db)
        queues = ['celery', 'feature_extraction', 'experiments', 'recordings']
        print(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        print("Redis queue status:")
        for queue in queues:
            length = r.llen(queue)
            print(f"  {queue}: {length} tasks")
            if length > 0:
                tasks = r.lrange(queue, 0, 2)  # Show first 3 tasks
                print(f"    First {len(tasks)} tasks:")
                for i, task in enumerate(tasks):
                    print(f"      {i+1}. {task[:100]}...")  # Show first 100 chars
        r.close()
    except Exception as e:
        print(f"Redis connection failed: {e}")

def check_celery_workers() -> None:
    """
    Check the status of Celery workers and print their registration, activity, and stats.
    """
    print("=" * 50)
    print("        Celery Worker Status Check")
    print("=" * 50)
    try:
        inspect = celery_app.control.inspect()
        registered = inspect.registered()
        active = inspect.active()
        stats = inspect.stats()
        print(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}")
        if registered:
            print("Registered Workers:")
            for worker_name, tasks in registered.items():
                print(f"  {worker_name}")
                if tasks:
                    print(f"    Supported tasks: {', '.join(tasks)}")
        else:
            print("No registered workers")
        if active:
            print("\nActive Workers:")
            for worker_name, tasks in active.items():
                print(f"  {worker_name}: {len(tasks)} active tasks")
                for task in tasks:
                    print(f"    - {task['name']} (ID: {task['id']})")
        else:
            print("\nNo active workers")
        if stats:
            print("\nWorker Statistics:")
            for worker_name, stat in stats.items():
                print(f"  {worker_name}:")
                print(f"    Pool size: {stat.get('pool', {}).get('max-concurrency', 'N/A')}")
                print(f"    Tasks processed: {stat.get('total', {}).get('task_queue.tasks.run_feature_set_task', 0)}")
    except Exception as e:
        print(f"Celery check failed: {e}")

def monitor_queues() -> None:
    """
    Continuously monitor the status of Redis queues and Celery workers.
    """
    print("=" * 50)
    print("        Queue Status Monitoring")
    print("=" * 50)
    try:
        while True:
            check_redis_queues()
            check_celery_workers()
            print("\n" + "="*50)
            time.sleep(10)  # Check every 10 seconds
    except KeyboardInterrupt:
        print("\nMonitoring stopped")

def main() -> None:
    """
    Main function to parse arguments and run the appropriate monitoring mode.
    """
    parser = argparse.ArgumentParser(description='Queue Monitoring Tool')
    parser.add_argument('--mode', choices=['redis', 'workers', 'monitor'], 
                       default='monitor', help='Monitoring mode')
    args = parser.parse_args()
    if args.mode == 'redis':
        check_redis_queues()
    elif args.mode == 'workers':
        check_celery_workers()
    elif args.mode == 'monitor':
        monitor_queues()

if __name__ == "__main__":
    main()
