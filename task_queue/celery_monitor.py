#!/usr/bin/env python3
import os
import sys
import time
import argparse
from typing import Any
from celery.result import AsyncResult
from task_queue.celery_app import celery_app
from task_queue.models import TaskManager

def monitor_tasks() -> None:
    """
    Monitor the status of all Celery tasks and print summary statistics and recent tasks.

    Returns:
        None
    """
    print("=" * 50)
    print("        Celery Task Monitor")
    print("=" * 50)
    task_manager = TaskManager()
    try:
        while True:
            tasks = task_manager.get_all_tasks()
            status_counts = {}
            for task in tasks:
                status = task['status']
                status_counts[status] = status_counts.get(status, 0) + 1

            print(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}")
            print("Task Status Summary:")
            for status, count in status_counts.items():
                print(f"  {status}: {count}")

            recent_tasks = tasks[-5:]
            if recent_tasks:
                print("\nRecent Tasks:")
                for task in recent_tasks:
                    print(f"  ID: {task['id']}, Type: {task['task_type']}, Status: {task['status']}")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nMonitoring stopped")

def monitor_workers() -> None:
    """
    Monitor the status of Celery workers and print active worker and task information.

    Returns:
        None
    """
    print("=" * 50)
    print("        Celery Worker Monitor")
    print("=" * 50)
    try:
        while True:
            inspect = celery_app.control.inspect()
            active_workers = inspect.active()
            registered_workers = inspect.registered()
            stats = inspect.stats()
            if active_workers:
                print(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print("Active Workers:")
                for worker_name, tasks in active_workers.items():
                    print(f"  {worker_name}: {len(tasks)} active tasks")
                    # Show up to 3 tasks for each worker
                    for task in tasks[:3]:
                        print(f"    - {task['name']} (ID: {task['id']})")
                    if len(tasks) > 3:
                        print(f"    ... {len(tasks) - 3} more tasks")
            else:
                print(f"\nTime: {time.strftime('%Y-%m-%d %H:%M:%S')}")
                print("No active workers")
            time.sleep(5)
    except KeyboardInterrupt:
        print("\nMonitoring stopped")

def get_task_result(task_id: str) -> None:
    """
    Get and print the result of a Celery task by its ID.

    Args:
        task_id (str): The ID of the Celery task.

    Returns:
        None
    """
    try:
        result = AsyncResult(task_id, app=celery_app)
        print(f"Task ID: {task_id}")
        print(f"Status: {result.state}")
        if result.ready():
            if result.successful():
                print("Result:")
                print(result.result)
            else:
                print("Error:")
                print(result.info)
        else:
            print("Task is still running...")
    except Exception as e:
        print(f"Failed to get task result: {e}")

def main() -> None:
    """
    Main entry point for the Celery monitor tool. Parses arguments and dispatches to the appropriate monitor mode.

    Returns:
        None
    """
    parser = argparse.ArgumentParser(description='Celery Monitoring Tool')
    parser.add_argument('--mode', choices=['tasks', 'workers', 'result'], 
                       default='tasks', help='Monitor mode')
    parser.add_argument('--task-id', help='Task ID (only used in result mode)')
    args = parser.parse_args()
    if args.mode == 'tasks':
        monitor_tasks()
    elif args.mode == 'workers':
        monitor_workers()
    elif args.mode == 'result':
        if not args.task_id:
            print("Error: --task-id argument is required in result mode")
            sys.exit(1)
        get_task_result(args.task_id)

if __name__ == "__main__":
    main() 