"""
Task Queue Module for EEG2Go

This module provides asynchronous task processing capabilities for the EEG2Go system.
It includes task management, worker processes, and queue operations.

Main components:
- TaskManager: Manages task creation, status updates, and retrieval
- TaskWorker: Processes tasks in background threads
- Task: Represents individual tasks with metadata and parameters
"""

from task_queue.models import Task, TaskManager, TaskStatus
from task_queue.task_worker import TaskWorker

__version__ = "1.0.0"
__author__ = "EEG2Go Team"

# Export main classes for easy import
__all__ = [
    'Task',
    'TaskManager', 
    'TaskWorker',
    'TaskStatus'
]

# Optional: Initialize default task manager instance
def get_default_task_manager():
    """Get the default task manager instance"""
    from .models import TaskManager
    return TaskManager()

# Optional: Initialize default task worker instance  
def get_default_task_worker():
    """Get the default task worker instance"""
    from .task_worker import TaskWorker
    from .models import TaskManager
    task_manager = TaskManager()
    return TaskWorker(task_manager)
