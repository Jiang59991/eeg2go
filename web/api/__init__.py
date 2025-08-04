"""
Task Queue Module for EEG2Go

This module provides asynchronous task processing capabilities for the EEG2Go system.
It includes task management and queue operations using Celery + Redis.

Main components:
- TaskManager: Manages task creation, status updates, and retrieval
- Task: Represents individual tasks with metadata and parameters
- Celery tasks: Distributed task processing
"""

from task_queue.models import Task, TaskManager, TaskStatus

__version__ = "1.0.0"
__author__ = "EEG2Go Team"

# Export main classes for easy import
__all__ = [
    'Task',
    'TaskManager', 
    'TaskStatus'
]

# Optional: Initialize default task manager instance
def get_default_task_manager():
    """Get the default task manager instance"""
    from task_queue.models import TaskManager
    return TaskManager()
