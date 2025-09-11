from task_queue.models import Task, TaskManager, TaskStatus

__version__ = "1.0.0"
__author__ = "EEG2Go Team"

__all__ = [
    'Task',
    'TaskManager', 
    'TaskStatus'
]

def get_default_task_manager() -> TaskManager:
    """
    Get the default task manager instance.

    Returns:
        TaskManager: The default TaskManager instance.
    """
    from task_queue.models import TaskManager
    return TaskManager()
