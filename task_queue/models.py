import json
import sqlite3
from datetime import datetime
from typing import Dict, Any, Optional

# Import shared enum
from .common import TaskStatus

class Task:
    def __init__(self, task_type: str, parameters: Dict[str, Any], 
                 dataset_id: Optional[int] = None, feature_set_id: Optional[int] = None,
                 experiment_type: Optional[str] = None, priority: int = 0):
        self.task_type = task_type
        self.parameters = parameters
        self.dataset_id = dataset_id
        self.feature_set_id = feature_set_id
        self.experiment_type = experiment_type
        self.priority = priority
        self.status = TaskStatus.PENDING
        self.result = None
        self.error_message = None
        self.created_at = datetime.now()
        self.started_at = None
        self.completed_at = None
        self.progress = 0.0
        self.processed_count = 0
        self.total_count = 0

class TaskManager:
    def __init__(self, db_path: str = "database/eeg2go.db"):
        self.db_path = db_path
    
    def create_task(self, task: Task) -> int:
        """Create a new task and return the task ID"""
        try:
            from logging_config import logger
            logger.info(f"TaskManager.create_task called, task_type={task.task_type}")
            
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Ensure correct data types
            dataset_id = int(task.dataset_id) if task.dataset_id is not None else None
            feature_set_id = int(task.feature_set_id) if task.feature_set_id is not None else None
            
            logger.info(f"Creating task: dataset_id={dataset_id}, feature_set_id={feature_set_id}")
            
            c.execute("""
                INSERT INTO tasks (task_type, status, parameters, dataset_id, feature_set_id, 
                                  experiment_type, priority, created_at, progress, processed_count, total_count)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                task.task_type,
                task.status.value,
                json.dumps(task.parameters),
                dataset_id,
                feature_set_id,
                task.experiment_type,
                task.priority,
                task.created_at,
                task.progress,
                task.processed_count,
                task.total_count
            ))
            
            task_id = c.lastrowid
            conn.commit()
            conn.close()
            
            logger.info(f"Task created successfully, ID: {task_id}")
            
            # Schedule Celery task
            logger.info(f"Start scheduling Celery task, task_id={task_id}")
            self._schedule_celery_task(task_id, task)
            
            return task_id
            
        except Exception as e:
            logger.error(f"Failed to create task: {e}")
            import traceback
            logger.error(f"Exception details: {traceback.format_exc()}")
            if 'conn' in locals():
                conn.close()
            raise e
    
    def _schedule_celery_task(self, task_id: int, task: Task):
        """Schedule a Celery task"""
        try:
            from logging_config import logger
            logger.info(f"_schedule_celery_task called, task_id={task_id}, task_type={task.task_type}")
            
            # Delayed import to avoid circular import
            from .tasks import feature_extraction_task, experiment_task
            
            if task.task_type == 'feature_extraction':
                if not task.dataset_id or not task.feature_set_id:
                    raise ValueError("Missing required parameters: dataset_id and feature_set_id")
                
                logger.info(f"Scheduling feature extraction task, task_id={task_id}")
                
                # Asynchronously execute feature extraction task
                celery_task = feature_extraction_task.delay(
                    task_id=task_id,
                    parameters=task.parameters,
                    dataset_id=task.dataset_id,
                    feature_set_id=task.feature_set_id
                )
                
                logger.info(f"Celery task scheduled successfully: {celery_task.id}")
                
                # Save Celery task ID to database (optional)
                self._save_celery_task_id(task_id, celery_task.id)
                
            elif task.task_type == 'experiment':
                if not task.dataset_id or not task.feature_set_id or not task.experiment_type:
                    raise ValueError("Missing required parameters: dataset_id, feature_set_id, and experiment_type")
                
                logger.info(f"Scheduling experiment task, task_id={task_id}")
                
                # Asynchronously execute experiment task
                celery_task = experiment_task.delay(
                    task_id=task_id,
                    parameters=task.parameters,
                    dataset_id=task.dataset_id,
                    feature_set_id=task.feature_set_id,
                    experiment_type=task.experiment_type
                )
                
                logger.info(f"Celery task scheduled successfully: {celery_task.id}")
                
                # Save Celery task ID to database (optional)
                self._save_celery_task_id(task_id, celery_task.id)
                
            else:
                raise ValueError(f"Unknown task type: {task.task_type}")
                
        except Exception as e:
            logger.error(f"Failed to schedule Celery task: {e}")
            import traceback
            logger.error(f"Exception details: {traceback.format_exc()}")
            # If scheduling fails, update task status to FAILED
            self.update_task_status(task_id, TaskStatus.FAILED, error_message=str(e))
            raise
    
    def _save_celery_task_id(self, task_id: int, celery_task_id: str):
        """Save Celery task ID to database (optional feature)"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # Check if celery_task_id column exists, add if not
            c.execute("PRAGMA table_info(tasks)")
            columns = [column[1] for column in c.fetchall()]
            
            if 'celery_task_id' not in columns:
                c.execute("ALTER TABLE tasks ADD COLUMN celery_task_id TEXT")
            
            c.execute("UPDATE tasks SET celery_task_id = ? WHERE id = ?", (celery_task_id, task_id))
            conn.commit()
            conn.close()
            
        except Exception as e:
            print(f"Error saving Celery task ID: {e}")
            # This error should not affect main functionality, so just log and do not raise

    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        """Get task information"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("SELECT * FROM tasks WHERE id = ?", (task_id,))
        row = c.fetchone()
        conn.close()
        
        if row:
            return {
                'id': row[0],
                'task_type': row[1],
                'status': row[2],
                'parameters': json.loads(row[3]) if row[3] else None,
                'result': json.loads(row[4]) if row[4] else None,
                'error_message': row[5],
                'created_at': row[6],
                'started_at': row[7],
                'completed_at': row[8],
                'priority': row[9],
                'dataset_id': row[10],
                'feature_set_id': row[11],
                'experiment_type': row[12],
                'progress': row[13],
                'processed_count': row[14],
                'total_count': row[15],
                'notes': row[16]
            }
        return None
    
    def update_task_status(self, task_id: int, status: TaskStatus, 
                          result: Optional[Dict] = None, error_message: Optional[str] = None):
        """Update task status"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        if status == TaskStatus.RUNNING:
            c.execute("""
                UPDATE tasks SET status = ?, started_at = ? WHERE id = ?
            """, (status.value, datetime.now(), task_id))
        elif status in [TaskStatus.COMPLETED, TaskStatus.FAILED]:
            c.execute("""
                UPDATE tasks SET status = ?, completed_at = ?, result = ?, error_message = ? 
                WHERE id = ?
            """, (status.value, datetime.now(), 
                  json.dumps(result) if result else None,
                  error_message, task_id))
        
        conn.commit()
        conn.close()
    
    def update_task_progress(self, task_id: int, progress: float, processed_count: int, total_count: int):
        """Update task progress"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            UPDATE tasks SET progress = ?, processed_count = ?, total_count = ? WHERE id = ?
        """, (progress, processed_count, total_count, task_id))
        
        conn.commit()
        conn.close()
    
    def get_pending_tasks(self, limit: int = 10) -> list:
        """Get pending tasks"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            SELECT * FROM tasks 
            WHERE status = 'pending' 
            ORDER BY priority DESC, created_at ASC 
            LIMIT ?
        """, (limit,))
        
        rows = c.fetchall()
        conn.close()
        
        return [{
            'id': row[0],
            'task_type': row[1],
            'parameters': json.loads(row[3]) if row[3] else None,
            'dataset_id': row[10],
            'feature_set_id': row[11],
            'experiment_type': row[12],
            'priority': row[9]
        } for row in rows]
    
    def get_all_tasks(self) -> list:
        """Get all tasks"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            # Check if table exists
            c.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='tasks'
            """)
            
            if not c.fetchone():
                # If table does not exist, return empty list
                return []
            
            c.execute("""
                SELECT * FROM tasks 
                ORDER BY created_at DESC
            """)
            
            rows = c.fetchall()
            
            tasks = []
            for row in rows:
                task = {
                    'id': row[0],
                    'task_type': row[1],
                    'status': row[2],
                    'parameters': json.loads(row[3]) if row[3] else None,
                    'result': json.loads(row[4]) if row[4] else None,
                    'error_message': row[5],
                    'created_at': row[6],
                    'started_at': row[7],
                    'completed_at': row[8],
                    'priority': row[9],
                    'dataset_id': row[10],
                    'feature_set_id': row[11],
                    'experiment_type': row[12],
                    'progress': row[13],
                    'processed_count': row[14],
                    'total_count': row[15],
                    'notes': row[16]
                }
                tasks.append(task)
            
            return tasks
            
        except Exception as e:
            print(f"Error getting all tasks: {e}")
            return []
        finally:
            conn.close()
