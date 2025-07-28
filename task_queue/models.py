import json
import sqlite3
from datetime import datetime
from enum import Enum
from typing import Dict, Any, Optional

class TaskStatus(Enum):
    PENDING = "pending"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"

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
        """创建新任务并返回任务ID"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # 确保数据类型正确
            dataset_id = int(task.dataset_id) if task.dataset_id is not None else None
            feature_set_id = int(task.feature_set_id) if task.feature_set_id is not None else None
            
            print(f"Creating task with dataset_id={dataset_id}, feature_set_id={feature_set_id}")
            
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
            
            print(f"Task created successfully with ID: {task_id}")
            return task_id
            
        except Exception as e:
            print(f"Error creating task: {e}")
            if 'conn' in locals():
                conn.close()
            raise e
    
    def get_task(self, task_id: int) -> Optional[Dict[str, Any]]:
        """获取任务信息"""
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
        """更新任务状态"""
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
        """更新任务进度"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            UPDATE tasks SET progress = ?, processed_count = ?, total_count = ? WHERE id = ?
        """, (progress, processed_count, total_count, task_id))
        
        conn.commit()
        conn.close()
    
    def get_pending_tasks(self, limit: int = 10) -> list:
        """获取待处理任务"""
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
        """获取所有任务"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        try:
            # 检查表是否存在
            c.execute("""
                SELECT name FROM sqlite_master 
                WHERE type='table' AND name='tasks'
            """)
            
            if not c.fetchone():
                # 如果表不存在，返回空列表
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
