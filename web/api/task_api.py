from flask import Blueprint, request, jsonify
import json
import os
from task_queue.models import TaskManager, Task, TaskStatus
from typing import Tuple

task_api = Blueprint('task_api', __name__)

task_manager = TaskManager()

@task_api.route('/api/tasks', methods=['POST'])
def create_task() -> Tuple:
    """
    Create a new task.
    Returns:
        Tuple: (json response, status code)
    """
    try:
        data = request.get_json()
        task_type = data.get('task_type')
        parameters = data.get('parameters', {})
        dataset_id = data.get('dataset_id')
        feature_set_id = data.get('feature_set_id')
        experiment_type = data.get('experiment_type')
        priority = data.get('priority', 0)
        
        if not task_type:
            return jsonify({'error': 'task_type is required'}), 400
        
        task = Task(task_type, parameters, dataset_id, feature_set_id, experiment_type, priority)
        task_id = task_manager.create_task(task)
        
        return jsonify({
            'task_id': task_id,
            'status': 'pending',
            'message': 'Task created successfully'
        }), 201
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@task_api.route('/api/tasks/<int:task_id>', methods=['GET'])
def get_task_status(task_id: int) -> Tuple:
    """
    Get the status of a task.
    Args:
        task_id (int): The ID of the task.
    Returns:
        Tuple: (json response, status code)
    """
    try:
        task = task_manager.get_task(task_id)
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        
        return jsonify(task), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@task_api.route('/api/task_details/<int:task_id>', methods=['GET'])
def get_task_details(task_id: int) -> Tuple:
    """
    Get detailed information of a task.
    Args:
        task_id (int): The ID of the task.
    Returns:
        Tuple: (json response, status code)
    """
    try:
        import sqlite3
        
        db_path = os.getenv('DATABASE_PATH', 'database/eeg2go.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Query task basic info, exclude large result field
        c.execute("""
            SELECT t.id, t.task_type, t.status, t.parameters, t.error_message, 
                   t.created_at, t.started_at, t.completed_at, t.priority,
                   t.dataset_id, t.feature_set_id, t.experiment_type, 
                   t.progress, t.processed_count, t.total_count, t.notes,
                   d.name as dataset_name, fs.name as feature_set_name
            FROM tasks t
            LEFT JOIN datasets d ON t.dataset_id = d.id
            LEFT JOIN feature_sets fs ON t.feature_set_id = fs.id
            WHERE t.id = ?
        """, (task_id,))
        
        row = c.fetchone()
        conn.close()
        
        if not row:
            return jsonify({
                'success': False,
                'error': 'Task not found'
            }), 404
        
        def safe_json_loads(json_str):
            """
            Safely load JSON string, return summary if too large.
            """
            if not json_str:
                return None
            try:
                if len(str(json_str)) > 10000:
                    return {
                        'type': 'large_data',
                        'size': len(str(json_str)),
                        'note': 'Data too large to display, use download link'
                    }
                return json.loads(json_str)
            except Exception as e:
                return {
                    'type': 'parse_error',
                    'error': str(e),
                    'preview': str(json_str)[:200] + "..." if len(str(json_str)) > 200 else str(json_str)
                }
        
        task_details = {
            'id': int(row[0]) if row[0] is not None else None,
            'task_type': str(row[1]) if row[1] is not None else None,
            'status': str(row[2]) if row[2] is not None else None,
            'parameters': safe_json_loads(row[3]),
            'error_message': str(row[4]) if row[4] is not None else None,
            'created_at': str(row[5]) if row[5] is not None else None,
            'started_at': str(row[6]) if row[6] is not None else None,
            'completed_at': str(row[7]) if row[7] is not None else None,
            'priority': int(row[8]) if row[8] is not None else 0,
            'dataset_id': int(row[9]) if row[9] is not None else None,
            'feature_set_id': int(row[10]) if row[10] is not None else None,
            'experiment_type': str(row[11]) if row[11] is not None else None,
            'progress': float(row[12]) if row[12] is not None else 0.0,
            'processed_count': int(row[13]) if row[13] is not None else 0,
            'total_count': int(row[14]) if row[14] is not None else 0,
            'notes': str(row[15]) if row[15] is not None else None,
            'dataset_name': str(row[16]) if row[16] is not None else None,
            'feature_set_name': str(row[17]) if row[17] is not None else None
        }
        
        return jsonify({
            'success': True,
            'task': task_details
        }), 200
        
    except Exception as e:
        print(f"Error in get_task_details: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@task_api.route('/api/tasks/<int:task_id>/cancel', methods=['POST'])
def cancel_task(task_id: int) -> Tuple:
    """
    Cancel a task.
    Args:
        task_id (int): The ID of the task.
    Returns:
        Tuple: (json response, status code)
    """
    try:
        task = task_manager.get_task(task_id)
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        
        if task['status'] in ['completed', 'failed']:
            return jsonify({'error': 'Cannot cancel completed or failed task'}), 400
        
        # Set task status to failed
        task_manager.update_task_status(task_id, TaskStatus.FAILED, 
                                      error_message='Task cancelled by user')
        
        return jsonify({'message': 'Task cancelled successfully'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@task_api.route('/api/tasks', methods=['GET'])
def get_tasks() -> Tuple:
    """
    Get a list of all tasks (basic info only).
    Returns:
        Tuple: (json response, status code)
    """
    try:
        import sqlite3
        
        db_path = os.getenv('DATABASE_PATH', 'database/eeg2go.db')
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Only select basic fields to reduce data size
        c.execute("""
            SELECT t.id, t.task_type, t.status, t.created_at, t.progress, t.priority,
                   d.name as dataset_name, fs.name as feature_set_name
            FROM tasks t
            LEFT JOIN datasets d ON t.dataset_id = d.id
            LEFT JOIN feature_sets fs ON t.feature_set_id = fs.id
            ORDER BY t.created_at DESC
            LIMIT 50
        """)
        
        rows = c.fetchall()
        conn.close()
        
        processed_tasks = []
        for row in rows:
            task_dict = {
                'id': int(row[0]) if row[0] is not None else None,
                'task_type': str(row[1]) if row[1] is not None else None,
                'status': str(row[2]) if row[2] is not None else None,
                'created_at': str(row[3]) if row[3] is not None else None,
                'progress': float(row[4]) if row[4] is not None else 0.0,
                'priority': int(row[5]) if row[5] is not None else 0,
                'dataset_name': str(row[6]) if row[6] is not None else None,
                'feature_set_name': str(row[7]) if row[7] is not None else None
            }
            processed_tasks.append(task_dict)
        
        return jsonify({
            'success': True,
            'tasks': processed_tasks
        }), 200
        
    except Exception as e:
        print(f"Error in get_tasks: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@task_api.route('/api/system/mode', methods=['GET'])
def get_system_mode() -> Tuple:
    """
    Get the current system running mode.
    Returns:
        Tuple: (json response, status code)
    """
    try:
        use_local_mode = os.getenv('USE_LOCAL_EXECUTOR', 'false').lower() == 'true'
        
        mode_info = {
            'mode': 'local' if use_local_mode else 'celery',
            'description': 'Local Executor Mode (No Redis)' if use_local_mode else 'Celery Distributed Mode (Using Redis)',
            'workers': int(os.getenv('LOCAL_EXECUTOR_WORKERS', '4')) if use_local_mode else 'N/A'
        }
        
        return jsonify({
            'success': True,
            'mode_info': mode_info
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
