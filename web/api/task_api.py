from flask import Blueprint, request, jsonify
from task_queue.models import TaskManager, Task, TaskStatus
from task_queue.task_worker import TaskWorker
import threading

task_api = Blueprint('task_api', __name__)

# 全局任务管理器
task_manager = TaskManager()

# 移除全局工作器变量和启动函数
# task_worker = TaskWorker(task_manager)
# def start_task_worker():
#     task_worker.start()

@task_api.route('/api/tasks', methods=['POST'])
def create_task():
    """创建新任务"""
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
        
        # 创建任务
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
def get_task_status(task_id):
    """获取任务状态"""
    try:
        task = task_manager.get_task(task_id)
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        
        return jsonify(task), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@task_api.route('/api/task_details/<int:task_id>', methods=['GET'])
def get_task_details(task_id):
    """获取任务详细信息"""
    try:
        task = task_manager.get_task(task_id)
        if not task:
            return jsonify({
                'success': False,
                'error': 'Task not found'
            }), 404
        
        return jsonify({
            'success': True,
            'task': task
        }), 200
        
    except Exception as e:
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500

@task_api.route('/api/tasks/<int:task_id>/cancel', methods=['POST'])
def cancel_task(task_id):
    """取消任务"""
    try:
        task = task_manager.get_task(task_id)
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        
        if task['status'] in ['completed', 'failed']:
            return jsonify({'error': 'Cannot cancel completed or failed task'}), 400
        
        # 更新任务状态为失败
        task_manager.update_task_status(task_id, TaskStatus.FAILED, 
                                      error_message='Task cancelled by user')
        
        return jsonify({'message': 'Task cancelled successfully'}), 200
        
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@task_api.route('/api/tasks', methods=['GET'])
def get_tasks():
    """获取所有任务列表"""
    try:
        tasks = task_manager.get_all_tasks()
        return jsonify({
            'success': True,
            'tasks': tasks
        })
    except Exception as e:
        print(f"Error in get_tasks: {e}")
        return jsonify({
            'success': False,
            'error': str(e)
        }), 500
