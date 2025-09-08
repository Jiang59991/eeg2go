"""
Celery task definition module
Convert the original task processing logic into Celery tasks
"""
import os
import sys
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, Optional
from celery import current_task
from celery.utils.log import get_task_logger
import traceback

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .celery_app import celery_app
from .common import TaskStatus
from .models import TaskManager
from eeg2fx.featureset_fetcher import run_feature_set
from feature_mill.experiment_engine import run_experiment

# Use global logging config instead of Celery's task logger
from logging_config import logger

def get_task_manager():
    """Get task manager instance"""
    db_path = os.getenv('DATABASE_PATH', 'database/eeg2go.db')
    return TaskManager(db_path)

@celery_app.task(bind=True, name='task_queue.tasks.feature_extraction_task')
def feature_extraction_task(self, task_id: int, parameters: Dict[str, Any], 
                           dataset_id: int, feature_set_id: int):
    """
    Feature extraction task
    """
    logger.info(f"=== Start executing feature extraction task {task_id} ===")
    logger.info(f"Task parameters: dataset_id={dataset_id}, feature_set_id={feature_set_id}")
    logger.info(f"Parameter details: {parameters}")
    
    try:
        # Update task status to running
        logger.info(f"Updating task status to running...")
        task_manager = get_task_manager()
        task_manager.update_task_status(task_id, TaskStatus.RUNNING)
        
        # Get all recording IDs in the dataset
        logger.info(f"Getting all recording IDs in dataset {dataset_id}...")
        recording_ids = _get_recording_ids_for_dataset(dataset_id)
        
        if not recording_ids:
            raise ValueError(f"No recordings found for dataset {dataset_id}")
        
        logger.info(f"Found {len(recording_ids)} recordings to process")
        
        # Update task total count
        task_manager.update_task_progress(task_id, 0.0, 0, len(recording_ids))
        
        results = {
            'dataset_id': dataset_id,
            'feature_set_id': feature_set_id,
            'total_recordings': len(recording_ids),
            'processed_recordings': 0,
            'successful_recordings': 0,
            'failed_recordings': 0,
            'errors': []
        }
        
        # Schedule all recording tasks for parallel processing
        logger.info(f"Scheduling {len(recording_ids)} recording tasks for parallel processing...")
        
        # 使用Celery的group功能来并行执行任务
        from celery import group
        
        # 创建任务组
        job = group([
            run_feature_set_task.s(feature_set_id, recording_id)
            for recording_id in recording_ids
        ])
        
        logger.info(f"All {len(recording_ids)} recording tasks scheduled, waiting for completion...")
        
        # 执行任务组并等待结果 - 修复：不在任务中等待结果
        result_group = job.apply_async(queue='recordings')
        
        # 不在这里等待结果，而是立即返回任务组ID
        # 让外部监控器来处理结果收集
        
        logger.info(f"Task group created with ID: {result_group.id}")
        
        # 更新任务状态为运行中，但标记为需要外部监控
        task_manager.update_task_status(task_id, TaskStatus.RUNNING, result={
            'task_group_id': result_group.id,
            'total_recordings': len(recording_ids),
            'status': 'scheduled'
        })
        
        logger.info(f"=== Feature extraction task {task_id} scheduled successfully ===")
        return {
            'task_group_id': result_group.id,
            'total_recordings': len(recording_ids),
            'status': 'scheduled'
        }
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Exception occurred during feature extraction task {task_id}: {error_message}")
        task_manager = get_task_manager()
        task_manager.update_task_status(task_id, TaskStatus.FAILED, error_message=error_message)
        logger.error(f"=== Feature extraction task {task_id} failed ===")
        raise

@celery_app.task(bind=True, name='task_queue.tasks.experiment_task')
def experiment_task(self, task_id: int, parameters: Dict[str, Any], 
                   dataset_id: int, feature_set_id: int, experiment_type: str):
    """
    Experiment task
    """
    logger.info(f"Start executing experiment task {task_id}")
    
    try:
        # Update task status to running
        logger.info(f"Updating task {task_id} status to RUNNING...")
        task_manager = get_task_manager()
        task_manager.update_task_status(task_id, TaskStatus.RUNNING)
        
        # Execute experiment
        # 生成唯一的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = parameters.get('output_dir', f'experiments/{experiment_type}_{task_id}_{timestamp}')
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        # 获取实验参数（嵌套在parameters字段中）
        extra_args = parameters.get('parameters', {}).copy()
        
        # 移除已明确传递的参数
        for key in ['dataset_id', 'feature_set_id', 'experiment_type', 'output_dir']:
            extra_args.pop(key, None)
        
        logger.info(f"Calling run_experiment with parameters: experiment_type={experiment_type}, dataset_id={dataset_id}, feature_set_id={feature_set_id}")
        
        result = run_experiment(
            experiment_type=experiment_type,
            dataset_id=dataset_id,
            feature_set_id=feature_set_id,
            output_dir=output_dir,
            extra_args=extra_args
        )
        
        logger.info(f"run_experiment completed successfully for task {task_id}")
        logger.info(f"Result: {result}")
        
        # 更新任务状态，包含输出目录信息
        result['output_dir'] = output_dir
        logger.info(f"Updating task {task_id} status to COMPLETED...")
        task_manager.update_task_status(task_id, TaskStatus.COMPLETED, result=result)
        
        # 更新数据库中的output_dir字段
        logger.info(f"Updating output_dir in database for task {task_id}...")
        conn = sqlite3.connect(task_manager.db_path)
        c = conn.cursor()
        c.execute("UPDATE tasks SET output_dir = ? WHERE id = ?", (output_dir, task_id))
        conn.commit()
        conn.close()
        
        logger.info(f"Experiment task {task_id} completed successfully, output_dir: {output_dir}")
        
        return result
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"Exception in experiment task {task_id}: {error_message}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        
        try:
            task_manager = get_task_manager()
            task_manager.update_task_status(task_id, TaskStatus.FAILED, error_message=error_message)
            logger.info(f"Task {task_id} status updated to FAILED")
        except Exception as update_error:
            logger.error(f"Failed to update task status to FAILED: {update_error}")
        
        logger.error(f"Experiment task {task_id} failed: {error_message}")
        raise

def _get_recording_ids_for_dataset(dataset_id: int) -> list:
    """Get all recording IDs in the dataset"""
    db_path = os.getenv('DATABASE_PATH', 'database/eeg2go.db')
    
    logger.info(f"Getting recording IDs for dataset {dataset_id} from database...")
    
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        c.execute("SELECT id FROM recordings WHERE dataset_id = ?", (dataset_id,))
        recording_ids = [row[0] for row in c.fetchall()]
        
        conn.close()
        
        logger.info(f"Successfully got {len(recording_ids)} recording IDs for dataset {dataset_id}")
        return recording_ids
        
    except Exception as e:
        logger.error(f"Failed to get recording IDs for dataset {dataset_id}: {e}")
        return [] 

@celery_app.task(bind=True, name='task_queue.tasks.run_feature_set_task')
def run_feature_set_task(self, feature_set_id: int, recording_id: int):
    """
    包装run_feature_set函数的Celery任务
    将整个run_feature_set函数作为任务执行
    """
    logger.info(f"=== Start run_feature_set task: feature_set_id={feature_set_id}, recording_id={recording_id} ===")
    
    try:
        # 设置环境变量，标识当前在Celery worker中运行
        import os
        os.environ['CELERY_WORKER_RUNNING'] = '1'
        
        # 直接调用run_feature_set函数
        from eeg2fx.featureset_fetcher import run_feature_set
        result = run_feature_set(feature_set_id, recording_id)
        
        logger.info(f"=== run_feature_set task completed: feature_set_id={feature_set_id}, recording_id={recording_id} ===")
        return {
            'feature_set_id': feature_set_id,
            'recording_id': recording_id,
            'success': True,
            'result': result
        }
        
    except Exception as e:
        error_message = str(e)
        logger.error(f"run_feature_set task failed: feature_set_id={feature_set_id}, recording_id={recording_id}, error={error_message}")
        return {
            'feature_set_id': feature_set_id,
            'recording_id': recording_id,
            'success': False,
            'error': error_message
        }
    finally:
        # 清理环境变量
        import os
        os.environ.pop('CELERY_WORKER_RUNNING', None) 