"""
Celery任务定义模块
将原有的任务处理逻辑转换为Celery任务
"""
import os
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, Optional
from celery import current_task
from celery.utils.log import get_task_logger

from .celery_app import celery_app
from .models import TaskManager, TaskStatus
from eeg2fx.featureset_fetcher import run_feature_set
from feature_mill.experiment_engine import run_experiment

logger = get_task_logger(__name__)

def get_task_manager():
    """获取任务管理器实例"""
    db_path = os.getenv('DATABASE_PATH', 'database/eeg2go.db')
    return TaskManager(db_path)

@celery_app.task(bind=True, name='task_queue.tasks.feature_extraction_task')
def feature_extraction_task(self, task_id: int, parameters: Dict[str, Any], 
                           dataset_id: int, feature_set_id: int):
    """
    特征提取任务
    """
    logger.info(f"开始执行特征提取任务 {task_id}")
    
    try:
        # 更新任务状态为运行中
        task_manager = get_task_manager()
        task_manager.update_task_status(task_id, TaskStatus.RUNNING)
        
        # 获取数据集中的所有录音ID
        recording_ids = _get_recording_ids_for_dataset(dataset_id)
        
        if not recording_ids:
            raise ValueError(f"No recordings found for dataset {dataset_id}")
        
        # 更新任务总数
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
        
        # 处理每个录音
        for i, recording_id in enumerate(recording_ids):
            try:
                # 更新进度
                progress = (i / len(recording_ids)) * 100
                task_manager.update_task_progress(task_id, progress, i, len(recording_ids))
                
                # 执行特征提取
                result = run_feature_set(
                    recording_id=recording_id,
                    feature_set_id=feature_set_id,
                    **parameters
                )
                
                results['successful_recordings'] += 1
                logger.info(f"录音 {recording_id} 特征提取成功")
                
            except Exception as e:
                error_msg = f"录音 {recording_id} 处理失败: {str(e)}"
                results['errors'].append(error_msg)
                results['failed_recordings'] += 1
                logger.error(error_msg)
            
            results['processed_recordings'] += 1
        
        # 更新最终进度
        task_manager.update_task_progress(task_id, 100.0, len(recording_ids), len(recording_ids))
        
        # 更新任务状态为完成
        task_manager.update_task_status(task_id, TaskStatus.COMPLETED, result=results)
        logger.info(f"特征提取任务 {task_id} 完成")
        
        return results
        
    except Exception as e:
        error_message = str(e)
        task_manager = get_task_manager()
        task_manager.update_task_status(task_id, TaskStatus.FAILED, error_message=error_message)
        logger.error(f"特征提取任务 {task_id} 失败: {error_message}")
        raise

@celery_app.task(bind=True, name='task_queue.tasks.experiment_task')
def experiment_task(self, task_id: int, parameters: Dict[str, Any], 
                   dataset_id: int, feature_set_id: int, experiment_type: str):
    """
    实验任务
    """
    logger.info(f"开始执行实验任务 {task_id}")
    
    try:
        # 更新任务状态为运行中
        task_manager = get_task_manager()
        task_manager.update_task_status(task_id, TaskStatus.RUNNING)
        
        # 执行实验
        result = run_experiment(
            dataset_id=dataset_id,
            feature_set_id=feature_set_id,
            experiment_type=experiment_type,
            **parameters
        )
        
        # 更新任务状态为完成
        task_manager.update_task_status(task_id, TaskStatus.COMPLETED, result=result)
        logger.info(f"实验任务 {task_id} 完成")
        
        return result
        
    except Exception as e:
        error_message = str(e)
        task_manager = get_task_manager()
        task_manager.update_task_status(task_id, TaskStatus.FAILED, error_message=error_message)
        logger.error(f"实验任务 {task_id} 失败: {error_message}")
        raise

def _get_recording_ids_for_dataset(dataset_id: int) -> list:
    """获取数据集中的所有录音ID"""
    db_path = os.getenv('DATABASE_PATH', 'database/eeg2go.db')
    
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        c.execute("SELECT id FROM recordings WHERE dataset_id = ?", (dataset_id,))
        recording_ids = [row[0] for row in c.fetchall()]
        
        conn.close()
        return recording_ids
        
    except Exception as e:
        logger.error(f"获取数据集 {dataset_id} 录音ID失败: {e}")
        return [] 