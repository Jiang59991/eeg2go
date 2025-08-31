#!/usr/bin/env python3
"""
简单本地执行器
不使用线程池，直接在当前线程中顺序执行所有任务
"""
import os
import sys
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, Optional

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .common import TaskStatus
from .models import TaskManager
from logging_config import logger

class SimpleLocalExecutor:
    """简单本地执行器，直接在当前线程中执行所有任务"""
    
    def __init__(self):
        logger.info("SimpleLocalExecutor initialized - no threading, direct execution")
    
    def execute_feature_extraction(self, task_id: int, parameters: Dict[str, Any], 
                                 dataset_id: int, feature_set_id: int):
        """直接执行特征提取任务"""
        logger.info(f"=== Start executing feature extraction task {task_id} ===")
        logger.info(f"Task parameters: dataset_id={dataset_id}, feature_set_id={feature_set_id}")
        
        # 更新任务状态为运行中
        task_manager = TaskManager()
        task_manager.update_task_status(task_id, TaskStatus.RUNNING)
        
        # 获取数据集中的所有recording IDs
        recording_ids = self._get_recording_ids_for_dataset(dataset_id)
        
        if not recording_ids:
            raise ValueError(f"No recordings found for dataset {dataset_id}")
        
        logger.info(f"Found {len(recording_ids)} recordings to process")
        
        # 更新任务进度
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
        
        # 直接顺序处理所有recording
        successful_count = 0
        failed_count = 0
        
        for i, recording_id in enumerate(recording_ids):
            logger.info(f"Processing recording {recording_id} ({i+1}/{len(recording_ids)})")
            
            try:
                # 直接调用run_feature_set函数
                from eeg2fx.featureset_fetcher import run_feature_set
                result = run_feature_set(feature_set_id, recording_id)
                
                if result:
                    successful_count += 1
                    logger.info(f"Recording {recording_id} processed successfully")
                else:
                    failed_count += 1
                    error_msg = "run_feature_set returned None"
                    results['errors'].append(f"Recording {recording_id}: {error_msg}")
                    logger.warning(f"Recording {recording_id} failed: {error_msg}")
                    
            except Exception as e:
                failed_count += 1
                error_msg = str(e)
                results['errors'].append(f"Recording {recording_id}: {error_msg}")
                logger.error(f"Recording {recording_id} failed with exception: {error_msg}")
            
            # 更新进度
            processed_count = i + 1
            progress = (processed_count / len(recording_ids)) * 100
            task_manager.update_task_progress(task_id, progress, processed_count, len(recording_ids))
        
        # 更新最终结果
        results['processed_recordings'] = successful_count + failed_count
        results['successful_recordings'] = successful_count
        results['failed_recordings'] = failed_count
        
        logger.info(f"Feature extraction completed: {successful_count} successful, {failed_count} failed")
        
        # 更新任务状态
        if failed_count == 0:
            task_manager.update_task_status(task_id, TaskStatus.COMPLETED, result=results)
        else:
            task_manager.update_task_status(task_id, TaskStatus.COMPLETED, result=results)
        
        return results
    
    def execute_experiment(self, task_id: int, parameters: Dict[str, Any], 
                         dataset_id: int, feature_set_id: int, experiment_type: str):
        """直接执行实验任务"""
        logger.info(f"Start executing experiment task {task_id}")
        
        # 更新任务状态为运行中
        task_manager = TaskManager()
        task_manager.update_task_status(task_id, TaskStatus.RUNNING)
        
        # 生成唯一的输出目录
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = parameters.get('output_dir', f'experiments/{experiment_type}_{task_id}_{timestamp}')
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        
        extra_args = parameters.copy()
        
        # 移除已明确传递的参数
        for key in ['dataset_id', 'feature_set_id', 'experiment_type', 'output_dir']:
            extra_args.pop(key, None)
        
        logger.info(f"Calling run_experiment with parameters: experiment_type={experiment_type}, dataset_id={dataset_id}, feature_set_id={feature_set_id}")
        
        # 直接调用run_experiment函数
        from feature_mill.experiment_engine import run_experiment
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
    
    def _get_recording_ids_for_dataset(self, dataset_id: int) -> list:
        """获取数据集中的所有recording IDs"""
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
    
    def submit_task(self, task_id: int, task_type: str, parameters: Dict[str, Any], 
                   dataset_id: Optional[int] = None, feature_set_id: Optional[int] = None,
                   experiment_type: Optional[str] = None):
        """提交任务到简单本地执行器"""
        logger.info(f"Task {task_id} submitted to simple local executor")
        
        try:
            if task_type == 'feature_extraction':
                return self.execute_feature_extraction(
                    task_id, parameters, dataset_id, feature_set_id
                )
            elif task_type == 'experiment':
                return self.execute_experiment(
                    task_id, parameters, dataset_id, feature_set_id, experiment_type
                )
            else:
                raise ValueError(f"Unknown task type: {task_type}")
                
        except Exception as e:
            error_message = str(e)
            logger.error(f"Task {task_id} failed: {error_message}")
            
            # 更新任务状态为失败
            task_manager = TaskManager()
            task_manager.update_task_status(task_id, TaskStatus.FAILED, error_message=error_message)
            
            raise

# 全局执行器实例
_simple_local_executor = None

def get_simple_local_executor() -> SimpleLocalExecutor:
    """获取全局简单本地执行器实例"""
    global _simple_local_executor
    if _simple_local_executor is None:
        _simple_local_executor = SimpleLocalExecutor()
    return _simple_local_executor
