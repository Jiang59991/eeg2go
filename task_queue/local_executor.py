#!/usr/bin/env python3
"""
本地任务执行器
不使用Redis，全部在本机运行完成
"""
import os
import sys
import sqlite3
import json
import threading
import time
import queue
from datetime import datetime
from typing import Dict, Any, Optional, Callable
from concurrent.futures import ThreadPoolExecutor, as_completed
import traceback

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .common import TaskStatus
from .models import TaskManager
from logging_config import logger

class LocalTaskExecutor:
    """本地任务执行器，不使用Redis"""
    
    def __init__(self, max_workers: int = 4):
        self.max_workers = max_workers
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.task_queue = queue.Queue()
        self.running_tasks = {}
        self.task_results = {}
        self.shutdown_event = threading.Event()
        
        # 启动任务处理线程
        self.worker_thread = threading.Thread(target=self._task_worker, daemon=True)
        self.worker_thread.start()
        
        logger.info(f"LocalTaskExecutor initialized with {max_workers} workers")
    
    def _task_worker(self):
        """任务处理工作线程"""
        while not self.shutdown_event.is_set():
            try:
                # 从队列中获取任务，超时1秒
                task_data = self.task_queue.get(timeout=1)
                if task_data is None:  # 关闭信号
                    break
                
                task_id, task_type, parameters, dataset_id, feature_set_id, experiment_type = task_data
                
                logger.info(f"Processing task {task_id} ({task_type}) in local executor")
                
                # 执行任务
                try:
                    if task_type == 'feature_extraction':
                        result = self._execute_feature_extraction(
                            task_id, parameters, dataset_id, feature_set_id
                        )
                    elif task_type == 'experiment':
                        result = self._execute_experiment(
                            task_id, parameters, dataset_id, feature_set_id, experiment_type
                        )
                    else:
                        raise ValueError(f"Unknown task type: {task_type}")
                    
                    # 存储结果
                    self.task_results[task_id] = {
                        'status': 'completed',
                        'result': result,
                        'completed_at': datetime.now()
                    }
                    
                    logger.info(f"Task {task_id} completed successfully")
                    
                except Exception as e:
                    error_message = str(e)
                    logger.error(f"Task {task_id} failed: {error_message}")
                    
                    self.task_results[task_id] = {
                        'status': 'failed',
                        'error': error_message,
                        'completed_at': datetime.now()
                    }
                
                finally:
                    # 标记任务完成
                    self.task_queue.task_done()
                    
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in task worker: {e}")
                time.sleep(1)
    
    def _execute_feature_extraction(self, task_id: int, parameters: Dict[str, Any], 
                                  dataset_id: int, feature_set_id: int):
        """执行特征提取任务"""
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
        
        # 使用线程池并行处理recording任务
        successful_count = 0
        failed_count = 0
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # 提交所有recording任务
            future_to_recording = {
                executor.submit(self._execute_single_recording, feature_set_id, recording_id): recording_id
                for recording_id in recording_ids
            }
            
            # 收集结果
            for future in as_completed(future_to_recording):
                recording_id = future_to_recording[future]
                try:
                    result = future.result()
                    if result.get('success'):
                        successful_count += 1
                        logger.info(f"Recording {recording_id} processed successfully")
                    else:
                        failed_count += 1
                        error_msg = result.get('error', 'Unknown error')
                        results['errors'].append(f"Recording {recording_id}: {error_msg}")
                        logger.warning(f"Recording {recording_id} failed: {error_msg}")
                except Exception as e:
                    failed_count += 1
                    error_msg = str(e)
                    results['errors'].append(f"Recording {recording_id}: {error_msg}")
                    logger.error(f"Recording {recording_id} failed with exception: {error_msg}")
                
                # 更新进度
                processed_count = successful_count + failed_count
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
    
    def _execute_single_recording(self, feature_set_id: int, recording_id: int):
        """执行单个recording的特征提取"""
        try:
            # 设置环境变量，标识当前在本地执行器中运行
            os.environ['LOCAL_EXECUTOR_RUNNING'] = '1'
            
            # 直接调用run_feature_set函数
            from eeg2fx.featureset_fetcher import run_feature_set
            result = run_feature_set(feature_set_id, recording_id)
            
            return {
                'feature_set_id': feature_set_id,
                'recording_id': recording_id,
                'success': True,
                'result': result
            }
            
        except Exception as e:
            error_message = str(e)
            logger.error(f"Recording {recording_id} failed: {error_message}")
            return {
                'feature_set_id': feature_set_id,
                'recording_id': recording_id,
                'success': False,
                'error': error_message
            }
        finally:
            # 清理环境变量
            os.environ.pop('LOCAL_EXECUTOR_RUNNING', None)
    
    def _execute_experiment(self, task_id: int, parameters: Dict[str, Any], 
                          dataset_id: int, feature_set_id: int, experiment_type: str):
        """执行实验任务"""
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
        """提交任务到本地执行器"""
        task_data = (task_id, task_type, parameters, dataset_id, feature_set_id, experiment_type)
        self.task_queue.put(task_data)
        self.running_tasks[task_id] = {
            'submitted_at': datetime.now(),
            'status': 'submitted'
        }
        logger.info(f"Task {task_id} submitted to local executor")
    
    def get_task_result(self, task_id: int) -> Optional[Dict[str, Any]]:
        """获取任务结果"""
        return self.task_results.get(task_id)
    
    def is_task_completed(self, task_id: int) -> bool:
        """检查任务是否完成"""
        return task_id in self.task_results
    
    def shutdown(self):
        """关闭执行器"""
        logger.info("Shutting down LocalTaskExecutor...")
        self.shutdown_event.set()
        self.task_queue.put(None)  # 发送关闭信号
        self.executor.shutdown(wait=True)
        logger.info("LocalTaskExecutor shutdown complete")

# 全局执行器实例
_local_executor = None

def get_local_executor() -> LocalTaskExecutor:
    """获取全局本地执行器实例"""
    global _local_executor
    if _local_executor is None:
        max_workers = int(os.getenv('LOCAL_EXECUTOR_WORKERS', '4'))
        _local_executor = LocalTaskExecutor(max_workers=max_workers)
    return _local_executor

def shutdown_local_executor():
    """关闭全局本地执行器"""
    global _local_executor
    if _local_executor is not None:
        _local_executor.shutdown()
        _local_executor = None

