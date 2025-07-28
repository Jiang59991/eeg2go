import threading
import time
import os
import sqlite3
from datetime import datetime
from typing import Dict, Any
from task_queue.models import TaskManager, TaskStatus
from eeg2fx.featureset_fetcher import run_feature_set
from feature_mill.experiment_engine import run_experiment
from logging_config import logger

class TaskWorker:
    def __init__(self, task_manager: TaskManager):
        self.task_manager = task_manager
        self.running = False
        self.worker_thread = None
    
    def start(self):
        """启动工作线程 - 在独立进程中不需要"""
        # 在独立进程中，我们不需要启动线程
        # 这个方法保留是为了兼容性
        logger.info("Task worker initialized (running in separate process)")
    
    def stop(self):
        """停止工作线程"""
        self.running = False
        if self.worker_thread:
            self.worker_thread.join()
        logger.info("Task worker stopped")
    
    def _work_loop(self):
        """工作循环 - 在独立进程中由worker_process.py调用"""
        while self.running:
            try:
                # 获取待处理任务
                pending_tasks = self.task_manager.get_pending_tasks(limit=5)
                
                for task_info in pending_tasks:
                    if not self.running:
                        break
                    
                    self._process_task(task_info)
                
                # 等待一段时间再检查新任务
                time.sleep(2)
                
            except Exception as e:
                logger.error(f"Error in task worker loop: {e}")
                time.sleep(5)
    
    def _process_task(self, task_info: Dict[str, Any]):
        """处理单个任务"""
        task_id = task_info['id']
        task_type = task_info['task_type']
        parameters = task_info['parameters']
        dataset_id = task_info.get('dataset_id')
        feature_set_id = task_info.get('feature_set_id')
        experiment_type = task_info.get('experiment_type')
        
        try:
            # 更新任务状态为运行中
            self.task_manager.update_task_status(task_id, TaskStatus.RUNNING)
            
            # 根据任务类型执行相应的处理
            if task_type == 'feature_extraction':
                result = self._execute_feature_extraction(task_id, parameters, dataset_id, feature_set_id)
            elif task_type == 'experiment':
                result = self._execute_experiment(task_id, parameters, dataset_id, feature_set_id, experiment_type)
            else:
                raise ValueError(f"Unknown task type: {task_type}")
            
            # 更新任务状态为完成
            self.task_manager.update_task_status(task_id, TaskStatus.COMPLETED, result=result)
            logger.info(f"Task {task_id} completed successfully")
            
        except Exception as e:
            error_message = str(e)
            self.task_manager.update_task_status(task_id, TaskStatus.FAILED, 
                                               error_message=error_message)
            logger.error(f"Task {task_id} failed: {error_message}")
    
    def _execute_feature_extraction(self, task_id: int, parameters: Dict[str, Any], 
                                  dataset_id: int, feature_set_id: int) -> Dict[str, Any]:
        """执行特征提取任务"""
        if not dataset_id or not feature_set_id:
            raise ValueError("Missing required parameters: dataset_id and feature_set_id")
        
        # 获取数据集中的所有录音ID
        recording_ids = self._get_recording_ids_for_dataset(dataset_id)
        
        if not recording_ids:
            raise ValueError(f"No recordings found for dataset {dataset_id}")
        
        # 更新任务总数
        self.task_manager.update_task_progress(task_id, 0.0, 0, len(recording_ids))
        
        results = {
            'dataset_id': dataset_id,
            'feature_set_id': feature_set_id,
            'total_recordings': len(recording_ids),
            'successful_recordings': 0,
            'failed_recordings': 0,
            'recording_results': {}
        }
        
        # 为每个录音执行特征提取
        for i, recording_id in enumerate(recording_ids):
            try:
                logger.info(f"Processing recording {recording_id} for feature extraction ({i+1}/{len(recording_ids)})")
                
                # 更新进度
                progress = ((i + 1) / len(recording_ids)) * 100
                self.task_manager.update_task_progress(task_id, progress, i+1, len(recording_ids))
                
                # 调用特征提取函数
                feature_results = run_feature_set(feature_set_id, recording_id)
                
                # 统计成功和失败的特征
                successful_features = sum(1 for result in feature_results.values() 
                                        if result and result.get('value') is not None)
                total_features = len(feature_results)
                
                results['recording_results'][recording_id] = {
                    'status': 'success',
                    'successful_features': successful_features,
                    'total_features': total_features,
                    'feature_results': feature_results
                }
                results['successful_recordings'] += 1
                
                logger.info(f"Recording {recording_id}: {successful_features}/{total_features} features extracted successfully")
                
            except Exception as e:
                logger.error(f"Failed to extract features for recording {recording_id}: {e}")
                results['recording_results'][recording_id] = {
                    'status': 'failed',
                    'error': str(e)
                }
                results['failed_recordings'] += 1
        
        # 更新最终进度
        self.task_manager.update_task_progress(task_id, 100.0, len(recording_ids), len(recording_ids))
        
        return results
    
    def _execute_experiment(self, task_id: int, parameters: Dict[str, Any], 
                          dataset_id: int, feature_set_id: int, experiment_type: str) -> Dict[str, Any]:
        """执行实验任务"""
        try:
            logger.info(f"开始执行实验任务 {task_id}: {experiment_type}")
            
            # 更新任务状态为运行中
            self.task_manager.update_task_status(task_id, TaskStatus.RUNNING)
            
            # 创建输出目录
            output_dir = f"experiments/{experiment_type}_{dataset_id}_{feature_set_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
            os.makedirs(output_dir, exist_ok=True)
            logger.info(f"创建输出目录: {output_dir}")
            
            # 获取实验参数
            experiment_params = parameters.get('parameters', {})
            logger.info(f"实验参数: {experiment_params}")
            
            # 运行实验
            from feature_mill.experiment_engine import run_experiment
            
            logger.info(f"调用run_experiment函数...")
            result = run_experiment(
                experiment_type=experiment_type,
                dataset_id=dataset_id,
                feature_set_id=feature_set_id,
                output_dir=output_dir,
                extra_args=experiment_params
            )
            
            logger.info(f"实验执行完成，结果: {result}")
            
            # 更新任务状态为完成
            self.task_manager.update_task_status(task_id, TaskStatus.COMPLETED, result=result)
            
            return result
            
        except Exception as e:
            logger.error(f"实验执行失败: {e}")
            import traceback
            logger.error(f"错误详情: {traceback.format_exc()}")
            self.task_manager.update_task_status(task_id, TaskStatus.FAILED, error_message=str(e))
            raise
    
    def _get_recording_ids_for_dataset(self, dataset_id: int) -> list:
        """获取数据集中的所有录音ID"""
        conn = sqlite3.connect(self.task_manager.db_path)
        c = conn.cursor()
        c.execute("SELECT id FROM recordings WHERE dataset_id = ?", (dataset_id,))
        recording_ids = [row[0] for row in c.fetchall()]
        conn.close()
        return recording_ids
