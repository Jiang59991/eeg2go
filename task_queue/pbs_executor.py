#!/usr/bin/env python3
"""
PBS任务执行器 - 与原有task_queue系统融合

这个模块负责将原有的任务转换为PBS作业并提交执行
"""

import os
import sys
import subprocess
import json
import time
from datetime import datetime
from typing import Dict, Any, Optional
from pathlib import Path
import sqlite3

from task_queue.models import TaskManager, TaskStatus
from task_queue.pbs_task_manager import pbs_manager

class PBSExecutor:
    """PBS任务执行器"""
    
    def __init__(self, task_manager: TaskManager):
        self.task_manager = task_manager
        self.template_dir = Path("pbs_templates")
    
    def submit_task_to_pbs(self, task_info: Dict[str, Any]) -> bool:
        """将任务提交到PBS系统"""
        task_id = task_info['id']
        task_type = task_info['task_type']
        parameters = task_info['parameters'] or {}
        dataset_id = task_info.get('dataset_id')
        feature_set_id = task_info.get('feature_set_id')
        
        try:
            print(f"提交任务 {task_id} 到PBS系统")
            
            if task_type == 'feature_extraction':
                return self._submit_feature_extraction_task(task_id, dataset_id, feature_set_id, parameters)
            elif task_type == 'experiment':
                return self._submit_experiment_task(task_id, dataset_id, feature_set_id, parameters)
            else:
                print(f"不支持的任务类型: {task_type}")
                return False
                
        except Exception as e:
            print(f"提交任务到PBS失败: {e}")
            # 更新任务状态为失败
            self.task_manager.update_task_status(task_id, TaskStatus.FAILED, 
                                               error_message=f"PBS提交失败: {e}")
            return False
    
    def _submit_feature_extraction_task(self, task_id: int, dataset_id: int, 
                                       feature_set_id: int, parameters: Dict[str, Any]) -> bool:
        """提交特征提取任务"""
        try:
            # 获取数据集中的录音ID列表
            recording_ids = self._get_recording_ids_for_dataset(dataset_id)
            
            if not recording_ids:
                error_msg = f"数据集 {dataset_id} 中没有找到录音记录"
                self.task_manager.update_task_status(task_id, TaskStatus.FAILED, 
                                                   error_message=error_msg)
                return False
            
            print(f"数据集 {dataset_id} 包含 {len(recording_ids)} 个录音记录")
            
            # 为每个录音提交PBS任务
            successful_submissions = 0
            
            for recording_id in recording_ids:
                try:
                    # 获取首选队列
                    preferred_queue = parameters.get('preferred_queue')
                    
                    # 使用PBS管理器提交任务
                    pbs_job_id = pbs_manager.submit_task(
                        task_type="feature_extraction",
                        recording_id=recording_id,
                        feature_set_id=feature_set_id,
                        dataset_id=dataset_id,
                        parameters=parameters,
                        preferred_queue=preferred_queue
                    )
                    
                    if pbs_job_id:
                        successful_submissions += 1
                        print(f"录音 {recording_id} 的PBS任务提交成功: {pbs_job_id}")
                    else:
                        print(f"录音 {recording_id} 的PBS任务提交失败")
                        
                except Exception as e:
                    print(f"提交录音 {recording_id} 的任务时出错: {e}")
            
            # 更新任务状态
            if successful_submissions > 0:
                # 更新任务为运行状态，并记录PBS信息
                self.task_manager.update_task_status(task_id, TaskStatus.RUNNING)
                self.task_manager.update_pbs_info(task_id, f"batch_{task_id}", "pbs_batch")
                
                print(f"特征提取任务 {task_id} 已提交 {successful_submissions}/{len(recording_ids)} 个PBS作业")
                return True
            else:
                error_msg = "所有PBS作业提交都失败了"
                self.task_manager.update_task_status(task_id, TaskStatus.FAILED, 
                                                   error_message=error_msg)
                return False
                
        except Exception as e:
            print(f"提交特征提取任务失败: {e}")
            self.task_manager.update_task_status(task_id, TaskStatus.FAILED, 
                                               error_message=str(e))
            return False
    
    def _submit_experiment_task(self, task_id: int, dataset_id: int, 
                               feature_set_id: int, parameters: Dict[str, Any]) -> bool:
        """提交实验任务"""
        try:
            # 选择最佳队列
            queue_name = pbs_manager.select_best_queue("experiment")
            
            # 准备PBS脚本
            script_content = self._prepare_experiment_pbs_script(
                task_id, dataset_id, feature_set_id, queue_name, parameters
            )
            
            # 创建临时脚本文件
            script_file = f"temp_experiment_{task_id}.pbs"
            with open(script_file, 'w') as f:
                f.write(script_content)
            
            # 提交任务
            cmd = ['qsub', script_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                pbs_job_id = result.stdout.strip()
                print(f"实验任务 {task_id} 提交成功: {pbs_job_id}")
                
                # 更新任务状态
                self.task_manager.update_task_status(task_id, TaskStatus.RUNNING)
                self.task_manager.update_pbs_info(task_id, pbs_job_id, queue_name)
                
                # 清理临时文件
                os.remove(script_file)
                return True
            else:
                print(f"实验任务提交失败: {result.stderr}")
                self.task_manager.update_task_status(task_id, TaskStatus.FAILED, 
                                                   error_message=f"PBS提交失败: {result.stderr}")
                return False
                
        except Exception as e:
            print(f"提交实验任务失败: {e}")
            self.task_manager.update_task_status(task_id, TaskStatus.FAILED, 
                                               error_message=str(e))
            return False
    
    def _prepare_experiment_pbs_script(self, task_id: int, dataset_id: int, 
                                      feature_set_id: int, queue_name: str, 
                                      parameters: Dict[str, Any]) -> str:
        """准备实验PBS脚本"""
        experiment_type = parameters.get('experiment_type', 'classification')
        
        script = f"""#!/bin/bash
#PBS -N eeg_experiment_{task_id}
#PBS -l nodes=1:ppn=8
#PBS -l walltime=04:00:00
#PBS -q {queue_name}
#PBS -j oe
#PBS -o logs/experiment_{task_id}.log

cd $PBS_O_WORKDIR

# 加载必要的模块
module load python/3.8
module load anaconda3

# 设置环境变量
export TASK_ID={task_id}
export DATASET_ID={dataset_id}
export FEATURE_SET_ID={feature_set_id}
export EXPERIMENT_TYPE={experiment_type}

echo "=== EEG实验任务开始 ==="
echo "时间: $(date)"
echo "任务ID: $TASK_ID"
echo "数据集ID: $DATASET_ID"
echo "特征集ID: $FEATURE_SET_ID"
echo "实验类型: $EXPERIMENT_TYPE"

# 执行实验
python3 -c "
import sys
import os
sys.path.append('/rds/general/user/zj724/home/eeg2go')

from feature_mill.experiment_engine import run_experiment
from datetime import datetime

try:
    task_id = int(os.environ['TASK_ID'])
    dataset_id = int(os.environ['DATASET_ID'])
    feature_set_id = int(os.environ['FEATURE_SET_ID'])
    experiment_type = os.environ['EXPERIMENT_TYPE']
    
    print(f'开始执行实验: {{experiment_type}}')
    
    # 创建输出目录
    output_dir = f'experiments/{{experiment_type}}_{{dataset_id}}_{{feature_set_id}}_{{datetime.now().strftime(\"%Y%m%d_%H%M%S\")}}'
    os.makedirs(output_dir, exist_ok=True)
    
    # 运行实验
    result = run_experiment(
        experiment_type=experiment_type,
        dataset_id=dataset_id,
        feature_set_id=feature_set_id,
        output_dir=output_dir,
        extra_args={json.dumps(parameters)}
    )
    
    print(f'实验完成，结果: {{result}}')
    
except Exception as e:
    print(f'实验执行失败: {{e}}')
    import traceback
    traceback.print_exc()
    sys.exit(1)
"

echo "=== EEG实验任务结束 ==="
echo "时间: $(date)"
echo "退出码: $?"
"""
        return script
    
    def _get_recording_ids_for_dataset(self, dataset_id: int) -> list:
        """获取数据集中的所有录音ID"""
        conn = sqlite3.connect(self.task_manager.db_path)
        c = conn.cursor()
        c.execute("SELECT id FROM recordings WHERE dataset_id = ?", (dataset_id,))
        recording_ids = [row[0] for row in c.fetchall()]
        conn.close()
        return recording_ids
    
    def monitor_pbs_tasks(self, interval: int = 60):
        """监控PBS任务状态"""
        print(f"开始监控PBS任务状态，间隔: {interval}秒")
        
        while True:
            try:
                # 获取运行中的任务
                running_tasks = self._get_running_pbs_tasks()
                
                for task in running_tasks:
                    task_id = task['id']
                    pbs_job_id = task['pbs_job_id']
                    
                    if pbs_job_id and pbs_job_id.startswith('batch_'):
                        # 批量任务，检查所有子任务
                        self._check_batch_task_status(task_id, pbs_job_id)
                    else:
                        # 单个任务，直接检查状态
                        status = self._check_pbs_job_status(pbs_job_id)
                        if status != task['status']:
                            self._update_task_status(task_id, status)
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("监控停止")
                break
            except Exception as e:
                print(f"监控出错: {e}")
                time.sleep(interval)
    
    def _get_running_pbs_tasks(self) -> list:
        """获取运行中的PBS任务"""
        conn = sqlite3.connect(self.task_manager.db_path)
        c = conn.cursor()
        
        c.execute("""
            SELECT id, pbs_job_id, status FROM tasks 
            WHERE status = 'running' AND pbs_job_id IS NOT NULL
        """)
        
        rows = c.fetchall()
        conn.close()
        
        return [{
            'id': row[0],
            'pbs_job_id': row[1],
            'status': row[2]
        } for row in rows]
    
    def _check_batch_task_status(self, task_id: int, batch_job_id: str):
        """检查批量任务状态"""
        # 这里需要实现批量任务的状态检查逻辑
        # 可以检查所有相关的PBS作业状态
        pass
    
    def _check_pbs_job_status(self, pbs_job_id: str) -> str:
        """检查PBS作业状态"""
        try:
            result = subprocess.run(['qstat', pbs_job_id], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                lines = result.stdout.strip().split('\n')
                if len(lines) >= 2:
                    status_line = lines[1]
                    if 'R' in status_line:
                        return 'running'
                    elif 'Q' in status_line:
                        return 'pending'
                    elif 'C' in status_line:
                        return 'completed'
                    elif 'E' in status_line:
                        return 'failed'
            
            return 'unknown'
            
        except Exception as e:
            print(f"检查任务状态失败 {pbs_job_id}: {e}")
            return 'unknown'
    
    def _update_task_status(self, task_id: int, status: str):
        """更新任务状态"""
        if status == 'completed':
            self.task_manager.update_task_status(task_id, TaskStatus.COMPLETED)
        elif status == 'failed':
            self.task_manager.update_task_status(task_id, TaskStatus.FAILED, 
                                               error_message="PBS作业执行失败")

# 全局实例
pbs_executor = None

def get_pbs_executor(task_manager: TaskManager) -> PBSExecutor:
    """获取PBS执行器实例"""
    global pbs_executor
    if pbs_executor is None:
        pbs_executor = PBSExecutor(task_manager)
    return pbs_executor 