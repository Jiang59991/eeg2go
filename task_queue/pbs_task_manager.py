#!/usr/bin/env python3
"""
PBS任务管理器 - 简单的资源感知调度

功能：
1. 根据 qstat -q 动态选择合适的队列
2. 提交PBS任务并记录任务信息
3. 跟踪任务状态
"""

import os
import sys
import subprocess
import sqlite3
import json
import time
from datetime import datetime
from typing import Dict, List, Optional, Tuple
from pathlib import Path

class PBSQueueInfo:
    """PBS队列信息"""
    def __init__(self, name: str, queued: int = 0, running: int = 0, max_cores: int = 0):
        self.name = name
        self.queued = queued
        self.running = running
        self.max_cores = max_cores
        self.available_cores = max_cores - running
    
    @property
    def load_factor(self) -> float:
        """计算负载因子"""
        if self.max_cores == 0:
            return float('inf')
        return (self.queued + self.running) / self.max_cores
    
    def __str__(self):
        return f"{self.name}: 排队={self.queued}, 运行={self.running}, 可用={self.available_cores}, 负载={self.load_factor:.2f}"

class PBSTaskManager:
    """PBS任务管理器"""
    
    def __init__(self, db_path: str = "database/pbs_tasks.db"):
        self.db_path = db_path
        self.template_dir = Path("pbs_templates")
        self.init_database()
    
    def init_database(self):
        """初始化数据库"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # 创建PBS任务表
        c.execute("""
            CREATE TABLE IF NOT EXISTS pbs_tasks (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                task_type TEXT NOT NULL,
                recording_id INTEGER,
                feature_set_id INTEGER,
                dataset_id INTEGER,
                pbs_job_id TEXT,
                queue_name TEXT,
                status TEXT DEFAULT 'pending',
                submitted_at TIMESTAMP,
                started_at TIMESTAMP,
                completed_at TIMESTAMP,
                parameters TEXT,
                result TEXT,
                error_message TEXT,
                log_file TEXT
            )
        """)
        
        # 创建队列状态表
        c.execute("""
            CREATE TABLE IF NOT EXISTS queue_status (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                queue_name TEXT NOT NULL,
                queued_jobs INTEGER,
                running_jobs INTEGER,
                available_cores INTEGER,
                recorded_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
            )
        """)
        
        conn.commit()
        conn.close()
        print(f"数据库初始化完成: {self.db_path}")
    
    def get_queue_status(self) -> Dict[str, PBSQueueInfo]:
        """获取队列状态"""
        try:
            # 执行 qstat -q 命令
            result = subprocess.run(['qstat', '-q'], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode != 0:
                print(f"qstat命令失败: {result.stderr}")
                return self._get_default_queues()
            
            return self._parse_qstat_output(result.stdout)
            
        except Exception as e:
            print(f"获取队列状态失败: {e}")
            return self._get_default_queues()
    
    def _parse_qstat_output(self, output: str) -> Dict[str, PBSQueueInfo]:
        """解析 qstat -q 输出"""
        queues = {}
        lines = output.strip().split('\n')
        
        for line in lines:
            if line.startswith('Queue') or '----' in line:
                continue
            
            parts = line.split()
            if len(parts) >= 4:
                queue_name = parts[0]
                try:
                    queued = int(parts[2]) if parts[2].isdigit() else 0
                    running = int(parts[3]) if parts[3].isdigit() else 0
                    
                    # 根据队列名称估算最大核心数
                    max_cores = self._estimate_queue_cores(queue_name)
                    
                    queues[queue_name] = PBSQueueInfo(
                        name=queue_name,
                        queued=queued,
                        running=running,
                        max_cores=max_cores
                    )
                except (ValueError, IndexError):
                    continue
        
        return queues
    
    def _estimate_queue_cores(self, queue_name: str) -> int:
        """根据队列名称估算核心数"""
        # 这里可以根据实际的队列配置进行调整
        if 'small' in queue_name:
            return 24
        elif 'medium' in queue_name:
            return 48
        elif 'large' in queue_name:
            return 72
        else:
            return 32  # 默认值
    
    def _get_default_queues(self) -> Dict[str, PBSQueueInfo]:
        """获取默认队列信息（当qstat不可用时）"""
        return {
            'v1_small24': PBSQueueInfo('v1_small24', 0, 0, 24),
            'v1_medium48': PBSQueueInfo('v1_medium48', 0, 0, 48),
            'v1_large72': PBSQueueInfo('v1_large72', 0, 0, 72)
        }
    
    def select_best_queue(self, task_type: str = "feature_extraction", preferred_queue: str = None) -> str:
        """选择最佳队列 - 优先选择最空闲的队列"""
        queues = self.get_queue_status()
        
        if not queues:
            return "v1_small24"  # 默认队列
        
        # 如果有首选队列且该队列可用，优先使用
        if preferred_queue and preferred_queue in queues:
            queue = queues[preferred_queue]
            if queue.queued < 100:  # 排队任务不太多
                print(f"使用首选队列: {preferred_queue}")
                return preferred_queue
        
        # 过滤掉负载过高的队列（排队任务超过50个）
        available_queues = {name: queue for name, queue in queues.items() 
                           if queue.queued < 50 and queue.load_factor < 5.0}
        
        if not available_queues:
            # 如果没有合适的队列，选择排队最少的队列
            best_queue = min(queues.values(), key=lambda q: q.queued)
            print(f"所有队列都较忙，选择排队最少的队列: {best_queue}")
            return best_queue.name
        
        # 优先选择排队任务最少的队列
        best_queue = min(available_queues.values(), key=lambda q: q.queued)
        print(f"选择最空闲队列: {best_queue}")
        return best_queue.name
    
    def submit_task(self, task_type: str, recording_id: int, 
                   feature_set_id: int, dataset_id: int = None,
                   parameters: Dict = None, preferred_queue: str = None) -> Optional[str]:
        """提交PBS任务"""
        try:
            # 选择最佳队列
            queue_name = self.select_best_queue(task_type, preferred_queue)
            
            # 准备PBS脚本
            script_content = self._prepare_pbs_script(
                task_type, recording_id, feature_set_id, dataset_id, queue_name
            )
            
            # 创建临时脚本文件
            script_file = f"temp_submit_{recording_id}_{feature_set_id}.pbs"
            with open(script_file, 'w') as f:
                f.write(script_content)
            
            # 提交任务
            cmd = ['qsub', script_file]
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
            
            if result.returncode == 0:
                pbs_job_id = result.stdout.strip()
                print(f"任务提交成功: {pbs_job_id}")
                
                # 记录到数据库
                self._record_task(task_type, recording_id, feature_set_id, 
                                dataset_id, pbs_job_id, queue_name, parameters)
                
                # 清理临时文件
                os.remove(script_file)
                
                return pbs_job_id
            else:
                print(f"任务提交失败: {result.stderr}")
                return None
                
        except Exception as e:
            print(f"提交任务时出错: {e}")
            return None
    
    def _prepare_pbs_script(self, task_type: str, recording_id: int,
                           feature_set_id: int, dataset_id: int, queue_name: str) -> str:
        """准备PBS脚本内容"""
        # 读取模板文件
        template_file = self.template_dir / "submit_feature.pbs"
        
        if template_file.exists():
            with open(template_file, 'r') as f:
                template = f.read()
        else:
            # 如果模板文件不存在，使用默认模板
            template = self._get_default_template()
        
        # 替换变量
        script = template.replace("${RECORDING_ID}", str(recording_id))
        script = script.replace("${FEATURE_SET_ID}", str(feature_set_id))
        script = script.replace("${DATASET_ID}", str(dataset_id or 0))
        
        # 更新队列名称
        script = script.replace("#PBS -q v1_small24", f"#PBS -q {queue_name}")
        
        return script
    
    def _get_default_template(self) -> str:
        """获取默认PBS模板"""
        return '''#!/bin/bash
#PBS -N eeg_feature_extraction
#PBS -l nodes=1:ppn=1
#PBS -l walltime=01:00:00
#PBS -q v1_small24
#PBS -j oe
#PBS -o logs/feature_extraction_${RECORDING_ID}.log

cd $PBS_O_WORKDIR
export RECORDING_ID=${RECORDING_ID}
export FEATURE_SET_ID=${FEATURE_SET_ID}
export DATASET_ID=${DATASET_ID}

echo "开始处理记录 ${RECORDING_ID}"
python3 -c "
import sys
import os
sys.path.append('/rds/general/user/zj724/home/eeg2go')
from eeg2fx.featureset_fetcher import run_feature_set

try:
    results = run_feature_set(${FEATURE_SET_ID}, ${RECORDING_ID})
    print(f'特征提取完成: {len(results)} 个特征')
except Exception as e:
    print(f'特征提取失败: {e}')
    sys.exit(1)
"
'''
    
    def _record_task(self, task_type: str, recording_id: int, feature_set_id: int,
                    dataset_id: int, pbs_job_id: str, queue_name: str, parameters: Dict):
        """记录任务到数据库"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            INSERT INTO pbs_tasks (task_type, recording_id, feature_set_id, dataset_id,
                                  pbs_job_id, queue_name, status, submitted_at, parameters)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            task_type, recording_id, feature_set_id, dataset_id,
            pbs_job_id, queue_name, 'pending', datetime.now(),
            json.dumps(parameters) if parameters else None
        ))
        
        conn.commit()
        conn.close()
    
    def update_task_status(self, pbs_job_id: str, status: str, 
                          result: str = None, error_message: str = None):
        """更新任务状态"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        if status == 'running':
            c.execute("""
                UPDATE pbs_tasks SET status = ?, started_at = ? WHERE pbs_job_id = ?
            """, (status, datetime.now(), pbs_job_id))
        elif status in ['completed', 'failed']:
            c.execute("""
                UPDATE pbs_tasks SET status = ?, completed_at = ?, result = ?, error_message = ? 
                WHERE pbs_job_id = ?
            """, (status, datetime.now(), result, error_message, pbs_job_id))
        
        conn.commit()
        conn.close()
    
    def get_pending_tasks(self) -> List[Dict]:
        """获取待处理任务"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            SELECT * FROM pbs_tasks WHERE status = 'pending' ORDER BY submitted_at ASC
        """)
        
        rows = c.fetchall()
        conn.close()
        
        return [{
            'id': row[0],
            'task_type': row[1],
            'recording_id': row[2],
            'feature_set_id': row[3],
            'dataset_id': row[4],
            'pbs_job_id': row[5],
            'queue_name': row[6],
            'status': row[7],
            'submitted_at': row[8],
            'parameters': json.loads(row[12]) if row[12] else None
        } for row in rows]
    
    def get_all_tasks(self) -> List[Dict]:
        """获取所有任务"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("SELECT * FROM pbs_tasks ORDER BY submitted_at DESC")
        rows = c.fetchall()
        conn.close()
        
        return [{
            'id': row[0],
            'task_type': row[1],
            'recording_id': row[2],
            'feature_set_id': row[3],
            'dataset_id': row[4],
            'pbs_job_id': row[5],
            'queue_name': row[6],
            'status': row[7],
            'submitted_at': row[8],
            'started_at': row[9],
            'completed_at': row[10],
            'parameters': json.loads(row[12]) if row[12] else None,
            'result': row[13],
            'error_message': row[14]
        } for row in rows]
    
    def monitor_tasks(self, interval: int = 60):
        """监控任务状态"""
        print(f"开始监控任务状态，间隔: {interval}秒")
        
        while True:
            try:
                # 获取待处理任务
                pending_tasks = self.get_pending_tasks()
                
                for task in pending_tasks:
                    pbs_job_id = task['pbs_job_id']
                    if pbs_job_id:
                        # 检查PBS任务状态
                        status = self._check_pbs_job_status(pbs_job_id)
                        if status != task['status']:
                            self.update_task_status(pbs_job_id, status)
                            print(f"任务 {pbs_job_id} 状态更新: {task['status']} -> {status}")
                
                # 记录队列状态
                self._record_queue_status()
                
                time.sleep(interval)
                
            except KeyboardInterrupt:
                print("监控停止")
                break
            except Exception as e:
                print(f"监控出错: {e}")
                time.sleep(interval)
    
    def _check_pbs_job_status(self, pbs_job_id: str) -> str:
        """检查PBS任务状态"""
        try:
            result = subprocess.run(['qstat', pbs_job_id], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                # 解析状态
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
    
    def _record_queue_status(self):
        """记录队列状态"""
        queues = self.get_queue_status()
        
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        for queue_name, queue_info in queues.items():
            c.execute("""
                INSERT INTO queue_status (queue_name, queued_jobs, running_jobs, available_cores)
                VALUES (?, ?, ?, ?)
            """, (queue_name, queue_info.queued, queue_info.running, queue_info.available_cores))
        
        conn.commit()
        conn.close()

# 全局实例
pbs_manager = PBSTaskManager() 