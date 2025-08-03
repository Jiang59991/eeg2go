#!/usr/bin/env python3
"""
独立的任务工作器进程

这个模块运行在独立的进程中，负责处理后台任务。
它通过数据库与主Flask应用通信。
"""

import os
import sys
import time
import signal
import logging
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

# 获取Worker ID和PID文件名
worker_id = os.getenv('WORKER_ID', '1')
pid_file = os.getenv('WORKER_PID_FILE', f"task_worker_{worker_id}.pid")

# 立即写入PID文件，确保管理器知道进程已启动
try:
    with open(pid_file, 'w') as f:
        f.write(str(os.getpid()))
    print(f"Worker {worker_id} PID file written: {os.getpid()}")
except Exception as e:
    print(f"Failed to write PID file: {e}")
    sys.exit(1)

try:
    from task_queue.models import TaskManager, TaskStatus
    from task_queue.task_worker import TaskWorker
    from logging_config import logger
except ImportError as e:
    print(f"Import error: {e}")
    if os.path.exists(pid_file):
        os.remove(pid_file)
    sys.exit(1)

class WorkerProcess:
    def __init__(self, db_path="database/eeg2go.db"):
        self.db_path = db_path
        self.pid_file = pid_file
        self.worker_id = worker_id
        self.running = False
        
        logger.info(f"Initializing worker {self.worker_id} with db_path: {db_path}")
        
        try:
            logger.info("Creating TaskManager...")
            self.task_manager = TaskManager(db_path)
            logger.info("TaskManager created successfully")
            
            logger.info("Creating TaskWorker...")
            self.task_worker = TaskWorker(self.task_manager)
            logger.info("TaskWorker created successfully")
            
        except Exception as e:
            logger.error(f"Failed to initialize task manager: {e}")
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)
            raise
        
        # 设置信号处理
        signal.signal(signal.SIGINT, self.signal_handler)
        signal.signal(signal.SIGTERM, self.signal_handler)
    
    def signal_handler(self, signum, frame):
        """处理退出信号"""
        logger.info(f"Worker {self.worker_id} received signal {signum}, shutting down...")
        self.stop()
    
    def start(self):
        """启动工作器进程"""
        logger.info(f"Starting task worker {self.worker_id}...")
        
        try:
            logger.info(f"Worker {self.worker_id} PID file exists: {os.getpid()}")
            
            self.running = True
            logger.info("Starting task worker...")
            self.task_worker.start()
            logger.info("Task worker started successfully")
            
            logger.info(f"Task worker {self.worker_id} started with PID: {os.getpid()}")
            logger.info("Entering main loop...")
            
            while self.running:
                try:
                    # 检查是否有待处理的任务
                    logger.debug(f"Worker {self.worker_id} checking for pending tasks...")
                    pending_tasks = self.task_manager.get_pending_tasks()
                    
                    if pending_tasks:
                        logger.info(f"Worker {self.worker_id} found {len(pending_tasks)} pending tasks")
                        for task in pending_tasks:
                            if self.running:
                                # 直接调用 _process_task 方法
                                logger.info(f"Worker {self.worker_id} processing task {task['id']}")
                                self.task_worker._process_task(task)
                    else:
                        logger.debug(f"Worker {self.worker_id} no pending tasks found")
                    
                    # 休眠一段时间再检查
                    time.sleep(5)
                    
                except Exception as e:
                    logger.error(f"Worker {self.worker_id} error in main loop: {e}")
                    time.sleep(10)  # 出错后等待更长时间
                    
        except KeyboardInterrupt:
            logger.info(f"Worker {self.worker_id} received keyboard interrupt")
        except Exception as e:
            logger.error(f"Worker {self.worker_id} error: {e}")
            import traceback
            logger.error(f"Traceback: {traceback.format_exc()}")
        finally:
            self.stop()
    
    def stop(self):
        """停止工作器进程"""
        logger.info(f"Stopping task worker {self.worker_id}...")
        self.running = False
        
        if hasattr(self, 'task_worker') and self.task_worker:
            try:
                self.task_worker.stop()
            except Exception as e:
                logger.error(f"Error stopping task worker: {e}")
        
        # 删除PID文件
        if os.path.exists(self.pid_file):
            try:
                os.remove(self.pid_file)
                logger.info(f"Worker {self.worker_id} PID file removed")
            except Exception as e:
                logger.error(f"Error removing PID file: {e}")
        
        logger.info(f"Task worker {self.worker_id} stopped")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Task Worker Process')
    parser.add_argument('--db-path', default='database/eeg2go.db', help='Database path')
    
    args = parser.parse_args()
    
    logger.info(f"Worker {worker_id} main function started")
    logger.info(f"Arguments: {args}")
    
    try:
        # 前台运行
        logger.info("Creating WorkerProcess instance...")
        worker = WorkerProcess(args.db_path)
        logger.info("WorkerProcess instance created successfully")
        
        logger.info("Starting worker...")
        worker.start()
    except Exception as e:
        logger.error(f"Failed to start worker: {e}")
        import traceback
        logger.error(f"Traceback: {traceback.format_exc()}")
        # 确保PID文件被删除
        if os.path.exists(pid_file):
            os.remove(pid_file)
        sys.exit(1)

if __name__ == "__main__":
    main()
