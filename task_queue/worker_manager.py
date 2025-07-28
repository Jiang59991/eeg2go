#!/usr/bin/env python3
"""
任务工作器管理脚本

用于启动、停止、重启和监控任务工作器进程。
"""

import os
import sys
import time
import signal
import subprocess
import psutil
import platform
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from logging_config import logger

class WorkerManager:
    def __init__(self, db_path="database/eeg2go.db"):
        self.db_path = db_path
        self.pid_file = "task_worker.pid"
        self.worker_script = "task_queue/worker_process.py"
        self.is_windows = platform.system() == "Windows"
    
    def is_running(self):
        """检查工作器是否正在运行"""
        if not os.path.exists(self.pid_file):
            return False
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # 检查进程是否存在
            if psutil.pid_exists(pid):
                process = psutil.Process(pid)
                # 检查是否是我们的工作器进程
                cmdline = " ".join(process.cmdline())
                if "worker_process.py" in cmdline:
                    return True
            
            # 进程不存在，删除PID文件
            os.remove(self.pid_file)
            return False
            
        except (ValueError, FileNotFoundError, psutil.NoSuchProcess):
            if os.path.exists(self.pid_file):
                os.remove(self.pid_file)
            return False
    
    def start(self):
        """启动工作器进程"""
        if self.is_running():
            logger.info("Task worker is already running")
            return True
        
        try:
            # 检查工作器脚本是否存在
            if not os.path.exists(self.worker_script):
                logger.error(f"Worker script not found: {self.worker_script}")
                return False
            
            # 启动工作器进程
            cmd = [
                sys.executable, 
                self.worker_script,
                "--db-path", self.db_path
            ]
            
            logger.info(f"Starting worker with command: {' '.join(cmd)}")
            
            # 根据操作系统选择不同的启动方式
            if self.is_windows:
                # Windows 系统：使用 CREATE_NEW_PROCESS_GROUP
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    cwd=os.getcwd(),  # 确保在正确的目录中运行
                    text=True  # 使用文本模式
                )
            else:
                # Unix/Linux 系统：使用 preexec_fn
                process = subprocess.Popen(
                    cmd,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    preexec_fn=os.setsid,
                    cwd=os.getcwd(),
                    text=True
                )
            
            logger.info(f"Worker process started with PID: {process.pid}")
            
            # 等待PID文件创建
            max_wait = 10  # 最多等待10秒
            for i in range(max_wait):
                if self.is_running():
                    logger.info("Task worker started successfully")
                    return True
                
                # 检查进程是否还在运行
                if process.poll() is not None:
                    # 进程已经退出，获取错误信息
                    stdout, stderr = process.communicate()
                    logger.error(f"Worker process exited after {i+1} seconds")
                    logger.error(f"STDOUT: {stdout}")
                    logger.error(f"STDERR: {stderr}")
                    return False
                
                logger.info(f"Waiting for PID file... ({i+1}/{max_wait})")
                time.sleep(1)
            
            # 如果超时，检查进程状态
            if process.poll() is None:
                logger.warning("Worker process is running but PID file not created after timeout")
                # 获取一些输出信息
                try:
                    stdout, stderr = process.communicate(timeout=2)
                    logger.info(f"STDOUT: {stdout}")
                    logger.info(f"STDERR: {stderr}")
                except subprocess.TimeoutExpired:
                    logger.info("Process is still running, no output available")
                
                # 强制杀死进程
                process.terminate()
                time.sleep(1)
                if process.poll() is None:
                    process.kill()
                
                return False
            else:
                stdout, stderr = process.communicate()
                logger.error(f"Worker process failed to start")
                logger.error(f"STDOUT: {stdout}")
                logger.error(f"STDERR: {stderr}")
                return False
                
        except Exception as e:
            logger.error(f"Error starting task worker: {e}")
            return False
    
    def stop(self):
        """停止工作器进程"""
        if not self.is_running():
            logger.info("Task worker is not running")
            return True
        
        try:
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            # 根据操作系统选择不同的停止方式
            if self.is_windows:
                # Windows 系统：发送 SIGTERM
                os.kill(pid, signal.SIGTERM)
            else:
                # Unix/Linux 系统：发送 SIGTERM
                os.kill(pid, signal.SIGTERM)
            
            # 等待进程结束
            for _ in range(10):  # 最多等待10秒
                if not self.is_running():
                    logger.info("Task worker stopped successfully")
                    return True
                time.sleep(1)
            
            # 如果进程还在运行，强制杀死
            if self.is_running():
                if self.is_windows:
                    os.kill(pid, signal.SIGKILL)
                else:
                    os.kill(pid, signal.SIGKILL)
                logger.info("Task worker force killed")
            
            return True
            
        except Exception as e:
            logger.error(f"Error stopping task worker: {e}")
            return False
    
    def restart(self):
        """重启工作器进程"""
        logger.info("Restarting task worker...")
        self.stop()
        time.sleep(2)
        return self.start()
    
    def status(self):
        """获取工作器状态"""
        if self.is_running():
            with open(self.pid_file, 'r') as f:
                pid = int(f.read().strip())
            
            process = psutil.Process(pid)
            return {
                'status': 'running',
                'pid': pid,
                'memory_usage': process.memory_info().rss,
                'cpu_percent': process.cpu_percent(),
                'create_time': datetime.fromtimestamp(process.create_time()).isoformat()
            }
        else:
            return {'status': 'stopped'}

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='Task Worker Manager')
    parser.add_argument('action', choices=['start', 'stop', 'restart', 'status'], 
                       help='Action to perform')
    parser.add_argument('--db-path', default='database/eeg2go.db', help='Database path')
    
    args = parser.parse_args()
    
    manager = WorkerManager(args.db_path)
    
    if args.action == 'start':
        success = manager.start()
        sys.exit(0 if success else 1)
    elif args.action == 'stop':
        success = manager.stop()
        sys.exit(0 if success else 1)
    elif args.action == 'restart':
        success = manager.restart()
        sys.exit(0 if success else 1)
    elif args.action == 'status':
        status = manager.status()
        print(f"Status: {status['status']}")
        if status['status'] == 'running':
            print(f"PID: {status['pid']}")
            print(f"Memory: {status['memory_usage'] / 1024 / 1024:.1f} MB")
            print(f"CPU: {status['cpu_percent']:.1f}%")
            print(f"Started: {status['create_time']}")
        sys.exit(0)

if __name__ == "__main__":
    main()


