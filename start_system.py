#!/usr/bin/env python3
"""
EEG2Go Celery Worker进程启动脚本

启动Celery Worker进程，Web应用需要单独启动
"""
import subprocess
import time
import os
import sys
import signal
import threading
from pathlib import Path

class SystemManager:
    def __init__(self, workers=2):
        self.workers = workers
        self.worker_processes = []
        self.running = False
        
    def start_workers(self):
        """启动Celery Worker进程"""
        print(f"启动 {self.workers} 个Celery Worker进程...")
        
        # 启动不同类型的worker
        worker_configs = [
            {'queue': 'feature_extraction', 'concurrency': 2, 'name': 'feature_worker'},
            {'queue': 'experiments', 'concurrency': 1, 'name': 'experiment_worker'},
            {'queue': 'default', 'concurrency': 1, 'name': 'default_worker'}
        ]
        
        for i, config in enumerate(worker_configs[:self.workers]):
            env = os.environ.copy()
            env['DATABASE_PATH'] = 'database/eeg2go.db'
            
            cmd = [
                sys.executable,
                "task_queue/celery_worker.py",
                "--queue", config['queue'],
                "--concurrency", str(config['concurrency']),
                "--hostname", f"{config['name']}@{os.uname().nodename if hasattr(os, 'uname') else 'localhost'}"
            ]
            
            if os.name == 'nt':  # Windows
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
                    text=True
                )
            else:  # Unix/Linux
                process = subprocess.Popen(
                    cmd,
                    env=env,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.PIPE,
                    text=True
                )
            
            self.worker_processes.append({
                'process': process,
                'config': config,
                'id': i + 1
            })
            
            print(f"Worker {i+1} ({config['queue']}) 已启动，PID: {process.pid}")
            time.sleep(2)  # 间隔启动
    
    def start(self):
        """启动Celery Worker进程"""
        print("=" * 50)
        print("        EEG2Go Celery Worker进程启动")
        print("=" * 50)
        
        try:
            # 启动Worker进程
            self.start_workers()
            
            self.running = True
            print("\n✅ Celery Worker进程启动完成！")
            print("请手动启动Web应用: python web/scripts/start_web_interface.py")
            print("按Ctrl+C停止Worker进程...")
            
            # 等待用户中断
            while self.running:
                time.sleep(1)
                
                # 检查进程状态
                for worker_info in self.worker_processes:
                    process = worker_info['process']
                    config = worker_info['config']
                    worker_id = worker_info['id']
                    
                    if process.poll() is not None:
                        print(f"❌ Worker {worker_id} ({config['queue']}) 已退出")
            
        except KeyboardInterrupt:
            print("\n正在停止Celery Worker进程...")
        finally:
            self.stop()
    
    def stop(self):
        """停止Celery Worker进程"""
        print("停止所有Celery Worker进程...")
        
        # 停止Worker进程
        for worker_info in self.worker_processes:
            process = worker_info['process']
            config = worker_info['config']
            worker_id = worker_info['id']
            
            if process.poll() is None:
                print(f"停止Worker {worker_id} ({config['queue']})...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        print("✅ Celery Worker进程已停止")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EEG2Go Celery Worker进程启动脚本')
    parser.add_argument('--workers', type=int, default=2, help='Worker进程数量')
    
    args = parser.parse_args()
    
    manager = SystemManager(args.workers)
    manager.start()

if __name__ == "__main__":
    main() 