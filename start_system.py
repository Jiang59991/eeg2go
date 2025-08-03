#!/usr/bin/env python3
"""
EEG2Go Worker进程启动脚本

只启动Worker进程，Web应用需要单独启动
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
        """启动Worker进程"""
        print(f"启动 {self.workers} 个Worker进程...")
        
        for i in range(self.workers):
            worker_id = i + 1
            env = os.environ.copy()
            env['WORKER_ID'] = str(worker_id)
            env['WORKER_PID_FILE'] = f"task_worker_{worker_id}.pid"
            
            cmd = [
                sys.executable,
                "task_queue/worker_process.py",
                "--db-path", "database/eeg2go.db"
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
            
            self.worker_processes.append(process)
            print(f"Worker {worker_id} 已启动，PID: {process.pid}")
            time.sleep(2)  # 间隔启动
    
    def start(self):
        """启动Worker进程"""
        print("=" * 50)
        print("        EEG2Go Worker进程启动")
        print("=" * 50)
        
        try:
            # 启动Worker进程
            self.start_workers()
            
            self.running = True
            print("\n✅ Worker进程启动完成！")
            print("请手动启动Web应用: python web/scripts/start_web_interface.py")
            print("按Ctrl+C停止Worker进程...")
            
            # 等待用户中断
            while self.running:
                time.sleep(1)
                
                # 检查进程状态
                for i, process in enumerate(self.worker_processes):
                    if process.poll() is not None:
                        print(f"❌ Worker {i+1} 已退出")
            
        except KeyboardInterrupt:
            print("\n正在停止Worker进程...")
        finally:
            self.stop()
    
    def stop(self):
        """停止Worker进程"""
        print("停止所有Worker进程...")
        
        # 停止Worker进程
        for i, process in enumerate(self.worker_processes):
            if process.poll() is None:
                print(f"停止Worker {i+1}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        # 清理PID文件
        for i in range(self.workers):
            pid_file = f"task_worker_{i+1}.pid"
            if os.path.exists(pid_file):
                try:
                    os.remove(pid_file)
                except:
                    pass
        
        print("✅ Worker进程已停止")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='EEG2Go Worker进程启动脚本')
    parser.add_argument('--workers', type=int, default=2, help='Worker进程数量')
    
    args = parser.parse_args()
    
    manager = SystemManager(args.workers)
    manager.start()

if __name__ == "__main__":
    main() 