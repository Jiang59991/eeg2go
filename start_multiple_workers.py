#!/usr/bin/env python3
"""
启动多个Worker进程的脚本
"""
import subprocess
import time
import os
import sys
from pathlib import Path

def start_worker(worker_id, db_path="database/eeg2go.db"):
    """启动单个worker进程"""
    cmd = [
        sys.executable,
        "task_queue/worker_process.py",
        "--db-path", db_path
    ]
    
    # 设置环境变量以区分不同的worker
    env = os.environ.copy()
    env['WORKER_ID'] = str(worker_id)
    env['WORKER_PID_FILE'] = f"task_worker_{worker_id}.pid"
    
    print(f"启动Worker {worker_id}...")
    
    # 在Windows上使用CREATE_NEW_PROCESS_GROUP
    if os.name == 'nt':
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            creationflags=subprocess.CREATE_NEW_PROCESS_GROUP,
            text=True
        )
    else:
        process = subprocess.Popen(
            cmd,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
    
    return process

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='启动多个Worker进程')
    parser.add_argument('--workers', type=int, default=2, help='Worker进程数量')
    parser.add_argument('--db-path', default='database/eeg2go.db', help='数据库路径')
    
    args = parser.parse_args()
    
    print(f"启动 {args.workers} 个Worker进程...")
    
    processes = []
    
    try:
        # 启动多个worker
        for i in range(args.workers):
            process = start_worker(i + 1, args.db_path)
            processes.append(process)
            time.sleep(2)  # 间隔2秒启动下一个
        
        print(f"已启动 {len(processes)} 个Worker进程")
        print("按Ctrl+C停止所有Worker...")
        
        # 等待用户中断
        try:
            while True:
                time.sleep(1)
                # 检查进程是否还在运行
                for i, process in enumerate(processes):
                    if process.poll() is not None:
                        print(f"Worker {i+1} 已退出")
                        
        except KeyboardInterrupt:
            print("\n正在停止所有Worker...")
            
    finally:
        # 停止所有进程
        for i, process in enumerate(processes):
            if process.poll() is None:
                print(f"停止Worker {i+1}...")
                process.terminate()
                try:
                    process.wait(timeout=5)
                except subprocess.TimeoutExpired:
                    process.kill()
        
        print("所有Worker已停止")

if __name__ == "__main__":
    main() 