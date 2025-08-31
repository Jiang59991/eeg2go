#!/usr/bin/env python3
"""
切换到本地模式的脚本
"""

import os
import sys

def switch_to_local_mode():
    """切换到本地模式"""
    print("正在切换到本地模式...")
    
    # 设置环境变量
    os.environ['USE_LOCAL_EXECUTOR'] = 'true'
    os.environ['LOCAL_EXECUTOR_WORKERS'] = '1'  # 本地模式使用单线程
    
    print("环境变量已设置:")
    print(f"  USE_LOCAL_EXECUTOR = {os.environ.get('USE_LOCAL_EXECUTOR', 'not set')}")
    print(f"  LOCAL_EXECUTOR_WORKERS = {os.environ.get('LOCAL_EXECUTOR_WORKERS', 'not set')}")
    
    # 启动本地系统
    print("\n启动本地模式系统...")
    os.system('python start_local_system.py')

def switch_to_celery_mode():
    """切换到Celery模式"""
    print("正在切换到Celery模式...")
    
    # 清除环境变量
    if 'USE_LOCAL_EXECUTOR' in os.environ:
        del os.environ['USE_LOCAL_EXECUTOR']
    if 'LOCAL_EXECUTOR_WORKERS' in os.environ:
        del os.environ['LOCAL_EXECUTOR_WORKERS']
    
    print("环境变量已清除，将使用默认的Celery模式")
    print("请手动启动Celery worker和web应用")

def show_current_mode():
    """显示当前模式"""
    use_local = os.environ.get('USE_LOCAL_EXECUTOR', 'false').lower() == 'true'
    mode = "本地模式" if use_local else "Celery模式"
    workers = os.environ.get('LOCAL_EXECUTOR_WORKERS', 'N/A')
    
    print(f"当前系统模式: {mode}")
    print(f"本地执行器工作线程数: {workers}")

if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("用法:")
        print("  python switch_to_local_mode.py local    # 切换到本地模式")
        print("  python switch_to_local_mode.py celery   # 切换到Celery模式")
        print("  python switch_to_local_mode.py status   # 显示当前模式")
        sys.exit(1)
    
    command = sys.argv[1].lower()
    
    if command == 'local':
        switch_to_local_mode()
    elif command == 'celery':
        switch_to_celery_mode()
    elif command == 'status':
        show_current_mode()
    else:
        print(f"未知命令: {command}")
        sys.exit(1)
