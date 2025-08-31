#!/usr/bin/env python3
"""
启动本地模式系统
不使用Redis，全部在本机运行完成
"""
import os
import sys
import signal
import time
from datetime import datetime

# 设置环境变量，启用本地模式
os.environ['USE_LOCAL_EXECUTOR'] = 'true'

# Add project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from logging_config import logger
from task_queue.local_executor import get_local_executor, shutdown_local_executor

def signal_handler(signum, frame):
    """信号处理器，用于优雅关闭"""
    logger.info(f"Received signal {signum}, shutting down...")
    shutdown_local_executor()
    sys.exit(0)

def main():
    """主函数"""
    print("=" * 60)
    print("EEG2Go 本地模式系统启动")
    print("=" * 60)
    print(f"启动时间: {datetime.now()}")
    print("模式: 本地执行器 (不使用Redis)")
    print("=" * 60)
    
    # 设置信号处理器
    signal.signal(signal.SIGINT, signal_handler)
    signal.signal(signal.SIGTERM, signal_handler)
    
    try:
        # 初始化本地执行器
        logger.info("Initializing local executor...")
        local_executor = get_local_executor()
        
        print("本地执行器已启动")
        print("系统已准备就绪，可以接收任务")
        print("按 Ctrl+C 退出")
        print("-" * 60)
        
        # 保持系统运行
        while True:
            time.sleep(1)
            
    except KeyboardInterrupt:
        logger.info("Received keyboard interrupt")
    except Exception as e:
        logger.error(f"System error: {e}")
    finally:
        logger.info("Shutting down local system...")
        shutdown_local_executor()
        print("系统已关闭")

if __name__ == "__main__":
    main()

