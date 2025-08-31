#!/usr/bin/env python3
"""
测试简单本地模式
"""
import os
import sys
import time
from datetime import datetime

# 设置环境变量，启用本地模式
os.environ['USE_LOCAL_EXECUTOR'] = 'true'

# Add project root directory to Python path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from task_queue.models import TaskManager, Task
from logging_config import logger

def test_simple_local_mode():
    """测试简单本地模式"""
    print("=" * 60)
    print("测试简单本地模式")
    print("=" * 60)
    
    try:
        # 创建任务管理器
        task_manager = TaskManager()
        
        # 创建一个测试实验任务
        print("创建测试实验任务...")
        test_task = Task(
            task_type='experiment',
            parameters={
                'test_mode': True,
                'output_dir': 'test_experiments/test_simple_local_mode'
            },
            dataset_id=5,  # 使用数据集4
            feature_set_id=2,  # 使用特征集2
            experiment_type='correlation'
        )
        
        # 创建任务
        task_id = task_manager.create_task(test_task)
        print(f"任务已创建，ID: {task_id}")
        
        # 由于是同步执行，任务应该立即完成
        print("等待任务完成...")
        time.sleep(2)  # 给一点时间让任务完成
        
        # 检查任务状态
        task = task_manager.get_task(task_id)
        if task:
            status = task.get('status')
            progress = task.get('progress', 0)
            print(f"任务状态: {status}, 进度: {progress:.1f}%")
            
            if status in ['completed', 'failed']:
                print(f"任务完成，最终状态: {status}")
                if status == 'completed':
                    result = task.get('result', {})
                    print(f"任务结果: {result}")
                else:
                    error = task.get('error_message', 'Unknown error')
                    print(f"任务失败: {error}")
            else:
                print("任务仍在运行中...")
        
        # 清理
        print("清理测试文件...")
        import shutil
        if os.path.exists('test_experiments'):
            shutil.rmtree('test_experiments')
        
        print("测试完成")
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        print(f"测试失败: {e}")

if __name__ == "__main__":
    test_simple_local_mode()
