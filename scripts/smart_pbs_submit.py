#!/usr/bin/env python3
"""
智能PBS任务提交脚本

智能选择最空闲的队列提交任务，并监控任务状态
"""

import os
import sys
import time
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

def find_best_queue():
    """找到最空闲的队列"""
    from task_queue.pbs_task_manager import pbs_manager
    
    queues = pbs_manager.get_queue_status()
    
    if not queues:
        return "v1_small24"
    
    # 按排队任务数量排序，选择最空闲的队列
    sorted_queues = sorted(queues.items(), key=lambda x: x[1].queued)
    
    print("=== 队列状态（按空闲程度排序）===")
    print(f"{'队列名称':<15} {'排队':<6} {'运行':<6} {'可用':<6} {'负载':<8}")
    print("-" * 50)
    
    for name, queue in sorted_queues[:10]:  # 显示前10个最空闲的队列
        print(f"{name:<15} {queue.queued:<6} {queue.running:<6} {queue.available_cores:<6} {queue.load_factor:<8.2f}")
    
    # 选择排队最少的队列
    best_queue_name = sorted_queues[0][0]
    best_queue = sorted_queues[0][1]
    
    print(f"\n✅ 推荐队列: {best_queue_name}")
    print(f"   排队任务: {best_queue.queued}")
    print(f"   运行任务: {best_queue.running}")
    print(f"   可用核心: {best_queue.available_cores}")
    print(f"   负载因子: {best_queue.load_factor:.2f}")
    
    return best_queue_name

def submit_smart_task(dataset_id: int, feature_set_id: int, test_mode: bool = True):
    """智能提交任务"""
    print("=== 智能PBS任务提交 ===")
    
    # 设置PBS执行模式
    os.environ['EEG2GO_EXECUTION_MODE'] = 'pbs'
    
    try:
        from web.api.task_api import task_manager
        from task_queue.models import Task
        from task_queue.task_worker import TaskWorker
        
        # 找到最佳队列
        best_queue = find_best_queue()
        
        # 创建任务
        task = Task(
            task_type="feature_extraction",
            parameters={
                'dataset_id': dataset_id,
                'feature_set_id': feature_set_id,
                'test': test_mode,
                'execution_mode': 'pbs',
                'preferred_queue': best_queue  # 添加首选队列信息
            },
            dataset_id=dataset_id,
            feature_set_id=feature_set_id,
            execution_mode='pbs'
        )
        
        task_id = task_manager.create_task(task)
        print(f"✅ 任务创建成功: ID={task_id}")
        
        # 处理任务
        task_worker = TaskWorker(task_manager)
        task_info = task_manager.get_task(task_id)
        
        print("开始处理PBS任务...")
        task_worker._process_task(task_info)
        
        # 检查结果
        updated_task = task_manager.get_task(task_id)
        print(f"✅ 任务状态: {updated_task['status']}")
        print(f"✅ PBS作业ID: {updated_task.get('pbs_job_id')}")
        print(f"✅ PBS队列: {updated_task.get('queue_name')}")
        
        if updated_task['status'] == 'running' and updated_task.get('pbs_job_id'):
            print("✅ PBS任务提交成功！")
            return task_id, updated_task['pbs_job_id']
        else:
            print(f"❌ PBS任务提交失败: {updated_task.get('error_message', 'Unknown error')}")
            return task_id, None
        
    except Exception as e:
        print(f"❌ 智能任务提交失败: {e}")
        import traceback
        traceback.print_exc()
        return None, None

def monitor_task_progress(task_id: int, pbs_job_id: str = None):
    """监控任务进度"""
    print(f"\n=== 监控任务 {task_id} 进度 ===")
    
    from task_queue.models import TaskManager
    import subprocess
    
    task_manager = TaskManager()
    
    try:
        while True:
            # 获取任务状态
            task = task_manager.get_task(task_id)
            if not task:
                print("任务不存在")
                break
            
            print(f"\n时间: {time.strftime('%H:%M:%S')}")
            print(f"任务状态: {task['status']}")
            
            if pbs_job_id:
                # 检查PBS作业状态
                try:
                    result = subprocess.run(['qstat', pbs_job_id], 
                                          capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if len(lines) >= 2:
                            status_line = lines[1]
                            if 'Q' in status_line:
                                print("PBS状态: 排队中")
                            elif 'R' in status_line:
                                print("PBS状态: 运行中")
                            elif 'C' in status_line:
                                print("PBS状态: 已完成")
                                break
                            elif 'E' in status_line:
                                print("PBS状态: 出错")
                                break
                    else:
                        print("PBS作业已完成或失败")
                        break
                        
                except Exception as e:
                    print(f"检查PBS状态失败: {e}")
            
            # 检查任务是否完成
            if task['status'] in ['completed', 'failed']:
                print(f"任务最终状态: {task['status']}")
                if task.get('error_message'):
                    print(f"错误信息: {task['error_message']}")
                break
            
            # 等待30秒再检查
            print("等待30秒...")
            time.sleep(30)
            
    except KeyboardInterrupt:
        print("\n监控已停止")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='智能PBS任务提交')
    parser.add_argument('--dataset-id', type=int, default=1, help='数据集ID')
    parser.add_argument('--feature-set-id', type=int, default=1, help='特征集ID')
    parser.add_argument('--test', action='store_true', help='测试模式')
    parser.add_argument('--monitor', action='store_true', help='监控任务进度')
    
    args = parser.parse_args()
    
    print("智能PBS任务提交系统")
    print("=" * 50)
    
    # 提交任务
    task_id, pbs_job_id = submit_smart_task(
        dataset_id=args.dataset_id,
        feature_set_id=args.feature_set_id,
        test_mode=args.test
    )
    
    if task_id and args.monitor:
        # 监控任务进度
        monitor_task_progress(task_id, pbs_job_id)
    
    if task_id:
        print(f"\n🎉 任务提交完成！")
        print(f"任务ID: {task_id}")
        if pbs_job_id:
            print(f"PBS作业ID: {pbs_job_id}")
        print(f"\n使用以下命令监控任务:")
        print(f"python3 scripts/monitor_pbs_tasks.py --task-id {task_id}")
        print(f"python3 scripts/monitor_pbs_tasks.py --continuous")
    else:
        print("\n❌ 任务提交失败")

if __name__ == "__main__":
    main() 