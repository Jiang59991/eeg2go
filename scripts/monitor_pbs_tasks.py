#!/usr/bin/env python3
"""
PBS任务监控脚本

监控PBS任务的运行状态，包括：
1. 排队状态
2. 运行状态
3. 完成状态
4. 失败状态
"""

import os
import sys
import time
import subprocess
from pathlib import Path
from datetime import datetime

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

class PBSTaskMonitor:
    """PBS任务监控器"""
    
    def __init__(self):
        from task_queue.models import TaskManager
        from task_queue.pbs_task_manager import pbs_manager
        
        self.task_manager = TaskManager()
        self.pbs_manager = pbs_manager
    
    def get_queue_status_summary(self):
        """获取队列状态摘要"""
        queues = self.pbs_manager.get_queue_status()
        
        print("=== 队列状态摘要 ===")
        print(f"{'队列名称':<15} {'排队':<6} {'运行':<6} {'可用':<6} {'负载':<8}")
        print("-" * 50)
        
        # 按排队任务数量排序
        sorted_queues = sorted(queues.items(), key=lambda x: x[1].queued)
        
        for name, queue in sorted_queues:
            print(f"{name:<15} {queue.queued:<6} {queue.running:<6} {queue.available_cores:<6} {queue.load_factor:<8.2f}")
        
        return queues
    
    def get_user_jobs(self):
        """获取当前用户的PBS作业"""
        try:
            result = subprocess.run(['qstat', '-u', os.getenv('USER', 'zj724')], 
                                  capture_output=True, text=True, timeout=10)
            
            if result.returncode == 0:
                return result.stdout
            else:
                return "无法获取用户作业信息"
        except Exception as e:
            return f"获取用户作业失败: {e}"
    
    def get_task_status(self):
        """获取任务状态"""
        from task_queue.models import TaskStatus
        
        # 获取所有任务
        all_tasks = self.task_manager.get_all_tasks()
        
        print("\n=== 任务状态摘要 ===")
        print(f"{'任务ID':<8} {'类型':<15} {'状态':<10} {'执行模式':<8} {'PBS作业ID':<12} {'队列':<12}")
        print("-" * 70)
        
        # 按创建时间倒序排列
        sorted_tasks = sorted(all_tasks, key=lambda x: x['created_at'], reverse=True)
        
        for task in sorted_tasks[:10]:  # 显示最近10个任务
            task_id = task['id']
            task_type = task['task_type']
            status = task['status']
            execution_mode = task.get('execution_mode', 'N/A')
            pbs_job_id = task.get('pbs_job_id', 'N/A')
            queue_name = task.get('queue_name', 'N/A')
            
            print(f"{task_id:<8} {task_type:<15} {status:<10} {execution_mode:<8} {pbs_job_id:<12} {queue_name:<12}")
        
        return all_tasks
    
    def monitor_specific_task(self, task_id: int):
        """监控特定任务"""
        task = self.task_manager.get_task(task_id)
        if not task:
            print(f"任务 {task_id} 不存在")
            return
        
        print(f"\n=== 监控任务 {task_id} ===")
        print(f"任务类型: {task['task_type']}")
        print(f"状态: {task['status']}")
        print(f"执行模式: {task.get('execution_mode', 'N/A')}")
        print(f"PBS作业ID: {task.get('pbs_job_id', 'N/A')}")
        print(f"队列: {task.get('queue_name', 'N/A')}")
        print(f"创建时间: {task['created_at']}")
        
        if task.get('pbs_job_id') and task.get('pbs_job_id') != 'N/A':
            pbs_job_id = task['pbs_job_id']
            
            # 检查PBS作业状态
            try:
                result = subprocess.run(['qstat', pbs_job_id], 
                                      capture_output=True, text=True, timeout=10)
                
                if result.returncode == 0:
                    print(f"\nPBS作业状态:")
                    print(result.stdout)
                    
                    # 解析状态
                    lines = result.stdout.strip().split('\n')
                    if len(lines) >= 2:
                        status_line = lines[1]
                        if 'Q' in status_line:
                            print("状态: 排队中")
                        elif 'R' in status_line:
                            print("状态: 运行中")
                        elif 'C' in status_line:
                            print("状态: 已完成")
                        elif 'E' in status_line:
                            print("状态: 出错")
                else:
                    print(f"PBS作业可能已完成或失败")
                    
            except Exception as e:
                print(f"检查PBS作业状态失败: {e}")
    
    def monitor_all_pbs_tasks(self):
        """监控所有PBS任务"""
        all_tasks = self.task_manager.get_all_tasks()
        pbs_tasks = [task for task in all_tasks if task.get('execution_mode') == 'pbs']
        
        if not pbs_tasks:
            print("没有PBS任务")
            return
        
        print(f"\n=== 监控 {len(pbs_tasks)} 个PBS任务 ===")
        
        for task in pbs_tasks:
            task_id = task['id']
            pbs_job_id = task.get('pbs_job_id')
            
            if pbs_job_id and pbs_job_id != 'N/A':
                print(f"\n任务 {task_id} (PBS作业: {pbs_job_id}):")
                
                try:
                    result = subprocess.run(['qstat', pbs_job_id], 
                                          capture_output=True, text=True, timeout=10)
                    
                    if result.returncode == 0:
                        lines = result.stdout.strip().split('\n')
                        if len(lines) >= 2:
                            status_line = lines[1]
                            if 'Q' in status_line:
                                print("  → 排队中")
                            elif 'R' in status_line:
                                print("  → 运行中")
                            elif 'C' in status_line:
                                print("  → 已完成")
                            elif 'E' in status_line:
                                print("  → 出错")
                    else:
                        print("  → 已完成或失败")
                        
                except Exception as e:
                    print(f"  → 检查失败: {e}")
    
    def continuous_monitoring(self, interval: int = 30):
        """持续监控"""
        print(f"开始持续监控，间隔: {interval}秒")
        print("按 Ctrl+C 停止监控")
        
        try:
            while True:
                # 清屏
                os.system('clear')
                
                print(f"=== PBS任务监控 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')} ===")
                
                # 显示队列状态
                self.get_queue_status_summary()
                
                # 显示用户作业
                print(f"\n=== 当前用户作业 ===")
                user_jobs = self.get_user_jobs()
                print(user_jobs)
                
                # 显示任务状态
                self.get_task_status()
                
                # 监控PBS任务
                self.monitor_all_pbs_tasks()
                
                print(f"\n下次更新: {interval}秒后...")
                time.sleep(interval)
                
        except KeyboardInterrupt:
            print("\n监控已停止")

def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description='PBS任务监控')
    parser.add_argument('--task-id', type=int, help='监控特定任务ID')
    parser.add_argument('--continuous', action='store_true', help='持续监控模式')
    parser.add_argument('--interval', type=int, default=30, help='监控间隔（秒）')
    
    args = parser.parse_args()
    
    monitor = PBSTaskMonitor()
    
    if args.task_id:
        # 监控特定任务
        monitor.monitor_specific_task(args.task_id)
    elif args.continuous:
        # 持续监控
        monitor.continuous_monitoring(args.interval)
    else:
        # 一次性监控
        monitor.get_queue_status_summary()
        monitor.get_task_status()
        monitor.monitor_all_pbs_tasks()

if __name__ == "__main__":
    main() 