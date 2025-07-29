#!/usr/bin/env python3
"""
PBS任务管理器命令行工具

简单的命令行界面来管理PBS任务
"""

import argparse
import sys
from datetime import datetime
from pbs_task_manager import pbs_manager

def show_queue_status():
    """显示队列状态"""
    print("=== PBS队列状态 ===")
    queues = pbs_manager.get_queue_status()
    
    if not queues:
        print("无法获取队列状态")
        return
    
    print(f"{'队列名称':<15} {'排队':<6} {'运行':<6} {'可用核心':<8} {'负载':<6}")
    print("-" * 50)
    
    for queue_name, queue_info in queues.items():
        print(f"{queue_name:<15} {queue_info.queued:<6} {queue_info.running:<6} "
              f"{queue_info.available_cores:<8} {queue_info.load_factor:<6.2f}")

def submit_feature_task(args):
    """提交特征提取任务"""
    print(f"提交特征提取任务:")
    print(f"  记录ID: {args.recording_id}")
    print(f"  特征集ID: {args.feature_set_id}")
    print(f"  数据集ID: {args.dataset_id}")
    
    # 选择最佳队列
    queue_name = pbs_manager.select_best_queue("feature_extraction")
    print(f"  选择队列: {queue_name}")
    
    # 提交任务
    pbs_job_id = pbs_manager.submit_task(
        task_type="feature_extraction",
        recording_id=args.recording_id,
        feature_set_id=args.feature_set_id,
        dataset_id=args.dataset_id,
        parameters={
            'priority': args.priority,
            'walltime': args.walltime
        }
    )
    
    if pbs_job_id:
        print(f"任务提交成功: {pbs_job_id}")
    else:
        print("任务提交失败")
        sys.exit(1)

def list_tasks(args):
    """列出任务"""
    if args.status:
        tasks = pbs_manager.get_pending_tasks() if args.status == 'pending' else pbs_manager.get_all_tasks()
    else:
        tasks = pbs_manager.get_all_tasks()
    
    if not tasks:
        print("没有找到任务")
        return
    
    print(f"=== PBS任务列表 ===")
    print(f"{'ID':<4} {'任务类型':<12} {'记录ID':<8} {'特征集ID':<8} {'PBS作业ID':<12} {'队列':<12} {'状态':<10} {'提交时间':<20}")
    print("-" * 100)
    
    for task in tasks:
        submitted_at = task['submitted_at']
        if isinstance(submitted_at, str):
            submitted_at = submitted_at[:19]  # 截取到秒
        
        print(f"{task['id']:<4} {task['task_type']:<12} {task['recording_id']:<8} "
              f"{task['feature_set_id']:<8} {task['pbs_job_id']:<12} {task['queue_name']:<12} "
              f"{task['status']:<10} {submitted_at:<20}")

def monitor_tasks(args):
    """监控任务状态"""
    print("开始监控任务状态...")
    print("按 Ctrl+C 停止监控")
    
    try:
        pbs_manager.monitor_tasks(interval=args.interval)
    except KeyboardInterrupt:
        print("\n监控已停止")

def show_task_details(args):
    """显示任务详情"""
    task_id = args.task_id
    
    # 从数据库获取任务详情
    conn = pbs_manager.db_path
    import sqlite3
    conn = sqlite3.connect(pbs_manager.db_path)
    c = conn.cursor()
    
    c.execute("SELECT * FROM pbs_tasks WHERE id = ?", (task_id,))
    row = c.fetchone()
    conn.close()
    
    if not row:
        print(f"任务 {task_id} 不存在")
        return
    
    print(f"=== 任务详情 (ID: {task_id}) ===")
    print(f"任务类型: {row[1]}")
    print(f"记录ID: {row[2]}")
    print(f"特征集ID: {row[3]}")
    print(f"数据集ID: {row[4]}")
    print(f"PBS作业ID: {row[5]}")
    print(f"队列: {row[6]}")
    print(f"状态: {row[7]}")
    print(f"提交时间: {row[8]}")
    print(f"开始时间: {row[9]}")
    print(f"完成时间: {row[10]}")
    
    if row[12]:  # parameters
        import json
        params = json.loads(row[12])
        print(f"参数: {params}")
    
    if row[13]:  # result
        print(f"结果: {row[13]}")
    
    if row[14]:  # error_message
        print(f"错误信息: {row[14]}")

def cancel_task(args):
    """取消任务"""
    task_id = args.task_id
    
    # 获取PBS作业ID
    conn = sqlite3.connect(pbs_manager.db_path)
    c = conn.cursor()
    c.execute("SELECT pbs_job_id FROM pbs_tasks WHERE id = ?", (task_id,))
    row = c.fetchone()
    conn.close()
    
    if not row or not row[0]:
        print(f"任务 {task_id} 不存在或没有PBS作业ID")
        return
    
    pbs_job_id = row[0]
    print(f"取消PBS作业: {pbs_job_id}")
    
    # 执行qdel命令
    import subprocess
    try:
        result = subprocess.run(['qdel', pbs_job_id], capture_output=True, text=True, timeout=10)
        if result.returncode == 0:
            print(f"任务 {task_id} 已取消")
            # 更新状态
            pbs_manager.update_task_status(pbs_job_id, 'cancelled')
        else:
            print(f"取消任务失败: {result.stderr}")
    except Exception as e:
        print(f"取消任务时出错: {e}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description='PBS任务管理器')
    subparsers = parser.add_subparsers(dest='command', help='可用命令')
    
    # 队列状态命令
    status_parser = subparsers.add_parser('status', help='显示队列状态')
    
    # 提交任务命令
    submit_parser = subparsers.add_parser('submit', help='提交特征提取任务')
    submit_parser.add_argument('recording_id', type=int, help='记录ID')
    submit_parser.add_argument('feature_set_id', type=int, help='特征集ID')
    submit_parser.add_argument('--dataset-id', type=int, help='数据集ID')
    submit_parser.add_argument('--priority', type=int, default=0, help='优先级')
    submit_parser.add_argument('--walltime', type=str, default='02:00:00', help='最大运行时间')
    
    # 列出任务命令
    list_parser = subparsers.add_parser('list', help='列出任务')
    list_parser.add_argument('--status', choices=['pending', 'running', 'completed', 'failed'], 
                            help='按状态筛选')
    
    # 监控命令
    monitor_parser = subparsers.add_parser('monitor', help='监控任务状态')
    monitor_parser.add_argument('--interval', type=int, default=60, help='监控间隔（秒）')
    
    # 任务详情命令
    detail_parser = subparsers.add_parser('detail', help='显示任务详情')
    detail_parser.add_argument('task_id', type=int, help='任务ID')
    
    # 取消任务命令
    cancel_parser = subparsers.add_parser('cancel', help='取消任务')
    cancel_parser.add_argument('task_id', type=int, help='任务ID')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    try:
        if args.command == 'status':
            show_queue_status()
        elif args.command == 'submit':
            submit_feature_task(args)
        elif args.command == 'list':
            list_tasks(args)
        elif args.command == 'monitor':
            monitor_tasks(args)
        elif args.command == 'detail':
            show_task_details(args)
        elif args.command == 'cancel':
            cancel_task(args)
    except KeyboardInterrupt:
        print("\n操作被用户中断")
        sys.exit(1)
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main() 