#!/usr/bin/env python3
import os
import sys
import subprocess
import sqlite3
from datetime import datetime
from typing import Dict, List

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from logging_config import logger

DB_PATH = os.path.abspath(os.path.join("database", "eeg2go.db"))


def check_job_status(job_id: str = None) -> Dict:
    """
    检查任务状态
    """
    try:
        if job_id:
            # 检查特定任务
            result = subprocess.run(["qstat", job_id], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                  text=True, timeout=10)
        else:
            # 检查所有任务
            result = subprocess.run(["qstat"], 
                                  stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                  text=True, timeout=10)
        
        if result.returncode == 0:
            return {"success": True, "output": result.stdout.strip()}
        else:
            return {"success": False, "error": result.stderr.strip()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def check_array_job_details(job_id: str) -> Dict:
    """
    检查array job的详细信息
    """
    try:
        # 检查array job的详细信息
        result = subprocess.run(["qstat", "-f", job_id], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              text=True, timeout=10)
        
        if result.returncode == 0:
            return {"success": True, "output": result.stdout.strip()}
        else:
            return {"success": False, "error": result.stderr.strip()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def check_array_job_subtasks(job_id: str) -> Dict:
    """
    检查array job的子任务状态
    """
    try:
        # 检查array job的子任务
        result = subprocess.run(["qstat", "-t", job_id], 
                              stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                              text=True, timeout=10)
        
        if result.returncode == 0:
            return {"success": True, "output": result.stdout.strip()}
        else:
            return {"success": False, "error": result.stderr.strip()}
    except Exception as e:
        return {"success": False, "error": str(e)}


def check_output_files() -> Dict:
    """
    检查输出文件
    """
    output_info = {}
    
    # 检查tmp目录
    tmp_dir = "tmp"
    if os.path.exists(tmp_dir):
        files = os.listdir(tmp_dir)
        output_info["tmp_files"] = files
        output_info["tmp_count"] = len(files)
    else:
        output_info["tmp_files"] = []
        output_info["tmp_count"] = 0
    
    # 检查logs目录
    logs_dir = "logs/experiment1"
    if os.path.exists(logs_dir):
        files = os.listdir(logs_dir)
        output_info["log_files"] = files
        output_info["log_count"] = len(files)
    else:
        output_info["log_files"] = []
        output_info["log_count"] = 0
    
    return output_info


def check_feature_extraction_progress() -> Dict:
    """
    检查特征提取进度
    """
    try:
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        
        # 检查实验1的featuresets
        c.execute("""
            SELECT fs.name, COUNT(fv.id) as feature_count
            FROM feature_sets fs
            LEFT JOIN feature_set_items fsi ON fs.id = fsi.feature_set_id
            LEFT JOIN feature_values fv ON fsi.fxdef_id = fv.fxdef_id
            WHERE fs.name LIKE 'exp1_%'
            GROUP BY fs.id, fs.name
        """)
        
        featureset_progress = {}
        for row in c.fetchall():
            featureset_progress[row[0]] = row[1]
        
        # 检查TUAB数据集的recordings
        c.execute("""
            SELECT COUNT(*) as total_recordings
            FROM recordings r
            JOIN datasets d ON r.dataset_id = d.id
            WHERE d.name = 'TUAB_v3.0.1'
        """)
        
        total_recordings = c.fetchone()[0]
        
        conn.close()
        
        return {
            "featureset_progress": featureset_progress,
            "total_recordings": total_recordings
        }
        
    except Exception as e:
        return {"error": str(e)}


def check_task_file() -> Dict:
    """
    检查任务文件
    """
    task_file = "tmp/experiment1_tasks.txt"
    
    if not os.path.exists(task_file):
        return {"exists": False, "error": "Task file not found"}
    
    try:
        with open(task_file, 'r') as f:
            lines = f.readlines()
        
        task_count = len(lines)
        sample_tasks = lines[:5] if lines else []
        
        return {
            "exists": True,
            "task_count": task_count,
            "sample_tasks": [line.strip() for line in sample_tasks]
        }
    except Exception as e:
        return {"exists": False, "error": str(e)}


def main():
    print("=== Experiment 1 Monitoring Report ===")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print()
    
    # 1. 检查任务状态
    print("1. Job Status:")
    job_status = check_job_status()
    if job_status["success"]:
        if job_status["output"]:
            print("Active jobs found:")
            print(job_status["output"])
        else:
            print("No active jobs found")
    else:
        print(f"Error checking job status: {job_status['error']}")
    print()
    
    # 2. 检查输出文件
    print("2. Output Files:")
    output_info = check_output_files()
    print(f"TMP files: {output_info['tmp_count']} files")
    if output_info['tmp_files']:
        print(f"Sample TMP files: {output_info['tmp_files'][:5]}")
    print(f"Log files: {output_info['log_count']} files")
    if output_info['log_files']:
        print(f"Sample log files: {output_info['log_files'][:5]}")
    print()
    
    # 3. 检查任务文件
    print("3. Task File:")
    task_info = check_task_file()
    if task_info["exists"]:
        print(f"Task file exists with {task_info['task_count']} tasks")
        print("Sample tasks:")
        for task in task_info["sample_tasks"]:
            print(f"  {task}")
    else:
        print(f"Task file error: {task_info['error']}")
    print()
    
    # 4. 检查特征提取进度
    print("4. Feature Extraction Progress:")
    progress = check_feature_extraction_progress()
    if "error" not in progress:
        print(f"Total TUAB recordings: {progress['total_recordings']}")
        print("Featureset progress:")
        for featureset, count in progress['featureset_progress'].items():
            print(f"  {featureset}: {count} features")
    else:
        print(f"Error checking progress: {progress['error']}")
    print()
    
    # 5. 提供检查特定任务的命令
    print("5. Useful Commands:")
    print("To check specific job (replace JOB_ID):")
    print("  qstat JOB_ID")
    print("  qstat -f JOB_ID")
    print("  qstat -t JOB_ID")
    print()
    print("To check output files:")
    print("  ls -la tmp/")
    print("  ls -la logs/experiment1/")
    print()
    print("To check recent log entries:")
    print("  tail -f logs/experiment1/array_job.log")


if __name__ == "__main__":
    main()
