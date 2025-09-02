#!/usr/bin/env python3
"""
通用自动化特征提取脚本
支持通过命令行传入dataset和featureset id，提交array job到HPC
"""

import os
import sys
import sqlite3
import argparse
import subprocess
from datetime import datetime
from typing import List, Dict, Optional

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logging_config import logger

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))


def get_dataset_info(dataset_id: int) -> Dict:
    """根据数据集ID获取数据集信息"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, description FROM datasets WHERE id = ?", (dataset_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        raise ValueError(f"Dataset ID {dataset_id} not found. Please check the dataset ID.")
    return {
        'id': row[0],
        'name': row[1],
        'description': row[2]
    }


def get_featureset_info(featureset_id: int) -> Dict:
    """根据特征集ID获取特征集信息"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name, description FROM feature_sets WHERE id = ?", (featureset_id,))
    row = c.fetchone()
    conn.close()
    if not row:
        raise ValueError(f"Featureset ID {featureset_id} not found. Please check the featureset ID.")
    return {
        'id': row[0],
        'name': row[1],
        'description': row[2]
    }


def get_dataset_recordings(dataset_id: int, limit: int = None, 
                          filter_abnormal: bool = None) -> List[Dict]:
    """
    获取数据集的recording列表
    
    Args:
        dataset_id: 数据集ID
        limit: 限制记录数量
        filter_abnormal: 是否只选择abnormal记录 (True=只选abnormal, False=只选normal, None=全部)
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 构建查询语句
    if filter_abnormal is not None:
        # 根据abnormal标签过滤
        c.execute("""
            SELECT r.id, r.subject_id, r.filename, r.path, r.duration, rm.abnormal
            FROM recordings r
            LEFT JOIN recording_metadata rm ON r.id = rm.recording_id
            WHERE r.dataset_id = ?
            AND (rm.abnormal = ? OR (rm.abnormal IS NULL AND ? = False))
            ORDER BY r.subject_id, r.duration DESC
        """, (dataset_id, '1' if filter_abnormal else '0', filter_abnormal))
    else:
        # 获取所有记录
        c.execute("""
            SELECT r.id, r.subject_id, r.filename, r.path, r.duration, rm.abnormal
            FROM recordings r
            LEFT JOIN recording_metadata rm ON r.id = rm.recording_id
            WHERE r.dataset_id = ?
            ORDER BY r.subject_id, r.duration DESC
        """, (dataset_id,))
    
    rows = c.fetchall()
    conn.close()
    
    recordings = []
    for row in rows:
        recordings.append({
            'recording_id': row[0],
            'subject_id': row[1], 
            'filename': row[2],
            'path': row[3],
            'duration': row[4],
            'is_abnormal': row[5] == '1' if isinstance(row[5], str) else bool(row[5])
        })
    
    if limit:
        recordings = recordings[:limit]
    
    return recordings


def get_subject_based_recordings(dataset_id: int, limit: int = None, 
                                prefer_abnormal: bool = True) -> List[Dict]:
    """
    获取基于subject的recording列表（每个subject选择1条recording）
    
    Args:
        dataset_id: 数据集ID
        limit: 限制subject数量
        prefer_abnormal: 是否优先选择abnormal记录
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 获取所有subject
    c.execute("""
        SELECT DISTINCT r.subject_id 
        FROM recordings r
        WHERE r.dataset_id = ? 
        ORDER BY r.subject_id
    """, (dataset_id,))
    
    subjects = [row[0] for row in c.fetchall()]
    
    if limit:
        subjects = subjects[:limit]
    
    # 为每个subject选择recording
    recordings = []
    for subject_id in subjects:
        if prefer_abnormal:
            # 优先选择abnormal标签的recording，然后按duration排序
            c.execute("""
                SELECT r.id, r.subject_id, r.filename, r.path, r.duration, rm.abnormal
                FROM recordings r
                LEFT JOIN recording_metadata rm ON r.id = rm.recording_id
                WHERE r.dataset_id = ? AND r.subject_id = ?
                ORDER BY COALESCE(rm.abnormal, '0') DESC, r.duration DESC
                LIMIT 1
            """, (dataset_id, subject_id))
        else:
            # 按duration排序选择最长的recording
            c.execute("""
                SELECT r.id, r.subject_id, r.filename, r.path, r.duration, rm.abnormal
                FROM recordings r
                LEFT JOIN recording_metadata rm ON r.id = rm.recording_id
                WHERE r.dataset_id = ? AND r.subject_id = ?
                ORDER BY r.duration DESC
                LIMIT 1
            """, (dataset_id, subject_id))
        
        row = c.fetchone()
        if row:
            recordings.append({
                'recording_id': row[0],
                'subject_id': row[1], 
                'filename': row[2],
                'path': row[3],
                'duration': row[4],
                'is_abnormal': row[5] == '1' if isinstance(row[5], str) else bool(row[5])
            })
    
    conn.close()
    return recordings


def submit_feature_extraction_jobs(recordings: List[Dict], featureset_id: int, 
                                  featureset_name: str, queue: str = None, 
                                  dry_run: bool = False, max_concurrent: int = 100,
                                  job_name: str = "eeg_features") -> Dict[str, List[str]]:
    """
    使用array jobs提交特征提取任务
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logs_dir = os.path.join(project_root, "logs", "feature_extraction")
    os.makedirs(logs_dir, exist_ok=True)
    
    # 确保tmp目录存在
    tmp_dir = os.path.join(project_root, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_results = {}
    
    # 创建任务文件
    task_file = os.path.join(tmp_dir, f"{job_name}_tasks_{timestamp}.txt")
    task_count = 0
    
    with open(task_file, 'w') as f:
        for recording in recordings:
            recording_id = recording['recording_id']
            # 格式：recording_id,featureset_id,featureset_name
            f.write(f"{recording_id},{featureset_id},{featureset_name}\n")
            task_count += 1
    
    logger.info(f"Created task file with {task_count} tasks")
    
    # 计算array job的范围
    # 根据Imperial RCS文档，array jobs最大限制是10,000
    if task_count > 10000:
        logger.warning(f"Task count ({task_count}) exceeds maximum array job size (10,000)")
        logger.info("Will split into multiple array jobs")
        
        # 分批处理
        batch_size = 10000
        num_batches = (task_count + batch_size - 1) // batch_size
        
        for batch_idx in range(num_batches):
            start_idx = batch_idx * batch_size + 1
            end_idx = min((batch_idx + 1) * batch_size, task_count)
            
            # 创建批次任务文件
            batch_task_file = os.path.join(tmp_dir, f"{job_name}_tasks_batch_{batch_idx}_{timestamp}.txt")
            with open(task_file, 'r') as src, open(batch_task_file, 'w') as dst:
                for i, line in enumerate(src, 1):
                    if start_idx <= i <= end_idx:
                        dst.write(line)
            
            # 提交批次array job
            job_id = submit_array_job(batch_task_file, start_idx, end_idx, queue, dry_run, max_concurrent, job_name)
            job_results[f"batch_{batch_idx}"] = [job_id]
            
    else:
        # 单个array job
        job_id = submit_array_job(task_file, 1, task_count, queue, dry_run, max_concurrent, job_name)
        job_results["single_array"] = [job_id]
    
    return job_results


def submit_array_job(task_file: str, start_idx: int, end_idx: int, 
                    queue: str = None, dry_run: bool = False, 
                    max_concurrent: int = 100, job_name: str = "eeg_features") -> str:
    """
    提交单个array job
    """
    # 检查任务文件是否存在
    if not os.path.exists(task_file):
        raise FileNotFoundError(f"Task file not found: {task_file}")
    
    # 检查PBS脚本是否存在
    pbs_script = "run_features_array.pbs"
    if not os.path.exists(pbs_script):
        raise FileNotFoundError(f"PBS script not found: {pbs_script}")
    
    # 设置环境变量
    env_vars = f"EEG2GO_LOG_FILE={os.path.join('logs', 'feature_extraction', 'array_job.log')}"
    
    # 构建qsub命令
    cmd = [
        "qsub",
        "-N", job_name,
        "-v", env_vars,
        "-J", f"{start_idx}-{end_idx}%{max_concurrent}",  # 限制并发数
        "-o", "tmp/",
        "-e", "tmp/",
    ]
    
    if queue:
        cmd.extend(["-q", queue])
    
    cmd.append(pbs_script)
    
    # 打印完整命令用于调试
    full_cmd = " ".join(cmd)
    logger.info(f"Submitting array job command: {full_cmd}")
    print(f"Submitting array job command: {full_cmd}")
    
    if dry_run:
        print(f"DRY RUN: {full_cmd}")
        return "DRY_RUN_JOB_ID"
    else:
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                  text=True, timeout=30)
            
            print(f"qsub stdout: {result.stdout}")
            print(f"qsub stderr: {result.stderr}")
            print(f"qsub return code: {result.returncode}")
            
            if result.returncode == 0:
                job_id = result.stdout.strip()
                logger.info(f"Successfully submitted array job {start_idx}-{end_idx}: {job_id}")
                print(f"Successfully submitted array job {start_idx}-{end_idx}: {job_id}")
                return job_id
            else:
                error_msg = f"Failed to submit array job: {result.stderr.strip()}"
                logger.error(error_msg)
                print(f"ERROR: {error_msg}")
                raise RuntimeError(error_msg)
                
        except subprocess.TimeoutExpired:
            error_msg = "qsub command timed out"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            raise RuntimeError(error_msg)
        except Exception as e:
            error_msg = f"Unexpected error submitting array job: {str(e)}"
            logger.error(error_msg)
            print(f"ERROR: {error_msg}")
            raise


def list_available_datasets():
    """列出可用的数据集"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, description FROM datasets ORDER BY name")
    datasets = c.fetchall()
    conn.close()
    
    print("Available datasets:")
    for name, description in datasets:
        print(f"  {name}: {description}")


def list_available_featuresets():
    """列出可用的特征集"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT name, description FROM feature_sets ORDER BY name")
    featuresets = c.fetchall()
    conn.close()
    
    print("Available featuresets:")
    for name, description in featuresets:
        print(f"  {name}: {description}")


def main():
    parser = argparse.ArgumentParser(
        description="Submit feature extraction jobs using array jobs on HPC"
    )
    parser.add_argument("--dataset-id", type=int, required=True, 
                       help="Dataset ID (use --list-datasets to see available datasets)")
    parser.add_argument("--featureset-id", type=int, required=True,
                       help="Featureset ID (use --list-featuresets to see available featuresets)")
    parser.add_argument("--queue", type=str, default=None, 
                       help="PBS queue name (optional)")
    parser.add_argument("--limit", type=int, default=None, 
                       help="Limit number of recordings (optional)")
    parser.add_argument("--subject-based", action="store_true",
                       help="Select one recording per subject (default: all recordings)")
    parser.add_argument("--prefer-abnormal", action="store_true", default=True,
                       help="When using --subject-based, prefer abnormal recordings (default: True)")
    parser.add_argument("--filter-abnormal", type=str, choices=["true", "false", "all"],
                       default="all", help="Filter recordings by abnormal status")
    parser.add_argument("--dry-run", action="store_true", 
                       help="Print qsub commands without submitting")
    parser.add_argument("--max-concurrent", type=int, default=20, 
                       help="Maximum number of concurrent array sub-jobs (default: 20)")
    parser.add_argument("--job-name", type=str, default="eeg_features",
                       help="Job name for PBS (default: eeg_features)")
    parser.add_argument("--list-datasets", action="store_true",
                       help="List available datasets and exit")
    parser.add_argument("--list-featuresets", action="store_true",
                       help="List available featuresets and exit")
    
    args = parser.parse_args()
    
    # 处理列表选项
    if args.list_datasets:
        list_available_datasets()
        return 0
    
    if args.list_featuresets:
        list_available_featuresets()
        return 0
    
    try:
        # 获取数据集信息
        dataset_info = get_dataset_info(args.dataset_id)
        print(f"Dataset: {dataset_info['name']} (ID: {dataset_info['id']})")
        
        # 获取特征集信息
        featureset_info = get_featureset_info(args.featureset_id)
        print(f"Featureset: {featureset_info['name']} (ID: {featureset_info['id']})")
        
        # 获取recordings
        if args.subject_based:
            recordings = get_subject_based_recordings(dataset_info['id'], args.limit, args.prefer_abnormal)
            print(f"Selected {len(recordings)} recordings (one per subject)")
        else:
            # 处理abnormal过滤
            filter_abnormal = None
            if args.filter_abnormal == "true":
                filter_abnormal = True
            elif args.filter_abnormal == "false":
                filter_abnormal = False
            
            recordings = get_dataset_recordings(dataset_info['id'], args.limit, filter_abnormal)
            print(f"Selected {len(recordings)} recordings")
        
        # 统计标签分布
        abnormal_count = sum(1 for r in recordings if r.get('is_abnormal', False))
        normal_count = len(recordings) - abnormal_count
        print(f"Label distribution: Normal={normal_count}, Abnormal={abnormal_count}")
        
        if not recordings:
            print("No recordings found")
            return 1
        
        # 提交特征提取任务
        print(f"\n=== Starting job submission ===")
        print(f"Dataset: {dataset_info['name']} (ID: {dataset_info['id']})")
        print(f"Featureset: {featureset_info['name']} (ID: {featureset_info['id']})")
        print(f"Recordings: {len(recordings)}")
        print(f"Queue: {args.queue}")
        print(f"Dry run: {args.dry_run}")
        print(f"Max concurrent: {args.max_concurrent}")
        print(f"Job name: {args.job_name}")
        
        try:
            job_results = submit_feature_extraction_jobs(
                recordings, featureset_info['id'], featureset_info['name'], args.queue, 
                args.dry_run, args.max_concurrent, args.job_name
            )
        except Exception as e:
            print(f"ERROR during job submission: {e}")
            logger.error(f"Job submission failed: {e}")
            return 1
        
        # 打印结果摘要
        print("\n=== Job Submission Summary ===")
        for job_type, job_ids in job_results.items():
            print(f"{job_type}: {len(job_ids)} jobs submitted")
            if job_ids:
                print(f"  Sample job IDs: {job_ids[:3]}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
