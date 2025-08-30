#!/usr/bin/env python3
import os
import sys
import sqlite3
import argparse
import subprocess
from datetime import datetime
from typing import List, Dict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logging_config import logger

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))


def get_tuab_dataset_id() -> int:
    """获取TUAB数据集ID"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM datasets WHERE name = ?", ("TUAB_v3.0.1",))
    row = c.fetchone()
    conn.close()
    if not row:
        raise ValueError("TUAB v3.0.1 dataset not found. Please import TUAB data first.")
    return row[0]


def get_experiment1_featuresets() -> Dict[str, int]:
    """获取实验1的featureset ID映射"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, name FROM feature_sets WHERE name LIKE 'exp1_%'")
    featuresets = {row[1]: row[0] for row in c.fetchall()}
    conn.close()
    return featuresets


def get_tuab_subset_recordings(dataset_id: int, limit: int = None) -> List[Dict]:
    """
    获取TUAB子集的recording列表（每个subject选择1条recording，优先选择abnormal）
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 为每个subject选择recording（优先选择abnormal，然后选择最长的）
    c.execute("""
        SELECT DISTINCT r.subject_id 
        FROM recordings r
        WHERE r.dataset_id = ? 
        ORDER BY r.subject_id
    """, (dataset_id,))
    
    subjects = [row[0] for row in c.fetchall()]
    
    # 为每个subject选择recording
    recordings = []
    for subject_id in subjects:
        # 优先选择abnormal标签的recording，然后按duration排序
        c.execute("""
            SELECT r.id, r.subject_id, r.filename, r.path, r.duration, rm.abnormal
            FROM recordings r
            JOIN recording_metadata rm ON r.id = rm.recording_id
            WHERE r.dataset_id = ? AND r.subject_id = ?
            ORDER BY rm.abnormal DESC, r.duration DESC
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
    
    if limit:
        recordings = recordings[:limit]
    
    return recordings


def submit_feature_extraction_jobs(recordings: List[Dict], featuresets: Dict[str, int], 
                                  queue: str = None, dry_run: bool = False, 
                                  max_concurrent: int = 100) -> Dict[str, List[str]]:
    """
    使用array jobs提交特征提取任务
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logs_dir = os.path.join(project_root, "logs", "experiment1")
    os.makedirs(logs_dir, exist_ok=True)
    
    # 确保tmp目录存在
    tmp_dir = os.path.join(project_root, "tmp")
    os.makedirs(tmp_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_results = {}
    
    # 创建任务文件
    task_file = os.path.join(tmp_dir, "experiment1_tasks.txt")
    task_count = 0
    
    with open(task_file, 'w') as f:
        for featureset_name, featureset_id in featuresets.items():
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
            batch_task_file = os.path.join(tmp_dir, f"experiment1_tasks_batch_{batch_idx}.txt")
            with open(task_file, 'r') as src, open(batch_task_file, 'w') as dst:
                for i, line in enumerate(src, 1):
                    if start_idx <= i <= end_idx:
                        dst.write(line)
            
            # 提交批次array job
            job_id = submit_array_job(batch_task_file, start_idx, end_idx, queue, dry_run, max_concurrent)
            job_results[f"batch_{batch_idx}"] = [job_id]
            
    else:
        # 单个array job
        job_id = submit_array_job(task_file, 1, task_count, queue, dry_run, max_concurrent)
        job_results["single_array"] = [job_id]
    
    return job_results


def submit_array_job(task_file: str, start_idx: int, end_idx: int, 
                    queue: str = None, dry_run: bool = False, 
                    max_concurrent: int = 100) -> str:
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
    env_vars = f"EEG2GO_LOG_FILE={os.path.join('logs', 'experiment1', 'array_job.log')}"
    
    # 构建qsub命令
    cmd = [
        "qsub",
        "-N", "eeg_features_array",
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


def main():
    parser = argparse.ArgumentParser(
        description="Submit feature extraction jobs for Experiment 1 (TUAB subset)"
    )
    parser.add_argument("--queue", type=str, default=None, help="PBS queue name (optional)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of recordings (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Print qsub commands without submitting")
    parser.add_argument("--pipelines", nargs="+", choices=["P0", "P1", "P2", "P3"], 
                       default=["P0", "P1", "P2", "P3"], help="Which pipelines to run")
    parser.add_argument("--max-concurrent", type=int, default=20, 
                       help="Maximum number of concurrent array sub-jobs (default: 20)")
    args = parser.parse_args()
    
    try:
        # 获取TUAB数据集ID
        dataset_id = get_tuab_dataset_id()
        print(f"TUAB dataset ID: {dataset_id}")
        
        # 获取实验1的featuresets
        all_featuresets = get_experiment1_featuresets()
        print(f"Found {len(all_featuresets)} experiment 1 featuresets")
        
        # 根据指定的pipeline过滤featuresets
        selected_featuresets = {}
        for name, featureset_id in all_featuresets.items():
            for pipeline in args.pipelines:
                if pipeline in name:
                    selected_featuresets[name] = featureset_id
                    break
        
        if not selected_featuresets:
            print("No featuresets found for specified pipelines")
            return 1
        
        print(f"Selected featuresets: {list(selected_featuresets.keys())}")
        
        # 获取TUAB子集recordings
        recordings = get_tuab_subset_recordings(dataset_id, args.limit)
        print(f"Selected {len(recordings)} recordings for Experiment 1")
        
        # 统计标签分布
        abnormal_count = sum(1 for r in recordings if r.get('is_abnormal', False))
        normal_count = len(recordings) - abnormal_count
        print(f"Label distribution: Normal={normal_count}, Abnormal={abnormal_count}")
        
        if not recordings:
            print("No recordings found for Experiment 1")
            return 1
        
        # 提交特征提取任务
        print(f"\n=== Starting job submission ===")
        print(f"Recordings: {len(recordings)}")
        print(f"Featuresets: {list(selected_featuresets.keys())}")
        print(f"Queue: {args.queue}")
        print(f"Dry run: {args.dry_run}")
        print(f"Max concurrent: {args.max_concurrent}")
        
        try:
            job_results = submit_feature_extraction_jobs(
                recordings, selected_featuresets, args.queue, args.dry_run, args.max_concurrent
            )
        except Exception as e:
            print(f"ERROR during job submission: {e}")
            logger.error(f"Job submission failed: {e}")
            return 1
        
        # 打印结果摘要
        print("\n=== Job Submission Summary ===")
        for featureset_name, job_ids in job_results.items():
            print(f"{featureset_name}: {len(job_ids)} jobs submitted")
            if job_ids:
                print(f"  Sample job IDs: {job_ids[:3]}")
        
        return 0
        
    except Exception as e:
        print(f"Error: {e}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
