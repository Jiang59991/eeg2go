#!/usr/bin/env python3
import os
import sys
import sqlite3
import argparse
import subprocess
from datetime import datetime
from typing import List, Dict

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
                'is_abnormal': bool(row[5])
            })
    
    conn.close()
    
    if limit:
        recordings = recordings[:limit]
    
    return recordings


def submit_feature_extraction_jobs(recordings: List[Dict], featuresets: Dict[str, int], 
                                  queue: str = None, dry_run: bool = False) -> Dict[str, List[str]]:
    """
    为每个pipeline提交特征提取任务
    """
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logs_dir = os.path.join(project_root, "logs", "experiment1")
    os.makedirs(logs_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    job_results = {}
    
    for featureset_name, featureset_id in featuresets.items():
        logger.info(f"Submitting jobs for {featureset_name} (ID: {featureset_id})")
        
        # 为每个pipeline创建单独的日志文件
        pipeline_log_file = os.path.join(logs_dir, f"{featureset_name}_{timestamp}.log")
        job_ids = []
        
        for recording in recordings:
            recording_id = recording['recording_id']
            job_name = f"exp1_{featureset_name}_{recording_id}"
            
            env_vars = (
                f"FEATURE_SET_ID={featureset_id},RECORDING_ID={recording_id},"
                f"EEG2GO_LOG_FILE={pipeline_log_file},EEG2GO_NO_FILE_LOG=0"
            )
            
            cmd = [
                "qsub",
                "-N", job_name,
                "-v", env_vars,
            ]
            
            if queue:
                cmd.extend(["-q", queue])
            
            cmd.append("run_features.pbs")
            
            if dry_run:
                line = f"DRY RUN: {' '.join(cmd)}"
                print(line)
                with open(pipeline_log_file, "a", encoding="utf-8") as f:
                    f.write(line + "\n")
            else:
                env = os.environ.copy()
                env["EEG2GO_NO_FILE_LOG"] = "1"
                
                result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, 
                                      text=True, env=env)
                
                if result.returncode == 0:
                    job_id = result.stdout.strip()
                    job_ids.append(job_id)
                    msg = f"Submitted {job_name}: {job_id}"
                else:
                    msg = f"Failed to submit {job_name}: {result.stderr.strip()}"
                
                print(msg)
                with open(pipeline_log_file, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
        
        job_results[featureset_name] = job_ids
    
    return job_results


def main():
    parser = argparse.ArgumentParser(
        description="Submit feature extraction jobs for Experiment 1 (TUAB subset)"
    )
    parser.add_argument("--queue", type=str, default=None, help="PBS queue name (optional)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of recordings (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Print qsub commands without submitting")
    parser.add_argument("--pipelines", nargs="+", choices=["P0", "P1", "P2", "P3"], 
                       default=["P0", "P1", "P2", "P3"], help="Which pipelines to run")
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
        job_results = submit_feature_extraction_jobs(
            recordings, selected_featuresets, args.queue, args.dry_run
        )
        
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
