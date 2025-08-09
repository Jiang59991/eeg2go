#!/usr/bin/env python3
import os
import sys
import sqlite3
import argparse
import subprocess
from datetime import datetime
from typing import List

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))


def get_recording_ids_for_dataset(dataset_id: int) -> List[int]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM recordings WHERE dataset_id = ? ORDER BY id", (dataset_id,))
    ids = [row[0] for row in c.fetchall()]
    conn.close()
    return ids


def main():
    parser = argparse.ArgumentParser(
        description="Submit one PBS job per recording to run run_feature_set on HPC"
    )
    parser.add_argument("--dataset", type=int, required=True, help="Dataset ID")
    parser.add_argument("--featureset", type=int, required=True, help="Feature set ID")
    parser.add_argument("--queue", type=str, default=None, help="PBS queue name (optional)")
    parser.add_argument("--limit", type=int, default=None, help="Submit only first N recordings (optional)")
    parser.add_argument("--dry-run", action="store_true", help="Print qsub commands without submitting")
    parser.add_argument(
        "--pbs-stdout-dir",
        type=str,
        default=None,
        help="Directory for PBS stdout/stderr (OU/ER). If not set, OU/ER go to default (cluster-managed).",
    )
    args = parser.parse_args()

    dataset_id = args.dataset
    feature_set_id = args.featureset

    recording_ids = get_recording_ids_for_dataset(dataset_id)
    if args.limit is not None:
        recording_ids = recording_ids[: args.limit]

    if not recording_ids:
        print(f"No recordings found for dataset {dataset_id}")
        return 1

    print(f"Submitting {len(recording_ids)} jobs for dataset {dataset_id}, feature_set {feature_set_id}")

    # 固定日志目录
    project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
    logs_dir = os.path.join(project_root, "logs")
    os.makedirs(logs_dir, exist_ok=True)

    # 准备本次批量提交的统一日志文件名（每次运行该脚本产生日志）
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    batch_log_file = os.path.join(logs_dir, f"submit_qsub_{dataset_id}_{feature_set_id}_{timestamp}.log")
    print(f"Batch log: {batch_log_file}")

    # 让所有被提交的PBS作业里的 Python logger 也写入这一份批次日志
    # 方法：把日志文件路径通过 -v 传给PBS，PBS内导出 EEG2GO_LOG_FILE，logging_config.py 会使用该文件

    for rid in recording_ids:
        job_name = f"feat_{dataset_id}_{rid}"
        env_vars = (
            f"FEATURE_SET_ID={feature_set_id},RECORDING_ID={rid},EEG2GO_LOG_FILE={batch_log_file},EEG2GO_NO_FILE_LOG=0"
        )
        cmd = [
            "qsub",
            "-N",
            job_name,
            "-v",
            env_vars,
        ]
        if args.queue:
            cmd.extend(["-q", args.queue])
        # 可选：控制PBS的stdout/stderr输出目录（避免产生过多文件时可不设置）
        if args.pbs_stdout_dir:
            cmd.extend(["-o", args.pbs_stdout_dir, "-e", args.pbs_stdout_dir])

        cmd.append("run_features.pbs")

        if args.dry_run:
            line = "DRY RUN: " + " ".join(cmd)
            print(line)
            with open(batch_log_file, "a", encoding="utf-8") as f:
                f.write(line + "\n")
        else:
            # 每次提交 qsub 前，确保 HPC 环境已准备：在当前进程内 source/activate（影响当前提交端日志需求）
            # 注意：PBS 节点内仍需在脚本中 source/activate
            env = os.environ.copy()
            env["EEG2GO_NO_FILE_LOG"] = "1"  # 防止该提交脚本自身触发文件日志
            result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)
            out_line = result.stdout.strip()
            err_line = result.stderr.strip()
            if result.returncode != 0:
                msg = f"Failed to submit job for recording {rid}: {err_line}"
                print(msg)
                with open(batch_log_file, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")
            else:
                msg = f"Submitted recording {rid}: {out_line}"
                print(msg)
                with open(batch_log_file, "a", encoding="utf-8") as f:
                    f.write(msg + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())


