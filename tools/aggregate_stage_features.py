#!/usr/bin/env python3
import os
import json
import sqlite3
from typing import List, Dict

import numpy as np
import pandas as pd

from logging_config import logger
from tools.extract_exp2_epoch_matrix import extract_epoch_matrix


DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))


def _ensure_stage_table():
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        CREATE TABLE IF NOT EXISTS stage_feature_table (
            dataset_id INTEGER,
            feature_set_id INTEGER,
            subject_id TEXT,
            stage TEXT,
            feature TEXT,
            median REAL,
            iqr REAL,
            n_epochs INTEGER,
            created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
        """
    )
    conn.commit()
    conn.close()


def _load_recording_subject_map(dataset_id: int) -> Dict[int, str]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id, subject_id FROM recordings WHERE dataset_id=?", (dataset_id,))
    mapping = {int(r): s for r, s in c.fetchall()}
    conn.close()
    return mapping


def compute_stage_feature_table(dataset_id: int, feature_set_id: int, limit: int | None = None,
                                min_epochs_per_stage: int = 3,
                                write_to_db: bool = True,
                                out_csv: str | None = None) -> pd.DataFrame:
    """
    从 epoch 级矩阵计算阶段级聚合：subject × stage × feature 的中位数与 IQR，
    纳入规则：该 subject×stage 的子窗数 ≥ min_epochs_per_stage。
    """
    # 1) 准备 epoch 级矩阵（不触发新计算，只读已有缓存）
    result = extract_epoch_matrix(dataset_id, feature_set_id, limit=limit, ensure_cache=False)
    df = result["matrix"].copy()
    if df.empty:
        raise ValueError("Epoch-level matrix is empty. Ensure caches exist or run extraction first.")

    # 2) 追加 subject_id 列
    rec_to_subj = _load_recording_subject_map(dataset_id)
    df["subject_id"] = df["recording_id"].map(rec_to_subj)

    # 3) 识别特征列（排除 meta 列）
    meta_cols = {"recording_id", "epoch_idx", "stage", "subject_id"}
    feature_cols = [c for c in df.columns if c not in meta_cols]

    # 4) 先统计每个 subject×stage 的总子窗数（用于纳入规则）
    stage_counts = (
        df.groupby(["subject_id", "stage"], dropna=False)["epoch_idx"].count().reset_index(name="n_stage_epochs")
    )

    # 5) 针对每个特征，计算中位数与 IQR（忽略 NaN），并合并计数后筛选
    records = []
    for feat in feature_cols:
        agg = df.groupby(["subject_id", "stage"], dropna=False)[feat].agg(
            median=lambda x: float(np.nanmedian(x.values)) if np.any(~np.isnan(x.values)) else np.nan,
            q1=lambda x: float(np.nanpercentile(x.values, 25)) if np.any(~np.isnan(x.values)) else np.nan,
            q3=lambda x: float(np.nanpercentile(x.values, 75)) if np.any(~np.isnan(x.values)) else np.nan,
        ).reset_index()
        agg["iqr"] = agg["q3"] - agg["q1"]
        agg = agg.drop(columns=["q1", "q3"])  # 只保留 IQR
        agg = agg.merge(stage_counts, on=["subject_id", "stage"], how="left")
        # 纳入规则：每位受试者在该阶段的子窗数 ≥ min_epochs_per_stage
        agg = agg[agg["n_stage_epochs"] >= int(min_epochs_per_stage)]
        agg["feature"] = feat
        agg = agg.rename(columns={"n_stage_epochs": "n_epochs"})
        records.append(agg)

    out = pd.concat(records, ignore_index=True)
    out = out[["subject_id", "stage", "feature", "median", "iqr", "n_epochs"]]

    # 6) 持久化
    if write_to_db:
        _ensure_stage_table()
        conn = sqlite3.connect(DB_PATH)
        c = conn.cursor()
        # 可选：清理同一 dataset_id + feature_set_id 的旧结果
        c.execute(
            "DELETE FROM stage_feature_table WHERE dataset_id=? AND feature_set_id=?",
            (dataset_id, feature_set_id),
        )
        # 批量插入
        rows = [
            (dataset_id, feature_set_id, r.subject_id, r.stage, r.feature, None if pd.isna(r.median) else float(r.median),
             None if pd.isna(r.iqr) else float(r.iqr), int(r.n_epochs))
            for r in out.itertuples(index=False)
        ]
        c.executemany(
            """
            INSERT INTO stage_feature_table
            (dataset_id, feature_set_id, subject_id, stage, feature, median, iqr, n_epochs)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )
        conn.commit()
        conn.close()

    if out_csv:
        os.makedirs(os.path.dirname(out_csv), exist_ok=True)
        out.to_csv(out_csv, index=False)
        logger.info(f"Saved stage_feature_table to {out_csv} with shape {out.shape}")

    return out


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Aggregate stage-level features (median/IQR per subject×stage×feature)")
    parser.add_argument("--dataset", type=int, required=True)
    parser.add_argument("--featureset", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--min-epochs", type=int, default=3)
    parser.add_argument("--out", type=str, default="outputs/stage_feature_table.csv")
    args = parser.parse_args()

    df = compute_stage_feature_table(
        dataset_id=args.dataset,
        feature_set_id=args.featureset,
        limit=args.limit,
        min_epochs_per_stage=args.min_epochs,
        write_to_db=True,
        out_csv=args.out,
    )
    logger.info(f"stage_feature_table rows={len(df)}")


if __name__ == "__main__":
    main()


