#!/usr/bin/env python3
import os
import time
import json
import sqlite3
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

from logging_config import logger
from eeg2fx.featureset_fetcher import run_feature_set, load_cached_feature_value
from eeg2fx.steps import load_recording, epoch_by_event


DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))

TARGET_CHANS = ["Fpz-Cz", "Pz-Oz"]
REL_POWER_BANDS = ["delta", "theta", "alpha", "beta"]
WIDE_TIME_FEATURES = [
    "peak_to_peak_amplitude",
    "crest_factor",
    "shape_factor",
    "impulse_factor",
    "margin_factor",
    "signal_variance",
    "zero_crossings",
    "signal_entropy",
    "signal_complexity",
    "signal_regularity",
    "signal_stability",
]
WIDE_FREQ_FEATURES = [
    "spectral_entropy",
    "alpha_peak_frequency",
    "theta_alpha_ratio",
    "spectral_edge_frequency",
    "spectral_centroid",
    "spectral_bandwidth",
    "spectral_rolloff",
    "spectral_flatness",
    "spectral_skewness",
    "spectral_kurtosis",
    "spectral_complexity",
]
WIDE_FEATURES = WIDE_TIME_FEATURES + WIDE_FREQ_FEATURES


def _get_recording_ids(dataset_id: int, limit: int | None = None) -> List[int]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM recordings WHERE dataset_id=? ORDER BY id", (dataset_id,))
    ids = [row[0] for row in c.fetchall()]
    conn.close()
    if limit is not None:
        ids = ids[: int(limit)]
    return ids


def _get_fxdefs_for_set(feature_set_id: int) -> List[Dict]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        SELECT fx.id, fx.func, fx.chans, fx.params, fx.shortname, fx.pipedef_id
        FROM feature_set_items AS fs
        JOIN fxdef AS fx ON fs.fxdef_id = fx.id
        WHERE fs.feature_set_id = ?
        ORDER BY fx.id
        """,
        (feature_set_id,),
    )
    rows = c.fetchall()
    conn.close()
    fxdefs = []
    for rid, func, chans, params_json, shortname, pipeid in rows:
        params = json.loads(params_json or "{}")
        fxdefs.append({
            "id": rid,
            "func": func,
            "chans": chans,
            "params": params,
            "shortname": shortname,
            "pipeid": pipeid,
        })
    return fxdefs


def _get_pipeline_epoch_params(pipeid: int) -> Tuple[float, List[str]]:
    """Read subepoch_len and include_values from pipeline definition (steps JSON)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT steps FROM pipedef WHERE id=?", (pipeid,))
    row = c.fetchone()
    conn.close()
    if not row:
        raise ValueError(f"Pipeline {pipeid} not found")
    steps = json.loads(row[0])
    subepoch_len = 10.0
    include_values = ["W", "N1", "N2", "N3", "REM"]
    for _, func, _inputs, params in steps:
        if func == "epoch_by_event":
            subepoch_len = float(params.get("subepoch_len", subepoch_len))
            if "include_values" in params and params["include_values"]:
                include_values = params["include_values"]
            break
    return subepoch_len, include_values


def _build_stage_vector(recording_id: int, pipeid: int) -> List[str]:
    """Recreate epochs for this recording to obtain per-epoch stage labels in the same order."""
    subepoch_len, include_values = _get_pipeline_epoch_params(pipeid)
    raw = load_recording(recording_id)
    epochs = epoch_by_event(
        raw,
        event_type="sleep_stage",
        recording_id=recording_id,
        subepoch_len=subepoch_len,
        drop_partial=True,
        min_overlap=0.8,
        include_values=include_values,
    )
    # Map event integer codes back to labels via events.event_id dict
    inv_map = {v: k for k, v in epochs.event_id.items()}
    labels = [inv_map[int(code)] for code in epochs.events[:, 2]]
    return labels


def _parse_values_from_cache(cached: Dict) -> List[float]:
    if not cached or cached.get("value") is None:
        return []
    v = cached["value"]
    return list(v) if isinstance(v, list) else [v]


def extract_epoch_matrix(dataset_id: int, feature_set_id: int, limit: int | None = None,
                         ensure_cache: bool = True) -> Dict:
    """
    - 确保（或触发）缓存：对缺失缓存的录波调用 run_feature_set
    - 组装每条录波的 epoch×feature 矩阵（46 维）并附加 stage
    - 记录耗时与失败率

    Returns
    -------
    dict: {
      'matrix': pandas.DataFrame,
      'per_recording': List[dict],
      'per_feature': pandas.DataFrame
    }
    """
    fxdefs = _get_fxdefs_for_set(feature_set_id)
    if not fxdefs:
        raise ValueError(f"No fxdefs for featureset {feature_set_id}")
    pipeid = fxdefs[0]["pipeid"]

    recording_ids = _get_recording_ids(dataset_id, limit)
    if not recording_ids:
        raise ValueError(f"No recordings in dataset {dataset_id}")

    # Group fxdefs
    rel_power_fx = [fx for fx in fxdefs if fx["func"] == "relative_power" and fx["params"].get("band") in REL_POWER_BANDS and fx["chans"] in TARGET_CHANS]
    wide_fx = [fx for fx in fxdefs if fx["func"] in WIDE_FEATURES and fx["chans"] in TARGET_CHANS]

    per_rec_logs = []
    per_feature_fail = {fx["id"]: 0 for fx in fxdefs}

    rows = []

    for rid in recording_ids:
        t0 = time.perf_counter()
        status = "ok"
        try:
            if ensure_cache:
                # 触发计算（仅对未缓存者，内部已判断）
                run_feature_set(feature_set_id, rid)

            # 读取阶段标签
            stages = _build_stage_vector(rid, pipeid)
            n_epochs = len(stages)

            # 收集相对功率：4×6=24 列
    rel_cols: Dict[str, List[float]] = {}
            for fx in rel_power_fx:
                cached = load_cached_feature_value(fx["id"], rid)
                vals = _parse_values_from_cache(cached)
                if len(vals) != n_epochs:
                    per_feature_fail[fx["id"]] += 1
                    vals = [np.nan] * n_epochs
                band = fx["params"].get("band")
                key = f"bp_rel__{band}__{fx['chans']}"
                rel_cols[key] = vals

            # 收集“宽特征”并对 2 通道做中位数聚合：得到 22 列
            wide_cols: Dict[str, List[float]] = {}
            for fname in WIDE_FEATURES:
                chan_vals = []
                for ch in TARGET_CHANS:
                    # 找到对应 fxdef
                    fx = next((f for f in wide_fx if f["func"] == fname and f["chans"] == ch), None)
                    if fx is None:
                        continue
                    cached = load_cached_feature_value(fx["id"], rid)
                    vals = _parse_values_from_cache(cached)
                    if len(vals) != n_epochs:
                        per_feature_fail[fx["id"]] += 1
                        vals = [np.nan] * n_epochs
                    chan_vals.append(vals)
                if chan_vals:
                    arr = np.array(chan_vals, dtype=float)  # shape: (6, n_epochs)
                    med = np.nanmedian(arr, axis=0)         # shape: (n_epochs,)
                    wide_cols[fname] = med.tolist()
                else:
                    wide_cols[fname] = [np.nan] * n_epochs

            # 组装本录波的矩阵
            for ei in range(n_epochs):
                row = {
                    "recording_id": rid,
                    "epoch_idx": ei,
                    "stage": stages[ei],
                }
                for k, v in rel_cols.items():
                    row[k] = v[ei]
                for k, v in wide_cols.items():
                    row[k] = v[ei]
                rows.append(row)

        except Exception as e:
            status = f"fail: {e}"
        finally:
            elapsed = time.perf_counter() - t0
            per_rec_logs.append({"recording_id": rid, "status": status, "seconds": elapsed})

    df = pd.DataFrame(rows)

    # 汇总 per-feature 失败率
    per_feat_df = pd.DataFrame({
        "fxdef_id": list(per_feature_fail.keys()),
        "fail_count": list(per_feature_fail.values()),
    })

    return {
        "matrix": df,
        "per_recording": per_rec_logs,
        "per_feature": per_feat_df,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Extract Exp2 epoch-level matrix (46 dims + stage)")
    parser.add_argument("--dataset", type=int, required=True)
    parser.add_argument("--featureset", type=int, required=True)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--out", type=str, default="outputs/exp2_epoch_matrix.csv")
    args = parser.parse_args()

    os.makedirs(os.path.dirname(args.out), exist_ok=True)

    result = extract_epoch_matrix(args.dataset, args.featureset, args.limit, ensure_cache=True)
    df = result["matrix"]
    df.to_csv(args.out, index=False)
    logger.info(f"Saved epoch matrix to {args.out} with shape {df.shape}")

    # 保存统计
    stats_path = args.out.replace(".csv", "__stats.json")
    with open(stats_path, "w", encoding="utf-8") as f:
        json.dump({
            "per_recording": result["per_recording"],
            "per_feature": result["per_feature"].to_dict(orient="list"),
        }, f, ensure_ascii=False, indent=2)
    logger.info(f"Saved stats to {stats_path}")


if __name__ == "__main__":
    main()


