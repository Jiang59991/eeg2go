#!/usr/bin/env python3
import os
import json
import math
import sqlite3
from typing import List, Dict, Tuple

import numpy as np
import pandas as pd

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))


def _get_recording_ids(dataset_id: int, limit: int | None = None) -> List[int]:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM recordings WHERE dataset_id=? ORDER BY id", (dataset_id,))
    ids = [r[0] for r in c.fetchall()]
    conn.close()
    return ids[: int(limit)] if limit else ids


def _load_fx_values(recording_id: int) -> pd.DataFrame:
    """Load feature_values joined with fxdef for one recording into a dataframe.
    Columns: fxdef_id, func, chans, value_json, shape
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        """
        SELECT fx.id, fx.func, fx.chans, fv.value, fv.shape
        FROM feature_values fv
        JOIN fxdef fx ON fx.id=fv.fxdef_id
        WHERE fv.recording_id=?
        ORDER BY fx.id
        """,
        (recording_id,),
    )
    rows = c.fetchall()
    conn.close()
    return pd.DataFrame(rows, columns=["fxdef_id", "func", "chans", "value_json", "shape"])


def _json_to_vector(value_json) -> List[float]:
    """Parse the stored JSON value into a flat list of floats with np.nan for nulls."""
    if value_json is None or value_json == "null":
        return []
    try:
        data = json.loads(value_json)
    except Exception:
        return []

    if isinstance(data, list):
        # structured list of dicts or a list of numbers
        out = []
        for item in data:
            if isinstance(item, dict) and "value" in item:
                v = item["value"]
            else:
                v = item
            if v is None or (isinstance(v, float) and math.isnan(v)):
                out.append(np.nan)
            else:
                try:
                    out.append(float(v))
                except Exception:
                    out.append(np.nan)
        return out
    else:
        # unknown shape
        return []


def audit_recording(recording_id: int) -> Dict:
    df = _load_fx_values(recording_id)
    if df.empty:
        return {"recording_id": recording_id, "n_features": 0}

    # decode vectors
    vectors: List[List[float]] = []
    for val in df["value_json"].tolist():
        vectors.append(_json_to_vector(val))

    # determine epoch length (use min length across features to be conservative)
    lengths = [len(v) for v in vectors if len(v) > 0]
    n_epochs = min(lengths) if lengths else 0

    # compute per-feature null stats and detect all-null features
    per_feature = []
    all_null_features = []
    for (fxid, func, chans, _val_json), vec in zip(df[["fxdef_id", "func", "chans", "value_json"]].itertuples(index=False), vectors):
        if n_epochs == 0 or len(vec) == 0:
            null_count = n_epochs
            ratio = 1.0 if n_epochs else 0.0
        else:
            v = np.array(vec[:n_epochs], dtype=float)
            null_mask = np.isnan(v)
            null_count = int(null_mask.sum())
            ratio = float(null_count) / float(n_epochs) if n_epochs > 0 else 0.0
        per_feature.append({
            "fxdef_id": fxid, "func": func, "chans": chans,
            "null_count": null_count, "null_ratio": ratio,
        })
        if null_count == n_epochs and n_epochs > 0:
            all_null_features.append((fxid, func, chans))

    per_feature_df = pd.DataFrame(per_feature)

    # per-epoch null counts across features
    per_epoch_null = []
    if n_epochs > 0:
        mat = np.full((len(vectors), n_epochs), np.nan, dtype=float)
        for i, vec in enumerate(vectors):
            if len(vec) >= n_epochs:
                mat[i, :] = np.array(vec[:n_epochs], dtype=float)
        per_epoch_null = np.isnan(mat).sum(axis=0).astype(int).tolist()

    return {
        "recording_id": recording_id,
        "n_features": len(vectors),
        "n_epochs": n_epochs,
        "per_epoch_null": per_epoch_null,
        "per_feature": per_feature_df,
        "all_null_features": all_null_features,
    }


def main():
    import argparse
    parser = argparse.ArgumentParser(description="Audit per-epoch NaN/null and all-null features")
    parser.add_argument("--dataset", type=int, help="Dataset ID")
    parser.add_argument("--recording", type=int, help="Recording ID (overrides dataset if provided)")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of recordings in dataset")
    parser.add_argument("--outdir", type=str, default="outputs/audit")
    args = parser.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    rec_ids: List[int]
    if args.recording:
        rec_ids = [args.recording]
    else:
        if not args.dataset:
            raise SystemExit("Must provide --dataset or --recording")
        rec_ids = _get_recording_ids(args.dataset, args.limit)

    summary_rows = []
    for rid in rec_ids:
        res = audit_recording(rid)
        n_epochs = res.get("n_epochs", 0)
        pf = res.get("per_feature")
        if isinstance(pf, pd.DataFrame) and not pf.empty:
            pf_path = os.path.join(args.outdir, f"per_feature_{rid}.csv")
            pf.to_csv(pf_path, index=False)
        # write per-epoch counts
        pe = res.get("per_epoch_null", [])
        if pe:
            pd.DataFrame({"epoch": list(range(n_epochs)), "null_features": pe}).to_csv(
                os.path.join(args.outdir, f"per_epoch_{rid}.csv"), index=False
            )
        # write all-null list
        with open(os.path.join(args.outdir, f"all_null_{rid}.json"), "w", encoding="utf-8") as f:
            json.dump(res.get("all_null_features", []), f, ensure_ascii=False, indent=2)

        summary_rows.append({
            "recording_id": rid,
            "n_features": res.get("n_features", 0),
            "n_epochs": n_epochs,
            "all_null_count": len(res.get("all_null_features", [])),
            "per_epoch_null_mean": float(np.mean(res.get("per_epoch_null", []) or [0])),
            "per_epoch_null_max": int(np.max(res.get("per_epoch_null", []) or [0])),
        })

    pd.DataFrame(summary_rows).to_csv(os.path.join(args.outdir, "summary.csv"), index=False)


if __name__ == "__main__":
    main()


