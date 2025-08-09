#!/usr/bin/env python3
import os
import sqlite3
from typing import Dict, List

from logging_config import logger
from database.add_fxdef import add_fxdefs
from database.add_featureset import add_featureset


DB_PATH = os.path.join(os.path.dirname(__file__), "eeg2go.db")

# Pipeline shortnames required (must already exist in pipedef)
PIPE_SHORTNAMES = [
    "P0_minimal_hp",
    "P1_hp_avg_reref",
    "P2_hp_notch50",
    "P3_hp_ica_auto",
]

# Bands and channels for Experiment 1 (relative bandpower)
BANDS = ["delta", "theta", "alpha", "beta"]
CHANS = ["F3", "F4", "C3", "C4", "O1", "O2"]


def get_pipeline_id_by_shortname(shortname: str) -> int:
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM pipedef WHERE shortname = ?", (shortname,))
    row = c.fetchone()
    conn.close()
    if not row:
        raise ValueError(f"Pipeline shortname '{shortname}' not found. 请先创建对应 pipeline（见 database/default_pipelines.py）。")
    return int(row[0])


def build_fxdefs_for_pipeline(pipe_id: int) -> List[int]:
    """
    为一个 pipeline 构建相对功率（relative_power）特征：
    频段：delta/theta/alpha/beta
    通道：F3/F4/C3/C4/O1/O2
    输出：按 epoch 的 1d 数组（录波级聚合在下游完成）
    """
    all_fxids: List[int] = []

    for band in BANDS:
        fxdef_spec = {
            "func": "relative_power",
            "pipeid": pipe_id,
            "shortname": f"bp_rel_{band}",
            "channels": CHANS,
            "params": {"band": band},
            "dim": "1d",
            "ver": "v1",
            "notes": f"Relative bandpower for {{chan}} in {band}",
        }
        fxids = add_fxdefs(fxdef_spec)
        all_fxids.extend(fxids)
        logger.info(f"Added relative_power({band}) with {len(fxids)} fxdefs for pipe {pipe_id}")

    return all_fxids


def create_experiment1_featuresets() -> Dict[str, int]:
    """
    为实验1创建4个 featureset：
      - exp1_bp_rel__P0_minimal_hp
      - exp1_bp_rel__P1_hp_avg_reref
      - exp1_bp_rel__P2_hp_notch50
      - exp1_bp_rel__P3_hp_ica_auto
    返回：{featureset_name: featureset_id}
    """
    results: Dict[str, int] = {}

    for shortname in PIPE_SHORTNAMES:
        pipe_id = get_pipeline_id_by_shortname(shortname)
        logger.info(f"Building fxdefs for pipeline {shortname} (id={pipe_id})")

        fxids = build_fxdefs_for_pipeline(pipe_id)

        fs_name = f"exp1_bp_rel__{shortname}"
        fs_id = add_featureset({
            "name": fs_name,
            "description": (
                f"Experiment1 bandpower_rel (delta/theta/alpha/beta) × 6 chans "
                f"for pipeline {shortname}; per-epoch 1d, downstream aggregate by median/IQR"
            ),
            "fxdef_ids": fxids,
        })
        logger.info(f"Created featureset '{fs_name}' (id={fs_id}) with {len(fxids)} fxdefs")
        results[fs_name] = fs_id

    return results


if __name__ == "__main__":
    results = create_experiment1_featuresets()
    print("Created featuresets:", results)


