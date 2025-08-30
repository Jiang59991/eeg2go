#!/usr/bin/env python3
import os
import sqlite3
from typing import Dict, List

from logging_config import logger
from database.add_pipeline import add_pipeline
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


# === Part A: 先创建/登记实验 1 用到的 pipelines ===
def create_experiment1_pipelines():
    """
    将实验 1 用到的 4 条 pipeline 写入 pipedef（若已存在，add_pipeline 通常会做去重/忽略）。
    """
    pipelines = [
        {
            "shortname": "P0_minimal_hp",
            "description": "Minimal preprocessing baseline: resample -> 0.5 Hz high-pass -> epoch",
            "source": "experiment1",
            "chanset": "10/20",
            "fs": 250.0,
            "hp": 0.5,
            "lp": 0,
            "epoch": 10.0,
            "sample_rating": 9.0,
            "steps": [
                ["resample", "resample", ["raw"], {"sfreq": 250.0}],
                ["hp", "filter", ["resample"], {"hp": 0.5, "lp": 0}],
                ["epoch", "epoch", ["hp"], {"duration": 10.0}],
            ],
        },
        {
            "shortname": "P1_hp_avg_reref",
            "description": "Minimal + average rereferencing: resample -> 0.5 Hz HP -> reref(avg) -> epoch",
            "source": "experiment1",
            "chanset": "10/20",
            "fs": 250.0,
            "hp": 0.5,
            "lp": 0,
            "epoch": 10.0,
            "sample_rating": 9.0,
            "steps": [
                ["resample", "resample", ["raw"], {"sfreq": 250.0}],
                ["hp", "filter", ["resample"], {"hp": 0.5, "lp": 0}],
                ["reref", "reref", ["hp"], {"method": "average"}],
                ["epoch", "epoch", ["reref"], {"duration": 10.0}],
            ],
        },
        {
            "shortname": "P2_hp_notch50",
            "description": "Minimal + 50 Hz notch: resample -> 0.5 Hz HP -> notch(50) -> epoch",
            "source": "experiment1",
            "chanset": "10/20",
            "fs": 250.0,
            "hp": 0.5,
            "lp": 0,
            "epoch": 10.0,
            "sample_rating": 9.0,
            "steps": [
                ["resample", "resample", ["raw"], {"sfreq": 250.0}],
                ["hp", "filter", ["resample"], {"hp": 0.5, "lp": 0}],
                ["notch", "notch_filter", ["hp"], {"freq": 50.0}],
                ["epoch", "epoch", ["notch"], {"duration": 10.0}],
            ],
        },
        {
            "shortname": "P3_hp_ica_auto",
            "description": "Minimal + ICA auto artifact removal: resample -> 0.5 Hz HP -> ICA(auto) -> epoch",
            "source": "experiment1",
            "chanset": "10/20",
            "fs": 250.0,
            "hp": 0.5,
            "lp": 0,
            "epoch": 10.0,
            "sample_rating": 9.0,
            "steps": [
                ["resample", "resample", ["raw"], {"sfreq": 250.0}],
                ["hp", "filter", ["resample"], {"hp": 0.5, "lp": 0}],
                ["ica", "ica", ["hp"], {"n_components": 20, "detect_artifacts": "auto"}],
                ["epoch", "epoch", ["ica"], {"duration": 10.0}],
            ],
        },
    ]

    for p in pipelines:
        logger.info(f"Adding pipeline: {p['shortname']}")
        add_pipeline(p)


# === Part B: 基于上述 pipelines，创建特征定义与特征集 ===
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
    create_experiment1_pipelines()

    results: Dict[str, int] = {}

    for shortname in PIPE_SHORTNAMES:
        pipe_id = get_pipeline_id_by_shortname(shortname)
        logger.info(f"Building fxdefs for pipeline {shortname} (id={pipe_id})")

        fxids = build_fxdefs_for_pipeline(pipe_id)

        fs_name = f"exp1_bp_rel__{shortname}"
        fs_id = add_featureset({
            "name": fs_name,
            "description": (
                "Experiment1: relative bandpower (delta/theta/alpha/beta) × 6 chans; "
                f"pipeline={shortname}; output per-epoch 1d; "
                "downstream aggregation: channel median → recording median."
            ),
            "fxdef_ids": fxids,
        })
        logger.info(f"Created featureset '{fs_name}' (id={fs_id}) with {len(fxids)} fxdefs")
        results[fs_name] = fs_id

    return results


if __name__ == "__main__":
    results = create_experiment1_featuresets()
    print("Created featuresets:", results)


