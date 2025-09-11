#!/usr/bin/env python3
import os
import sqlite3
from typing import Dict, List

from logging_config import logger
from database.add_pipeline import add_pipeline
from database.add_fxdef import add_fxdefs
from database.add_featureset import add_featureset

DB_PATH = os.path.join(os.path.dirname(__file__), "eeg2go.db")

PIPE_SHORTNAMES: List[str] = [
    "P0_minimal_hp",
    "P1_hp_avg_reref",
    "P2_hp_notch50",
    "P3_hp_ica_auto",
]

BANDS: List[str] = ["delta", "theta", "alpha", "beta"]
CHANS: List[str] = ["F3", "F4", "C3", "C4", "O1", "O2"]

def create_experiment1_pipelines() -> None:
    """
    Add the four pipelines used in Experiment 1 to the pipedef table.
    If a pipeline already exists, add_pipeline will ignore or deduplicate.
    Returns:
        None
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

def get_pipeline_id_by_shortname(shortname: str) -> int:
    """
    Get the pipeline id from pipedef table by its shortname.
    Args:
        shortname (str): The shortname of the pipeline.
    Returns:
        int: The id of the pipeline.
    Raises:
        ValueError: If the pipeline is not found.
    """
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
    Build relative_power feature definitions for a given pipeline.
    Args:
        pipe_id (int): The id of the pipeline.
    Returns:
        List[int]: List of feature definition ids created.
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
    Create four featuresets for Experiment 1, one for each pipeline.
    Returns:
        Dict[str, int]: Mapping from featureset name to featureset id.
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
