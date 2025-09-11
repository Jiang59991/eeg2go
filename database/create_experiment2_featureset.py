#!/usr/bin/env python3
import os
import sqlite3
from typing import Dict, List

from logging_config import logger
from database.add_fxdef import add_fxdefs
from database.add_featureset import add_featureset
from database.add_pipeline import add_pipeline

DB_PATH: str = os.path.join(os.path.dirname(__file__), "eeg2go.db")
PIPE_SHORTNAME: str = "P0_sleep10s_hp"
CHANS: List[str] = ["Fpz-Cz", "Pz-Oz"]
BANDS: List[str] = ["delta", "theta", "alpha", "beta"]

TIME_FEATURES: List[str] = [
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

FREQ_FEATURES: List[str] = [
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

def ensure_experiment2_pipeline() -> int:
    """
    Ensure the baseline pipeline for Experiment 2 exists in the database.
    If it does not exist, create it. Return the pipeline id.

    Returns:
        int: The id of the pipeline.
    """
    conn: sqlite3.Connection = sqlite3.connect(DB_PATH)
    c: sqlite3.Cursor = conn.cursor()
    c.execute("SELECT id FROM pipedef WHERE shortname = ?", (PIPE_SHORTNAME,))
    row = c.fetchone()
    conn.close()
    if row:
        return int(row[0])

    pipe: Dict = {
        "shortname": PIPE_SHORTNAME,
        "description": "Exp2 baseline: resample(250) → HP 0.5 Hz → epoch_by_event(sleep_stage, 10s)",
        "source": "experiment2",
        "chanset": "10/20",
        "fs": 250.0,
        "hp": 0.5,
        "lp": 0,
        "epoch": 10.0,
        "steps": [
            ["resample", "resample", ["raw"], {"sfreq": 250.0}],
            ["hp", "filter", ["resample"], {"hp": 0.5, "lp": 0}],
            [
                "ep_stage",
                "epoch_by_event",
                ["hp"],
                {
                    "event_type": "sleep_stage",
                    "subepoch_len": 10.0,
                    "drop_partial": True,
                    "min_overlap": 0.8,
                    "include_values": ["W", "N1", "N2", "N3", "REM"],
                },
            ],
        ],
    }
    logger.info(f"Creating pipeline '{PIPE_SHORTNAME}' for Experiment 2 …")
    pipe_id: int = add_pipeline(pipe)
    return int(pipe_id)

def build_fxdefs_for_pipeline(pipe_id: int) -> List[int]:
    """
    Build all feature definitions (fxdefs) for Experiment 2 for a given pipeline.

    Args:
        pipe_id (int): The id of the pipeline.

    Returns:
        List[int]: List of fxdef ids created.
    """
    all_fxids: List[int] = []

    for band in BANDS:
        fxdef_spec: Dict = {
            "func": "relative_power",
            "pipeid": pipe_id,
            "shortname": f"bp_rel_{band}",
            "channels": CHANS,
            "params": {"band": band},
            "dim": "1d",
            "ver": "v1",
            "notes": f"Relative bandpower for {{chan}} in {band}",
        }
        fxids: List[int] = add_fxdefs(fxdef_spec)
        all_fxids.extend(fxids)
        logger.info(f"Added relative_power({band}) with {len(fxids)} fxdefs for pipe {pipe_id}")

    for fname in TIME_FEATURES:
        fxdef_spec: Dict = {
            "func": fname,
            "pipeid": pipe_id,
            "shortname": fname,
            "channels": CHANS,
            "params": {},
            "dim": "1d",
            "ver": "v1",
            "notes": f"{fname} for {{chan}} (10s subepochs)",
        }
        fxids: List[int] = add_fxdefs(fxdef_spec)
        all_fxids.extend(fxids)
        logger.info(f"Added {fname} with {len(fxids)} fxdefs for pipe {pipe_id}")

    for fname in FREQ_FEATURES:
        fxdef_spec: Dict = {
            "func": fname,
            "pipeid": pipe_id,
            "shortname": fname,
            "channels": CHANS,
            "params": {},
            "dim": "1d",
            "ver": "v1",
            "notes": f"{fname} for {{chan}} (10s subepochs)",
        }
        fxids: List[int] = add_fxdefs(fxdef_spec)
        all_fxids.extend(fxids)
        logger.info(f"Added {fname} with {len(fxids)} fxdefs for pipe {pipe_id}")

    return all_fxids

def create_experiment2_featureset() -> Dict[str, int]:
    """
    Create the Experiment 2 featureset and add it to the database.

    Returns:
        Dict[str, int]: Mapping from featureset name to featureset id.
    """
    results: Dict[str, int] = {}

    pipe_id: int = ensure_experiment2_pipeline()
    logger.info(f"Building fxdefs for pipeline {PIPE_SHORTNAME} (id={pipe_id})")

    fxids: List[int] = build_fxdefs_for_pipeline(pipe_id)

    fs_name: str = f"exp2_wide_sleep_featureset__{PIPE_SHORTNAME}"
    fs_id: int = add_featureset({
        "name": fs_name,
        "description": (
            "Experiment 2: Sleep-EDFx two-channel (Fpz-Cz, Pz-Oz); 10s subepochs aligned to sleep stages; "
            "features = 4 relative bands × 2 chans + 22 wide features (median over 2 chans)."
        ),
        "fxdef_ids": fxids,
    })
    logger.info(f"Created featureset '{fs_name}' (id={fs_id}) with {len(fxids)} fxdefs")
    results[fs_name] = fs_id

    return results

if __name__ == "__main__":
    results: Dict[str, int] = create_experiment2_featureset()
    print("Created featuresets:", results)
