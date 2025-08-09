#!/usr/bin/env python3
import os
import sqlite3
from typing import Dict, List

from logging_config import logger
from database.add_fxdef import add_fxdefs
from database.add_featureset import add_featureset
from database.add_pipeline import add_pipeline


DB_PATH = os.path.join(os.path.dirname(__file__), "eeg2go.db")

# Baseline pipeline for Experiment 2 (sleep-stage-aware 10s subepochs)
PIPE_SHORTNAME = "P0_sleep10s_hp"

# Channels and bands (Sleep-EDFx uses Fpz-Cz and Pz-Oz)
CHANS = ["Fpz-Cz", "Pz-Oz"]
BANDS = ["delta", "theta", "alpha", "beta"]

# Wide feature list (22 total: 11 time + 11 freq)
TIME_FEATURES = [
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

FREQ_FEATURES = [
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
    """Create the baseline pipeline for Experiment 2 if it doesn't exist, return its id."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM pipedef WHERE shortname = ?", (PIPE_SHORTNAME,))
    row = c.fetchone()
    conn.close()
    if row:
        return int(row[0])

    pipe = {
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
    pipe_id = add_pipeline(pipe)
    return int(pipe_id)


def build_fxdefs_for_pipeline(pipe_id: int) -> List[int]:
    """
    Build fxdefs for Experiment 2 feature set on a given pipeline.
    Includes:
      - relative_power for 4 bands × 6 chans
      - 22 wide features (time+freq) × 6 chans
    """
    all_fxids: List[int] = []

    # 1) Relative power (per-channel)
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

    # 2) Wide time-domain features
    for fname in TIME_FEATURES:
        fxdef_spec = {
            "func": fname,
            "pipeid": pipe_id,
            "shortname": fname,
            "channels": CHANS,
            "params": {},
            "dim": "1d",
            "ver": "v1",
            "notes": f"{fname} for {{chan}} (10s subepochs)",
        }
        fxids = add_fxdefs(fxdef_spec)
        all_fxids.extend(fxids)
        logger.info(f"Added {fname} with {len(fxids)} fxdefs for pipe {pipe_id}")

    # 3) Wide frequency-domain features
    for fname in FREQ_FEATURES:
        fxdef_spec = {
            "func": fname,
            "pipeid": pipe_id,
            "shortname": fname,
            "channels": CHANS,
            "params": {},
            "dim": "1d",
            "ver": "v1",
            "notes": f"{fname} for {{chan}} (10s subepochs)",
        }
        fxids = add_fxdefs(fxdef_spec)
        all_fxids.extend(fxids)
        logger.info(f"Added {fname} with {len(fxids)} fxdefs for pipe {pipe_id}")

    return all_fxids


def create_experiment2_featureset() -> Dict[str, int]:
    """
    Create the Experiment 2 featureset:
      - featureset name: exp2_wide_sleep_featureset__P0_sleep10s
      - bound to pipeline: P0_sleep10s_hp
    """
    results: Dict[str, int] = {}

    pipe_id = ensure_experiment2_pipeline()
    logger.info(f"Building fxdefs for pipeline {PIPE_SHORTNAME} (id={pipe_id})")

    fxids = build_fxdefs_for_pipeline(pipe_id)

    fs_name = f"exp2_wide_sleep_featureset__{PIPE_SHORTNAME}"
    fs_id = add_featureset({
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
    results = create_experiment2_featureset()
    print("Created featuresets:", results)


