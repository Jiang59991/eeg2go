from database.add_pipeline import add_pipeline
from database.add_fxdef import add_fxdefs
from database.add_featureset import add_featureset
from eeg2fx.function_registry import FEATURE_FUNCS, FEATURE_METADATA
import sqlite3
import os
from logging_config import logger

DB_PATH = os.path.join(os.path.dirname(__file__), "eeg2go.db")

def get_latest_fxdef_ids(n):
    """Return last n fxdef IDs in ascending order (used immediately after batch insert)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM fxdef ORDER BY id DESC LIMIT ?", (n,))
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in reversed(rows)]

def create_all_features_featureset():
    """
    Creates a feature set containing all available features from the eeg2fx/feature directory.
    Handles channel configurations intelligently based on feature type.
    """
    logger.info("Creating a feature set containing all available features...")
    available_features = list(FEATURE_FUNCS.keys())
    logger.info(f"Found {len(available_features)} available feature functions.")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM pipedef ORDER BY id LIMIT 1")
    pipeline_id = c.fetchone()[0]
    conn.close()
    logger.info(f"Using pipeline ID: {pipeline_id}")
    all_fxdef_ids = []
    for feature_func in available_features:
        logger.info(f"Processing feature: {feature_func}")
        metadata = FEATURE_METADATA.get(feature_func, {})
        feature_type = metadata.get("type", "single_channel")
        if feature_type == "single_channel":
            common_channels = ["C3", "C4", "Pz", "O1", "O2"]
            try:
                fxdef_ids = add_fxdefs({
                    "func": feature_func,
                    "pipeid": pipeline_id,
                    "shortname": feature_func,
                    "channels": common_channels,
                    "params": {},
                    "dim": "1d",
                    "ver": "v1",
                    "notes": f"{feature_func} for {{chan}}"
                })
                all_fxdef_ids.extend(fxdef_ids)
                logger.info(f"  ✓ Added {len(fxdef_ids)} {feature_func} feature definitions.")
            except Exception as e:
                logger.error(f"  ✗ Adding {feature_func} failed: {e}")
        elif feature_type == "channel_pair":
            default_pairs = metadata.get("default_pairs", [["C3", "C4"]])
            hyphen_pairs = [f"{pair[0]}-{pair[1]}" for pair in default_pairs]
            try:
                fxdef_ids = add_fxdefs({
                    "func": feature_func,
                    "pipeid": pipeline_id,
                    "shortname": feature_func,
                    "channels": hyphen_pairs,
                    "params": {},
                    "dim": "1d",
                    "ver": "v1",
                    "notes": f"{feature_func} for {{chan}}"
                })
                all_fxdef_ids.extend(fxdef_ids)
                logger.info(f"  ✓ Added {len(fxdef_ids)} {feature_func} feature definitions.")
            except Exception as e:
                logger.error(f"  ✗ Adding {feature_func} failed: {e}")
        elif feature_type == "scalar":
            try:
                fxdef_ids = add_fxdefs({
                    "func": feature_func,
                    "pipeid": pipeline_id,
                    "shortname": feature_func,
                    "channels": ["C3"],
                    "params": {},
                    "dim": "scalar",
                    "ver": "v1",
                    "notes": f"{feature_func} (scalar)"
                })
                all_fxdef_ids.extend(fxdef_ids)
                logger.info(f"  ✓ Added {len(fxdef_ids)} {feature_func} feature definitions.")
            except Exception as e:
                logger.error(f"  ✗ Adding {feature_func} failed: {e}")
    try:
        set_id = add_featureset({
            "name": "all_available_features",
            "description": f"Feature set containing all available features from the eeg2fx/feature directory, totaling {len(all_fxdef_ids)} feature definitions",
            "fxdef_ids": all_fxdef_ids
        })
        logger.info(f"✓ Successfully created feature set 'all_available_features' (ID: {set_id}), containing {len(all_fxdef_ids)} feature definitions.")
        return set_id
    except Exception as e:
        logger.error(f"✗ Creating feature set failed: {e}")
        return None

def register_comparison_feature_sets():
    all_fxids = []

    # 1. Same pipeline, multiple channels
    entropy_multi_ch = []
    for ch in ["C3", "C4", "Pz"]:
        entropy_multi_ch.append({
            "func": "spectral_entropy",
            "pipeid": 5,  # pipeline: entropy_eval_base
            "shortname": "entropy",
            "channels": [ch],
            "params": {},
            "dim": "1d",
            "ver": "v1",
            "notes": "Spectral entropy for {chan}"
        })

    logger.info("Creating feature set: set_entropy_multi_ch")
    for fxdef in entropy_multi_ch:
        all_fxids += add_fxdefs(fxdef)

    add_featureset({
        "name": "Entropy across multiple channels",
        "description": "Same pipeline & function, C3/C4/Pz",
        "fxdef_ids": all_fxids
    })

    # 2. Same channel, same function, different pipelines (differs by one node)
    logger.info("Adding comparison pipelines...")

    base_steps = [
        ["flt", "filter", ["raw"], {"hp": 1.0, "lp": 35.0}],
        ["epoch", "epoch", ["flt"], {"duration": 2.0}]
    ]

    zscore_steps = [
        ["flt", "filter", ["raw"], {"hp": 1.0, "lp": 35.0}],
        ["notch", "notch_filter", ["flt"], {"freq": 50}],
        ["epoch", "epoch", ["notch"], {"duration": 2.0}]
    ]

    pipe1 = {
        "shortname": "entropy_base",
        "description": "Base pipeline without zscore",
        "source": "comparison",
        "chanset": "10/20",
        "fs": 250,
        "hp": 1.0,
        "lp": 35.0,
        "epoch": 2.0,
        "steps": base_steps
    }

    pipe2 = {
        "shortname": "entropy_notch",
        "description": "Adds notch filter before epoch",
        "source": "comparison",
        "chanset": "10/20",
        "fs": 250,
        "hp": 1.0,
        "lp": 35.0,
        "epoch": 2.0,
        "steps": zscore_steps
    }

    id1 = add_pipeline(pipe1)
    id2 = add_pipeline(pipe2)

    entropy_multi_pipe = [
        {
            "func": "spectral_entropy",
            "pipeid": id1,
            "shortname": "entropy_nz",
            "channels": ["C3"],
            "params": {},
            "dim": "1d",
            "ver": "v1",
            "notes": "Entropy C3 no zscore"
        },
        {
            "func": "spectral_entropy",
            "pipeid": id2,
            "shortname": "entropy_notch",
            "channels": ["C3"],
            "params": {},
            "dim": "1d",
            "ver": "v1",
            "notes": "Entropy C3 with notch"
        }
    ]

    all_fxids = []
    logger.info("Creating feature set: set_entropy_multi_pipe")
    for fxdef in entropy_multi_pipe:
        all_fxids += add_fxdefs(fxdef)

    add_featureset({
        "name": "Entropy from different pipelines",
        "description": "Same function & channel, but pipeline differs (zscore)",
        "fxdef_ids": all_fxids
    })

if __name__ == "__main__":
    # Create a feature set containing all features
    create_all_features_featureset()
    
    # Create comparison feature sets (optional)
    # register_comparison_feature_sets()
