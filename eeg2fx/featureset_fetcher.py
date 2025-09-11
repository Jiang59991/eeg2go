import os
import sqlite3
import json
import gc
import time
import random
from collections import deque
import inspect
from .featureset_grouping import load_fxdefs_for_set
from .pipeline_executor import resolve_function, run_pipeline
from .feature_saver import save_feature_values
from .steps import RecordingTooLargeError, load_recording
import numpy as np
from logging_config import logger
from .feature.common import wrap_structured_result, auto_gc
from typing import Any, Dict, Tuple, Optional

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))
MAX_MEMORY_GB = 1

@auto_gc
def load_cached_feature_value(fxid, recording_id, max_retries=3) -> Optional[Dict[str, Any]]:
    """
    Load a cached feature value from the database for a given feature id and recording id.
    Returns a dictionary with keys: value, dim, shape, notes, or None if not found or error.
    """
    for attempt in range(max_retries):
        try:
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            c = conn.cursor()
            c.execute("""
                SELECT value, dim, shape, notes
                FROM feature_values
                WHERE fxdef_id = ? AND recording_id = ?
            """, (fxid, recording_id))
            row = c.fetchone()
            conn.close()
            break
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                delay = random.uniform(0.1, 1.0) * (attempt + 1)
                logger.warning(f"Database locked reading fxid={fxid}, attempt {attempt + 1}/{max_retries}, retrying in {delay:.2f}s")
                time.sleep(delay)
                continue
            else:
                logger.error(f"Database error reading fxid={fxid} after {attempt + 1} attempts: {e}")
                return None
        except Exception as e:
            logger.error(f"Unexpected error reading fxid={fxid}: {e}")
            return None

    if row:
        value, dim, shape, notes = row
        if value == "null":
            parsed_value = None
        else:
            parsed_value = json.loads(value)
        result = {
            "value": parsed_value,
            "dim": dim,
            "shape": json.loads(shape),
            "notes": notes
        }
        if notes and ("Feature generation failed:" in notes or "ERROR:" in notes.upper()):
            logger.warning(f"[SKIP] fxdef_id={fxid} recording_id={recording_id} - previously failed: {notes}")
        return result
    return None

@auto_gc
def check_all_features_cached(feature_set_id: str, recording_id: int) -> Tuple[bool, Dict[str, Any]]:
    """
    Check if all features in a feature set are already cached in the database for a given recording.
    Returns (all_cached, results_dict).
    """
    fxdefs = load_fxdefs_for_set(feature_set_id)
    results = {}
    all_cached = True
    for fx in fxdefs:
        fxid = fx["id"]
        cached = load_cached_feature_value(fxid, recording_id)
        if cached:
            results[fxid] = cached
        else:
            all_cached = False
    return all_cached, results

@auto_gc
def run_feature_set(feature_set_id: str, recording_id: int) -> Dict[str, Any]:
    """
    Execute a feature set extraction task for a single recording file.
    If all features are cached, return them. Otherwise, extract features and cache them.
    """
    all_cached, cached_results = check_all_features_cached(feature_set_id, recording_id)
    if all_cached:
        logger.info(f"[CACHE HIT] All features for feature_set_id={feature_set_id}, recording_id={recording_id} are already cached")
        return cached_results

    try:
        raw = load_recording(recording_id)
    except RecordingTooLargeError as e:
        logger.warning(f"[SKIP] Recording {recording_id}: {str(e)}")
        fxdefs = load_fxdefs_for_set(feature_set_id)
        results = {}
        error_msg = str(e)
        for fx in fxdefs:
            fxid = fx["id"]
            results[fxid] = {
                "value": None,
                "dim": "unknown", 
                "shape": [],
                "notes": error_msg
            }
        save_feature_values(recording_id, results)
        return results

    fxdefs = load_fxdefs_for_set(feature_set_id)
    results = {}
    value_cache = {}
    raw_node_cache_key = ('raw', '{}', ())
    value_cache[raw_node_cache_key] = raw

    for fx in fxdefs:
        fxid = fx["id"]
        cached = load_cached_feature_value(fxid, recording_id)
        if cached:
            results[fxid] = cached
            continue

        pipeid = fx["pipeid"]
        chan = fx["chans"]
        params = fx["params"]
        func = fx["func"]
        feature_type = fx.get("feature_type", "single_channel")
        node_output = {}

        try:
            run_pipeline(pipeid, recording_id, value_cache, node_output)
            output_node = get_pipeline_output_node(pipeid)
            fx_input = node_output[output_node]
            fx_func = resolve_function(func)
            if feature_type == "single_channel":
                chans_for_func = chan
                chan_for_split = chan
            elif feature_type == "channel_pair":
                if "-" in chan:
                    ch1, ch2 = chan.split("-", 1)
                    chans_for_func = [ch1.strip(), ch2.strip()]
                else:
                    chans_for_func = [chan, "C3"]
                chan_for_split = chan
            elif feature_type == "scalar":
                chans_for_func = chan
                chan_for_split = chan
            else:
                chans_for_func = chan
                chan_for_split = chan

            sig = inspect.signature(fx_func)
            if 'chans' in sig.parameters:
                fx_output = fx_func(fx_input, chans=chans_for_func, **params)
            else:
                fx_output = fx_func(fx_input, **params)

            final_result = split_channel(fx_output, chan_for_split)
            results[fxid] = prepare_feature_output(final_result)
        except Exception as e:
            error_msg = f"Feature generation failed: {str(e)}"
            logger.error(f"[ERROR] fxdef_id={fxid} recording_id={recording_id}: {error_msg}")
            import traceback
            logger.error(f"[ERROR] Full traceback:\n{traceback.format_exc()}")
            results[fxid] = {
                "value": None,
                "dim": "unknown", 
                "shape": [],
                "notes": error_msg
            }
        finally:
            if 'fx_input' in locals():
                del fx_input
            if 'fx_output' in locals():
                del fx_output
            if 'final_result' in locals():
                del final_result
            node_output.clear()
            del node_output
            gc.collect()

    save_feature_values(recording_id, results)
    del value_cache
    gc.collect()
    return results

def _create_error_result(feature_set_id: str, recording_id: int, error_msg: str) -> Dict[str, Any]:
    """
    Create an error result dictionary for all features in a feature set.
    """
    fxdefs = load_fxdefs_for_set(feature_set_id)
    results = {}
    for fx in fxdefs:
        fxid = fx["id"]
        results[fxid] = {
            "value": None,
            "dim": "unknown",
            "shape": [],
            "notes": f"Task execution failed: {error_msg}"
        }
    return results

@auto_gc
def get_pipeline_output_node(pipeid) -> Any:
    """
    Get the output node name for a given pipeline id from the database.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT output_node FROM pipedef WHERE id = ?", (pipeid,))
    row = c.fetchone()
    conn.close()
    if not row:
        raise ValueError(f"Cannot find output node for pipeline {pipeid}")
    return row[0]

@auto_gc
def split_channel(result_dict, chan) -> Any:
    """
    Extract data for the specified channel from the result dictionary.
    Tries direct match, case-insensitive match, and channel pair match.
    """
    if not isinstance(result_dict, dict):
        return []
    if chan in result_dict:
        return result_dict[chan]
    chan_upper = chan.upper()
    for key, value in result_dict.items():
        if key.upper() == chan_upper:
            return value
    for key, value in result_dict.items():
        if key.startswith(chan + "_"):
            return value
    return []

@auto_gc
def prepare_feature_output(vals) -> Dict[str, Any]:
    """
    Prepare feature output for storage and retrieval.
    Handles different data types including time series data with epoch information.
    """
    if vals is None:
        return {
            "value": None,
            "dim": "unknown",
            "shape": [],
            "notes": "No data"
        }
    if isinstance(vals, list) and len(vals) > 0:
        if isinstance(vals[0], dict) and "epoch" in vals[0] and "value" in vals[0]:
            return {
                "value": vals,
                "dim": "1d",
                "shape": [len(vals)],
                "notes": f"1D time series with {len(vals)} epochs, each containing epoch, start, end, value"
            }
        elif isinstance(vals[0], (int, float, np.number)):
            return {
                "value": vals,
                "dim": "1d",
                "shape": [len(vals)],
                "notes": f"1D array with {len(vals)} values"
            }
        else:
            return {
                "value": vals,
                "dim": "1d",
                "shape": [len(vals)],
                "notes": f"1D array with {len(vals)} items"
            }
    elif isinstance(vals, (int, float, np.number)):
        return {
            "value": [vals],
            "dim": "scalar",
            "shape": [1],
            "notes": "Scalar value"
        }
    else:
        return {
            "value": vals,
            "dim": "unknown",
            "shape": [],
            "notes": f"Unknown data type: {type(vals)}"
        }

if __name__ == "__main__":
    feature_set_id = 2
    recording_id = 7

    print(f"Running feature set '{feature_set_id}' on recording {recording_id}")
    results = run_feature_set(feature_set_id, recording_id)

    print(f"\nExtracted {len(results)} features:\n")
    for fxid, res in results.items():
        dim = res["dim"]
        shape = res["shape"]
        preview = res["value"][:5] if isinstance(res["value"], list) else res["value"]
        shape_str = "×".join(str(s) for s in shape) if shape else "-"
        print(f"fxdef_id={fxid:>3} | dim={dim:<6} | shape={shape_str:<8} | value={preview}")
