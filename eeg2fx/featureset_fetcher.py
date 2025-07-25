import os
import sqlite3
import json
import gc
from collections import deque
import inspect
from eeg2fx.featureset_grouping import build_feature_dag, load_fxdefs_for_set
from eeg2fx.pipeline_executor import resolve_function
from eeg2fx.pipeline_executor import toposort
from eeg2fx.featureset_grouping import load_pipeline_structure
from eeg2fx.feature_saver import save_feature_values
from eeg2fx.steps import RecordingTooLargeError, load_recording
import numpy as np
from logging_config import logger

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))
MAX_MEMORY_GB = 3  # Set the memory usage limit for a single recording file (GB)

def load_cached_feature_value(fxid, recording_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT value, dim, shape, notes
        FROM feature_values
        WHERE fxdef_id = ? AND recording_id = ?
    """, (fxid, recording_id))
    row = c.fetchone()
    conn.close()

    if row:
        value, dim, shape, notes = row
        
        # Handle null values
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
        
        # Log if this is a failed feature
        if notes and ("Feature generation failed:" in notes or "ERROR:" in notes.upper()):
            logger.warning(f"[SKIP] fxdef_id={fxid} recording_id={recording_id} - previously failed: {notes}")
            
        return result
    return None

def run_pipeline_with_cache(pipeid, recording_id, value_cache, node_output):
    """
    Execute one pipeline, use shared value_cache to avoid redoing nodes.
    Return all intermediate results for this pipeline.
    """
    dag = load_pipeline_structure(pipeid)
    execution_order = toposort(dag)

    for nid in execution_order:
        node = dag[nid]
        func_name = node["func"]
        params = node["params"]
        input_ids = node["inputnodes"]
        inputs = [node_output[i] for i in input_ids]

        cache_key = (
            func_name,
            json.dumps(params, sort_keys=True),
            tuple(input_ids)
        )

        if cache_key in value_cache:
            output = value_cache[cache_key]
            # print(f"[CACHE HIT] func={func_name} | key={cache_key}")
        else:
            func = resolve_function(func_name)
            # print(f"[EXECUTE] func={func_name} | input_nodes={input_ids} | params={params}")

            if func_name == "raw":
                output = func(recording_id, **params)
            else:
                if "chans" in node:
                    output = func(*inputs, chans=node["chans"], **params)
                else:
                    output = func(*inputs, **params)
            value_cache[cache_key] = output
            # print(f"[RESULT] node={nid} → output_type={type(output)} | shape={getattr(output, 'shape', 'N/A') or getattr(output, 'get_data', lambda: 'no get_data')()}")

        node_output[nid] = output

    return node_output

def check_all_features_cached(feature_set_id: str, recording_id: int) -> tuple[bool, dict]:
    """
    检查特征集中的所有特征是否都已经在数据库中缓存
    
    Args:
        feature_set_id: 特征集ID
        recording_id: 录音ID
    
    Returns:
        tuple: (是否全部缓存, 缓存结果字典)
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
            # 不需要在这里创建空结果，因为如果不全缓存就会重新计算
    
    return all_cached, results

def run_feature_set(feature_set_id: str, recording_id: int):
    """
    Execute a feature set extraction task for a single recording file.
    First check if all features are already cached, if so return them directly.
    Otherwise, check the estimated memory usage before starting, and skip if it is too large.
    """
    # --- Pre-check: 检查所有特征是否都已经缓存 ---
    all_cached, cached_results = check_all_features_cached(feature_set_id, recording_id)
    if all_cached:
        logger.info(f"[CACHE HIT] All features for feature_set_id={feature_set_id}, recording_id={recording_id} are already cached")
        return cached_results
    
    # --- Memory usage pre-check and data loading ---
    try:
        raw = load_recording(recording_id)
    except RecordingTooLargeError as e:
        logger.warning(f"[SKIP] Recording {recording_id}: {str(e)}")
        
        # 为所有特征创建失败记录
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
        
        # 保存失败记录并提前返回
        save_feature_values(recording_id, results)
        return results

    # --- Normal feature extraction ---
    fxdefs = load_fxdefs_for_set(feature_set_id)

    results = {}
    value_cache = {}

    # Put the preloaded raw object into the cache to avoid reloading
    raw_node_cache_key = ('raw', '{}', ())
    value_cache[raw_node_cache_key] = raw

    for fx in fxdefs:
        fxid = fx["id"]
        cached = load_cached_feature_value(fxid, recording_id)
        if cached:
            # print(f"[HIT] fxdef_id={fxid} recording_id={recording_id} loaded from feature_values table")
            results[fxid] = cached
            continue

        pipeid = fx["pipeid"]
        chan = fx["chans"]
        params = fx["params"]
        func = fx["func"]
        feature_type = fx.get("feature_type", "single_channel")

        # Create fresh node_output for each feature to prevent memory accumulation
        node_output = {}

        try:
            # Run pipeline and release node_output after use
            run_pipeline_with_cache(pipeid, recording_id, value_cache, node_output)
            output_node = get_pipeline_output_node(pipeid)
            fx_input = node_output[output_node]
            fx_func = resolve_function(func)
            
            # Handle channel parameters according to the feature type
            if feature_type == "single_channel":
                chans_for_func = chan
                chan_for_split = chan
            elif feature_type == "channel_pair":
                if "-" in chan:
                    ch1, ch2 = chan.split("-", 1)
                    chans_for_func = [ch1.strip(), ch2.strip()]
                else:
                    # Single channel, use default pairing
                    chans_for_func = [chan, "C3"]  # or other default pairing logic
                chan_for_split = chan  # Keep the original format for splitting
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
            # Store failed feature with error information
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
            # Clean up memory for this feature - CRITICAL for preventing memory leaks
            if 'fx_input' in locals():
                del fx_input
            if 'fx_output' in locals():
                del fx_output
            if 'final_result' in locals():
                del final_result
            # Clear node_output to prevent memory accumulation
            node_output.clear()
            del node_output
            # Force garbage collection
            gc.collect()

    save_feature_values(recording_id, results)

    # Clean up after full run
    del value_cache
    gc.collect()

    return results

def get_pipeline_output_node(pipeid):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT output_node FROM pipedef WHERE id = ?", (pipeid,))
    row = c.fetchone()
    conn.close()
    if not row:
        raise ValueError(f"Cannot find output node for pipeline {pipeid}")
    return row[0]


def split_channel(result_dict, chan):
    """
    Intelligently extract data from the dictionary returned by the feature function for the specified channel.
    It will try in turn:
    1. Direct match (e.g., 'C3' in {'C3': ...})
    2. Case-insensitive match (e.g., 'pz' in {'PZ': ...})
    3. Standardized name match (e.g., 'EEG Fp1-REF' vs 'FP1')
    4. Channel pair match (e.g., 'C3-C4' in {'C3-C4_asymmetry': ...})
    """
    
    if not isinstance(result_dict, dict):
        return []

    # 1. Direct match
    if chan in result_dict:
        return result_dict[chan]

    # 2. Case-insensitive match
    chan_upper = chan.upper()
    for key, value in result_dict.items():
        if key.upper() == chan_upper:
            return value

    # 3. Standardized match
    std_chan = standardize_channel_name(chan)
    for key, value in result_dict.items():
        std_key = standardize_channel_name(key)
        if std_key == std_chan:
            return value
    
    # 4. Channel pair match (for channel pair features)
    # Look for keys that start with the channel pair followed by underscore
    # e.g., 'C3-C4' should match 'C3-C4_asymmetry', 'C3-C4_coherence', etc.
    for key, value in result_dict.items():
        if key.startswith(chan + "_"):
            return value
            
    return []

def prepare_feature_output(vals):
    """
    Prepare feature output for storage and retrieval.
    Handle different data types including time series data with epoch information.
    """
    
    if vals is None:
        return {
            "value": None,
            "dim": "unknown",
            "shape": [],
            "notes": "No data"
        }
    
    # Check if it is time series data (a list of dictionaries containing epoch information)
    if isinstance(vals, list) and len(vals) > 0:
        
        if isinstance(vals[0], dict) and "epoch" in vals[0] and "value" in vals[0]:
            # This is time series data, keep the complete structured format, but dim is still 1d
            # Each element contains: epoch, start, end, value
            return {
                "value": vals,  # Keep the complete structured data
                "dim": "1d",    # Use 1d instead of time_series
                "shape": [len(vals)],
                "notes": f"1D time series with {len(vals)} epochs, each containing epoch, start, end, value"
            }
        elif isinstance(vals[0], (int, float, np.number)):
            # Normal 1d numerical array
            # print(f"[DEBUG] Numeric array detected")
            return {
                "value": vals,
                "dim": "1d",
                "shape": [len(vals)],
                "notes": f"1D array with {len(vals)} values"
            }
        else:
            # Other types of 1d data
            # print(f"[DEBUG] Other list type detected")
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
    feature_set_id = 3
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
