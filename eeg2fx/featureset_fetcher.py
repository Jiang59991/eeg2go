import os
import sqlite3
import json
import gc
from collections import deque
import inspect
from eeg2fx.featureset_grouping import build_feature_dag, load_fxdefs_for_set
from eeg2fx.pipeline_executor import resolve_function, load_recording
from eeg2fx.pipeline_executor import toposort
from eeg2fx.featureset_grouping import load_pipeline_structure
from eeg2fx.feature_saver import save_feature_values
from eeg2fx.steps import filter, reref, zscore, epoch, notch_filter, resample, ica
from eeg2fx.feature.common import standardize_channel_name
import numpy as np

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))
MAX_MEMORY_GB = 10  # 设置单个录音文件的内存使用上限（GB）

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
            print(f"[SKIP] fxdef_id={fxid} recording_id={recording_id} - previously failed: {notes}")
            
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
            print(f"[CACHE HIT] func={func_name} | key={cache_key}")
        else:
            func = resolve_function(func_name)
            print(f"[EXECUTE] func={func_name} | input_nodes={input_ids} | params={params}")

            if func_name == "raw":
                output = func(recording_id, **params)
            else:
                if "chans" in node:
                    output = func(*inputs, chans=node["chans"], **params)
                else:
                    output = func(*inputs, **params)
            value_cache[cache_key] = output
            print(f"[RESULT] node={nid} → output_type={type(output)} | shape={getattr(output, 'shape', 'N/A') or getattr(output, 'get_data', lambda: 'no get_data')()}")

        node_output[nid] = output

    return node_output

def run_feature_set(feature_set_id: str, recording_id: int):
    """
    为单个录音文件执行一个特征集的提取任务。
    在开始前会检查文件预估内存占用，如果过大则跳过。
    """
    # --- 内存占用预检查 ---
    raw = load_recording(recording_id)
    n_channels = len(raw.ch_names)
    n_samples = raw.n_times
    # MNE加载数据到内存时通常使用float64 (8 bytes)
    estimated_mb = (n_channels * n_samples * 8) / (1024**2)
    
    if estimated_mb > MAX_MEMORY_GB * 1024:
        print(f"[WARN] Recording {recording_id} is too large ({estimated_mb:.2f} MB), exceeding limit of {MAX_MEMORY_GB*1024:.2f} MB. Skipping.")
        
        fxdefs = load_fxdefs_for_set(feature_set_id)
        results = {}
        error_msg = f"Recording too large to process ({estimated_mb:.2f} MB)"
        
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

    # --- 正常执行特征提取 ---
    fxdefs = load_fxdefs_for_set(feature_set_id)

    results = {}
    value_cache = {}

    # 将预加载的raw对象放入缓存，避免重复加载
    raw_node_cache_key = ('raw', '{}', ())
    value_cache[raw_node_cache_key] = raw

    for fx in fxdefs:
        fxid = fx["id"]
        cached = load_cached_feature_value(fxid, recording_id)
        if cached:
            print(f"[HIT] fxdef_id={fxid} recording_id={recording_id} loaded from feature_values table")
            results[fxid] = cached
            continue

        pipeid = fx["pipeid"]
        chan = fx["chans"]
        params = fx["params"]
        func = fx["func"]

        # Create fresh node_output for each feature to prevent memory accumulation
        node_output = {}

        try:
            # Run pipeline and release node_output after use
            run_pipeline_with_cache(pipeid, recording_id, value_cache, node_output)
            output_node = get_pipeline_output_node(pipeid)
            fx_input = node_output[output_node]

            fx_func = resolve_function(func)
            
            # --- Special handling for asymmetry features ---
            chans_for_func = chan
            chan_for_split = chan
            
            if func == 'alpha_asymmetry':
                # The function expects a list of channels, e.g., ['C3', 'C4']
                chans_for_func = chan.split('-')
                # The function returns a hardcoded key, so we must use it for splitting
                chan_for_split = "C4-C3_asymmetry"

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
            print(f"[ERROR] fxdef_id={fxid} recording_id={recording_id}: {error_msg}")
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
    智能地从特征函数返回的字典中提取指定通道的数据。
    它会依次尝试：
    1. 直接匹配 (e.g., 'C3' in {'C3': ...})
    2. 忽略大小写匹配 (e.g., 'pz' in {'PZ': ...})
    3. 标准化名称匹配 (e.g., 'EEG Fp1-REF' vs 'FP1')
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
        if standardize_channel_name(key) == std_chan:
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
    
    # 检查是否是时间序列数据（包含epoch信息的字典列表）
    if isinstance(vals, list) and len(vals) > 0:
        if isinstance(vals[0], dict) and "epoch" in vals[0] and "value" in vals[0]:
            # 这是时间序列数据，保持完整的结构化格式，但dim仍为1d
            # 每个元素包含: epoch, start, end, value
            return {
                "value": vals,  # 保持完整的结构化数据
                "dim": "1d",    # 使用1d而不是time_series
                "shape": [len(vals)],
                "notes": f"1D time series with {len(vals)} epochs, each containing epoch, start, end, value"
            }
        elif isinstance(vals[0], (int, float, np.number)):
            # 普通的1d数值数组
            return {
                "value": vals,
                "dim": "1d",
                "shape": [len(vals)],
                "notes": f"1D array with {len(vals)} values"
            }
        else:
            # 其他类型的1d数据
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
    recording_id = 1

    print(f"Running feature set '{feature_set_id}' on recording {recording_id}")
    results = run_feature_set(feature_set_id, recording_id)

    print(f"\nExtracted {len(results)} features:\n")
    for fxid, res in results.items():
        dim = res["dim"]
        shape = res["shape"]
        preview = res["value"][:5] if isinstance(res["value"], list) else res["value"]
        shape_str = "×".join(str(s) for s in shape) if shape else "-"
        print(f"fxdef_id={fxid:>3} | dim={dim:<6} | shape={shape_str:<8} | value={preview}")
        # print(json.dumps(results[fxid]["value"][:2], indent=2))
