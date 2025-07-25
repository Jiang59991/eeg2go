import os
import sqlite3
import json
import gc
import hashlib
from collections import deque
import inspect
from eeg2fx.featureset_grouping import build_feature_dag, load_fxdefs_for_set
from eeg2fx.steps import load_recording, RecordingTooLargeError
from eeg2fx.feature_saver import save_feature_values
from eeg2fx.feature.common import standardize_channel_name
import numpy as np
from logging_config import logger
from eeg2fx.node_executor import NodeExecutor

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
            break
    
    return all_cached, results

def run_feature_set(feature_set_id: str, recording_id: int):
    """
    Execute a feature set extraction task using DAG execution.
    First check if all features are already cached, if so return them directly.
    Otherwise, build DAG and execute all nodes in topological order.
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

    # --- DAG-based feature extraction ---
    fxdefs = load_fxdefs_for_set(feature_set_id)
    
    # 构建DAG
    dag = build_feature_dag(fxdefs)
    logger.info(f"构建DAG完成，包含 {len(dag)} 个节点")

    executor = NodeExecutor(recording_id)
    node_outputs = executor.execute_dag(dag)
    logger.info(f"DAG执行完成，输出节点数: {len(node_outputs)}")
    
    # 提取特征结果
    results = {}
    for fx in fxdefs:
        fxid = fx["id"]
        cached = load_cached_feature_value(fxid, recording_id)
        if cached:
            results[fxid] = cached
            continue

        chan = fx["chans"]
        params = fx["params"]
        func = fx["func"]

        try:
            # 从DAG输出中提取特征
            # 查找对应的分割节点（特征节点 -> 分割节点）
            split_id = None
            split_upstream_hash = None
            for node_id, node in dag.items():
                if (node["func"] == "split_channel" and 
                    node["params"].get("chan") == chan and
                    fxid in node.get("fxdef_ids", [])):
                    split_id = node_id
                    # 获取分割节点的upstream_hash
                    upstream_paths = node.get("upstream_paths", set())
                    if upstream_paths:
                        # 取第一个upstream_path的upstream_hash
                        split_upstream_hash = list(upstream_paths)[0][3]
                    break
            
            if split_upstream_hash and split_upstream_hash in node_outputs:
                # 直接使用分割节点的输出
                final_result = node_outputs[split_upstream_hash]
                results[fxid] = prepare_feature_output(final_result)
                logger.debug(f"从分割节点 {split_id} (upstream_hash: {split_upstream_hash}) 提取特征 {fxid}")
            else:
                # 如果找不到分割节点，记录错误
                logger.warning(f"Split node not found for fxid={fxid}, chan={chan}")
                results[fxid] = {
                    "value": None,
                    "dim": "unknown", 
                    "shape": [],
                    "notes": f"Split node not found in DAG for chan={chan}"
                }
            
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

    save_feature_values(recording_id, results)

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
    feature_set_id = 4
    recording_id = 22

    print(f"Running feature set '{feature_set_id}' on recording {recording_id}")
    results = run_feature_set(feature_set_id, recording_id)

    print(f"\nExtracted {len(results)} features:\n")
    for fxid, res in results.items():
        dim = res["dim"]
        shape = res["shape"]
        preview = res["value"][:5] if isinstance(res["value"], list) else res["value"]
        shape_str = "×".join(str(s) for s in shape) if shape else "-"
        print(f"fxdef_id={fxid:>3} | dim={dim:<6} | shape={shape_str:<8} | value={preview}")
