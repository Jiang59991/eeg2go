import sqlite3
import json
import os
import numpy as np
import time
import random
from logging_config import logger

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))

def save_feature_values(recording_id, results, max_retries=5):
    """
    Save extracted feature values to the database with retry mechanism for database locks.
    
    Parameters:
        recording_id (int): The recording ID
        results (dict): {fxdef_id: {value, dim, shape, notes}}
        max_retries (int): Maximum number of retry attempts
    """
    for attempt in range(max_retries):
        try:
            # 使用WAL模式提高并发性能
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            c = conn.cursor()

            for fxid, res in results.items():
                # Handle None values (failed features)
                if res["value"] is None:
                    value = "null"
                else:
                    # 对于结构化数据，直接存储完整的JSON
                    value = json.dumps(res["value"], ensure_ascii=False)
                    
                dim = res.get("dim", "unknown")
                shape = json.dumps(res.get("shape", []))
                notes = res.get("notes", "")

                try:
                    c.execute("""
                        INSERT OR REPLACE INTO feature_values 
                        (fxdef_id, recording_id, value, dim, shape, notes)
                        VALUES (?, ?, ?, ?, ?, ?)
                    """, (fxid, recording_id, value, dim, shape, notes))
                except Exception as e:
                    logger.error(f"Failed to insert fxid={fxid}: {e}")

            conn.commit()
            conn.close()
            
            # Count successful and failed features
            successful = sum(1 for res in results.values() if res["value"] is not None)
            failed = len(results) - successful
            logger.info(f"Saved {len(results)} feature values for recording {recording_id} (successful: {successful}, failed: {failed})")
            return  # 成功保存，退出重试循环
            
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
                # 随机延迟避免冲突
                delay = random.uniform(0.1, 2.0) * (attempt + 1)
                logger.warning(f"Database locked for recording {recording_id}, attempt {attempt + 1}/{max_retries}, retrying in {delay:.2f}s: {e}")
                time.sleep(delay)
                continue
            else:
                logger.error(f"Database error for recording {recording_id} after {attempt + 1} attempts: {e}")
                raise
        except Exception as e:
            logger.error(f"Unexpected error saving features for recording {recording_id}: {e}")
            raise


def is_all_nan_feature(value_json):
    """
    Check if a feature has all nan values across all epochs.
    
    Args:
        value_json (str): JSON string of the feature value
        
    Returns:
        bool: True if all epoch values are nan
    """
    try:
        if value_json is None or value_json == "null":
            return False
            
        data = json.loads(value_json)
        
        # Check if it's a structured result (dict with channel names)
        if isinstance(data, dict):
            for channel_data in data.values():
                if isinstance(channel_data, list):
                    for epoch_data in channel_data:
                        if isinstance(epoch_data, dict) and "value" in epoch_data:
                            value = epoch_data["value"]
                            # Check if value is not nan
                            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                                return False
                else:
                    # Single value case
                    if channel_data is not None and not (isinstance(channel_data, float) and np.isnan(channel_data)):
                        return False
        elif isinstance(data, list):
            # Direct list of values
            for value in data:
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    return False
        else:
            # Single value case
            if data is not None and not (isinstance(data, float) and np.isnan(data)):
                return False
                
        return True
        
    except (json.JSONDecodeError, TypeError):
        return False

def get_failed_features_stats():
    """
    Get statistics about failed features in the database.
    Includes features that failed completely and features where all epoch values are nan.
    
    Returns:
        dict: Statistics about failed features
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Count total failed features (completely failed)
    c.execute("""
        SELECT COUNT(*) FROM feature_values 
        WHERE notes LIKE '%Feature generation failed:%' OR notes LIKE '%ERROR:%'
    """)
    total_failed = c.fetchone()[0]
    
    # Get all features to check for all-nan values
    c.execute("""
        SELECT fxdef_id, recording_id, value 
        FROM feature_values 
        WHERE value IS NOT NULL 
        AND value != 'null'
        AND notes NOT LIKE '%Feature generation failed:%' 
        AND notes NOT LIKE '%ERROR:%'
    """)
    all_features = c.fetchall()
    
    # Count features where all epoch values are nan
    all_nan_features = []
    for fxdef_id, recording_id, value_json in all_features:
        if is_all_nan_feature(value_json):
            all_nan_features.append((fxdef_id, recording_id))
    
    total_all_nan = len(all_nan_features)
    
    # Count by fxdef_id (completely failed)
    c.execute("""
        SELECT fxdef_id, COUNT(*) as count 
        FROM feature_values 
        WHERE notes LIKE '%Feature generation failed:%' OR notes LIKE '%ERROR:%'
        GROUP BY fxdef_id
        ORDER BY count DESC
    """)
    failed_by_fxdef = dict(c.fetchall())
    
    # Count by fxdef_id (all nan values)
    all_nan_by_fxdef = {}
    for fxdef_id, _ in all_nan_features:
        all_nan_by_fxdef[fxdef_id] = all_nan_by_fxdef.get(fxdef_id, 0) + 1
    
    # Count by recording_id (completely failed)
    c.execute("""
        SELECT recording_id, COUNT(*) as count 
        FROM feature_values 
        WHERE notes LIKE '%Feature generation failed:%' OR notes LIKE '%ERROR:%'
        GROUP BY recording_id
        ORDER BY count DESC
    """)
    failed_by_recording = dict(c.fetchall())
    
    # Count by recording_id (all nan values)
    all_nan_by_recording = {}
    for _, recording_id in all_nan_features:
        all_nan_by_recording[recording_id] = all_nan_by_recording.get(recording_id, 0) + 1
    
    # Get common error messages
    c.execute("""
        SELECT notes, COUNT(*) as count 
        FROM feature_values 
        WHERE notes LIKE '%Feature generation failed:%' OR notes LIKE '%ERROR:%'
        GROUP BY notes
        ORDER BY count DESC
        LIMIT 10
    """)
    common_errors = dict(c.fetchall())
    
    conn.close()
    
    return {
        "total_failed": total_failed,
        "total_all_nan": total_all_nan,
        "total_problematic": total_failed + total_all_nan,
        "failed_by_fxdef": failed_by_fxdef,
        "all_nan_by_fxdef": all_nan_by_fxdef,
        "failed_by_recording": failed_by_recording,
        "all_nan_by_recording": all_nan_by_recording,
        "common_errors": common_errors
    }
