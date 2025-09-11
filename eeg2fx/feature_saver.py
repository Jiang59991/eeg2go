import sqlite3
import json
import os
import numpy as np
import time
import random
from logging_config import logger
from typing import Dict, Any, List, Tuple

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))

def save_feature_values(
    recording_id: int, 
    results: Dict[Any, Dict[str, Any]], 
    max_retries: int = 5
) -> None:
    """
    Save extracted feature values to the database with retry mechanism for database locks.

    Args:
        recording_id (int): The recording ID.
        results (dict): {fxdef_id: {value, dim, shape, notes}}.
        max_retries (int): Maximum number of retry attempts.
    Returns:
        None
    """
    for attempt in range(max_retries):
        try:
            # Use WAL mode for better concurrency
            conn = sqlite3.connect(DB_PATH, timeout=30.0)
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            conn.execute("PRAGMA cache_size=10000")
            conn.execute("PRAGMA temp_store=MEMORY")
            c = conn.cursor()

            for fxid, res in results.items():
                if res["value"] is None:
                    value = "null"
                else:
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
            successful = sum(1 for res in results.values() if res["value"] is not None)
            failed = len(results) - successful
            logger.info(f"Saved {len(results)} feature values for recording {recording_id} (successful: {successful}, failed: {failed})")
            return
            
        except sqlite3.OperationalError as e:
            if "database is locked" in str(e) and attempt < max_retries - 1:
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

def is_all_nan_feature(value_json: str) -> bool:
    """
    Check if a feature has all nan values across all epochs.

    Args:
        value_json (str): JSON string of the feature value.

    Returns:
        bool: True if all epoch values are nan, False otherwise.
    """
    try:
        if value_json is None or value_json == "null":
            return False
        data = json.loads(value_json)
        if isinstance(data, dict):
            for channel_data in data.values():
                if isinstance(channel_data, list):
                    for epoch_data in channel_data:
                        if isinstance(epoch_data, dict) and "value" in epoch_data:
                            value = epoch_data["value"]
                            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                                return False
                else:
                    if channel_data is not None and not (isinstance(channel_data, float) and np.isnan(channel_data)):
                        return False
        elif isinstance(data, list):
            for value in data:
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    return False
        else:
            if data is not None and not (isinstance(data, float) and np.isnan(data)):
                return False
        return True
    except (json.JSONDecodeError, TypeError):
        return False

def get_failed_features_stats() -> Dict[str, Any]:
    """
    Get statistics about failed features in the database.
    Includes features that failed completely and features where all epoch values are nan.

    Returns:
        dict: Statistics about failed features.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT COUNT(*) FROM feature_values 
        WHERE notes LIKE '%Feature generation failed:%' OR notes LIKE '%ERROR:%'
    """)
    total_failed = c.fetchone()[0]
    c.execute("""
        SELECT fxdef_id, recording_id, value 
        FROM feature_values 
        WHERE value IS NOT NULL 
        AND value != 'null'
        AND notes NOT LIKE '%Feature generation failed:%' 
        AND notes NOT LIKE '%ERROR:%'
    """)
    all_features = c.fetchall()
    all_nan_features: List[Tuple[Any, Any]] = []
    for fxdef_id, recording_id, value_json in all_features:
        if is_all_nan_feature(value_json):
            all_nan_features.append((fxdef_id, recording_id))
    total_all_nan = len(all_nan_features)
    c.execute("""
        SELECT fxdef_id, COUNT(*) as count 
        FROM feature_values 
        WHERE notes LIKE '%Feature generation failed:%' OR notes LIKE '%ERROR:%'
        GROUP BY fxdef_id
        ORDER BY count DESC
    """)
    failed_by_fxdef = dict(c.fetchall())
    all_nan_by_fxdef: Dict[Any, int] = {}
    for fxdef_id, _ in all_nan_features:
        all_nan_by_fxdef[fxdef_id] = all_nan_by_fxdef.get(fxdef_id, 0) + 1
    c.execute("""
        SELECT recording_id, COUNT(*) as count 
        FROM feature_values 
        WHERE notes LIKE '%Feature generation failed:%' OR notes LIKE '%ERROR:%'
        GROUP BY recording_id
        ORDER BY count DESC
    """)
    failed_by_recording = dict(c.fetchall())
    all_nan_by_recording: Dict[Any, int] = {}
    for _, recording_id in all_nan_features:
        all_nan_by_recording[recording_id] = all_nan_by_recording.get(recording_id, 0) + 1
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
