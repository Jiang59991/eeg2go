import sqlite3
import json
import os

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))

def save_feature_values(recording_id, results):
    """
    Save extracted feature values to the database.
    
    Parameters:
        recording_id (int): The recording ID
        results (dict): {fxdef_id: {value, dim, shape, notes}}
    """
    conn = sqlite3.connect(DB_PATH)
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
            print(f"[ERROR] Failed to insert fxid={fxid}: {e}")

    conn.commit()
    conn.close()
    
    # Count successful and failed features
    successful = sum(1 for res in results.values() if res["value"] is not None)
    failed = len(results) - successful
    print(f"Saved {len(results)} feature values for recording {recording_id} (successful: {successful}, failed: {failed})")


def get_failed_features_stats():
    """
    Get statistics about failed features in the database.
    
    Returns:
        dict: Statistics about failed features
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # Count total failed features
    c.execute("""
        SELECT COUNT(*) FROM feature_values 
        WHERE notes LIKE '%Feature generation failed:%' OR notes LIKE '%ERROR:%'
    """)
    total_failed = c.fetchone()[0]
    
    # Count by fxdef_id
    c.execute("""
        SELECT fxdef_id, COUNT(*) as count 
        FROM feature_values 
        WHERE notes LIKE '%Feature generation failed:%' OR notes LIKE '%ERROR:%'
        GROUP BY fxdef_id
        ORDER BY count DESC
    """)
    failed_by_fxdef = dict(c.fetchall())
    
    # Count by recording_id
    c.execute("""
        SELECT recording_id, COUNT(*) as count 
        FROM feature_values 
        WHERE notes LIKE '%Feature generation failed:%' OR notes LIKE '%ERROR:%'
        GROUP BY recording_id
        ORDER BY count DESC
    """)
    failed_by_recording = dict(c.fetchall())
    
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
        "failed_by_fxdef": failed_by_fxdef,
        "failed_by_recording": failed_by_recording,
        "common_errors": common_errors
    }
