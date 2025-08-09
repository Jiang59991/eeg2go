import os
import sqlite3
from logging_config import logger

DB_PATH = os.path.join(os.path.dirname(__file__), "eeg2go.db")

def add_featureset(featureset_spec):
    """
    Register a new feature set with a list of fxdef IDs.
    
    Input format:
    {
        "name": "Human-friendly name",
        "description": "Optional description",
        "fxdef_ids": [1, 2, 3]
    }
    """
    # --- Check input fields ---
    required_keys = ["name", "fxdef_ids"]
    for key in required_keys:
        if key not in featureset_spec:
            raise ValueError(f"Missing required field: '{key}'")

    set_name = featureset_spec["name"]
    description = featureset_spec.get("description", "")
    fxdef_ids = featureset_spec["fxdef_ids"]

    if not isinstance(fxdef_ids, list) or not fxdef_ids:
        raise ValueError("fxdef_ids must be a non-empty list of integers.")

    if not all(isinstance(fxid, int) for fxid in fxdef_ids):
        raise TypeError("All fxdef_ids must be integers.")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 1. Validate fxdef_ids exist
    placeholders = ",".join(["?"] * len(fxdef_ids))
    c.execute(f"SELECT id FROM fxdef WHERE id IN ({placeholders})", fxdef_ids)
    found_ids = {row[0] for row in c.fetchall()}

    missing = set(fxdef_ids) - found_ids
    if missing:
        raise ValueError(f"The following fxdef_ids do not exist: {sorted(missing)}")

    # 2. Check for duplicates: same name and same fxdef_id set
    c.execute("SELECT id FROM feature_sets WHERE name = ?", (set_name,))
    existing_ids = [row[0] for row in c.fetchall()]

    for fsid in existing_ids:
        c.execute("""
            SELECT fxdef_id FROM feature_set_items
            WHERE feature_set_id = ?
        """, (fsid,))
        existing_fxids = sorted([row[0] for row in c.fetchall()])
        if existing_fxids == sorted(fxdef_ids):
            raise ValueError(f"A feature set with name '{set_name}' and identical fxdef_ids already exists (id={fsid}).")

    # 3. Insert new feature set
    c.execute("""
        INSERT INTO feature_sets (name, description)
        VALUES (?, ?)
    """, (set_name, description))
    set_id = c.lastrowid

    # 4. Insert items
    for fxid in fxdef_ids:
        c.execute("""
            INSERT INTO feature_set_items (feature_set_id, fxdef_id)
            VALUES (?, ?)
        """, (set_id, fxid))

    conn.commit()
    conn.close()

    logger.info(f"Feature set '{set_name}' (id={set_id}) registered with {len(fxdef_ids)} features.")
    return set_id
