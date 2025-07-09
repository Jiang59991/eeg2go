import os
import sqlite3
import json

DB_PATH = os.path.join(os.path.dirname(__file__), "eeg2go.db")

def add_fxdefs(fxdef_spec):
    """
    Insert one or more feature definitions (fxdef) into the database based on structured specification.

    Reference fxdef_spec format:
    {
        "func": "entropy",               # Feature function name (must match a function in feature_functions.py)
        "pipeid": 5,                     # ID of the associated pipeline (pipedef.id)
        "shortname": "entropy",          # Base shortname, will be expanded per channel (e.g. entropy_C3)
        "channels": ["C3", "C4", "Pz"],  # Channels to generate one fxdef entry per channel
        "params": {"method": "spectral"},# Parameters passed to the feature function
        "dim": "1d",                     # Dimensionality of the output: "scalar", "1d", or "2d"
        "ver": "v1",                     # Version string (optional, defaults to "v1")
        "notes": "Spectral entropy for {chan}" # Optional note template (supports {chan} placeholder)
    }

    For each channel listed, a new fxdef entry will be created with:
        - shortname: shortname_chan
        - chans: chan
        - all other fields populated from fxdef_spec
    """
    # --- Required fields and expected types ---
    required_fields = {
        "func": str,
        "pipeid": int,
        "shortname": str,
        "channels": list,
        "params": dict,
    }

    # Check presence and types
    for key, expected_type in required_fields.items():
        if key not in fxdef_spec:
            raise ValueError(f"Missing required field: '{key}'")
        if not isinstance(fxdef_spec[key], expected_type):
            raise TypeError(f"Field '{key}' must be of type {expected_type.__name__}, but got {type(fxdef_spec[key]).__name__}")

    # Check channels is non-empty list of strings
    if not fxdef_spec["channels"]:
        raise ValueError("Field 'channels' must be a non-empty list of channel names.")
    if not all(isinstance(ch, str) and ch.strip() for ch in fxdef_spec["channels"]):
        raise ValueError("Each entry in 'channels' must be a non-empty string.")

    # Optional fields
    dim = fxdef_spec.get("dim", "1d")
    ver = fxdef_spec.get("ver", "v1")
    notes_template = fxdef_spec.get("notes", "")

    # Validate dim
    if dim not in {"scalar", "1d", "2d"}:
        raise ValueError(f"Invalid dim value: '{dim}'. Must be one of 'scalar', '1d', or '2d'.")

    # Extract fields
    func = fxdef_spec["func"]
    pipeid = fxdef_spec["pipeid"]
    base_shortname = fxdef_spec["shortname"]
    channels = fxdef_spec["channels"]
    params = fxdef_spec["params"]

    # Check pipeid exists
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM pipedef WHERE id = ?", (pipeid,))
    if not c.fetchone():
        raise ValueError(f"Invalid pipedef_id: {pipeid} does not exist in pipedef table.")

    inserted_ids = []
    skipped_count = 0

    # Special handling for paired-channel features like asymmetry
    if func == 'alpha_asymmetry':
        if len(channels) != 2:
            raise ValueError(f"Feature '{func}' requires a list of exactly two channels.")

        # Create a single definition for the channel pair
        chan_pair_str = f"{channels[0]}-{channels[1]}"
        full_shortname = f"{base_shortname}_{chan_pair_str}"
        note = notes_template.replace("{chan}", chan_pair_str)
        params_json = json.dumps(params, sort_keys=True)

        # Check for existence and insert if new
        c.execute("""
            SELECT id FROM fxdef 
            WHERE shortname = ? AND ver = ? AND pipedef_id = ? AND chans = ? AND params = ?
        """, (full_shortname, ver, pipeid, chan_pair_str, params_json))
        
        existing = c.fetchone()
        if existing:
            print(f"fxdef already exists: {full_shortname} (fxid={existing[0]}) - skipping")
            inserted_ids.append(existing[0])
            skipped_count += 1
        else:
            c.execute("""
                INSERT INTO fxdef (shortname, ver, dim, func, pipedef_id, chans, params, notes)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (full_shortname, ver, dim, func, pipeid, chan_pair_str, params_json, note))
            fxid = c.lastrowid
            inserted_ids.append(fxid)
            print(f"fxdef added: {full_shortname} (fxid={fxid})")
        
        # We don't loop for this feature type
        channels = []

    # Insert for each channel (will be skipped for alpha_asymmetry)
    for chan in channels:
        full_shortname = f"{base_shortname}_{chan}"
        note = notes_template.replace("{chan}", chan)
        params_json = json.dumps(params, sort_keys=True)

        # Check if this exact fxdef already exists
        c.execute("""
            SELECT id FROM fxdef 
            WHERE shortname = ? AND ver = ? AND dim = ? AND func = ? 
            AND pipedef_id = ? AND chans = ? AND params = ?
        """, (full_shortname, ver, dim, func, pipeid, chan, params_json))
        
        existing = c.fetchone()
        if existing:
            print(f"fxdef already exists: {full_shortname} (fxid={existing[0]}) - skipping")
            inserted_ids.append(existing[0])
            skipped_count += 1
            continue

        # Insert new fxdef
        c.execute("""
            INSERT INTO fxdef (shortname, ver, dim, func, pipedef_id, chans, params, notes)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            full_shortname,
            ver,
            dim,
            func,
            pipeid,
            chan,
            params_json,
            note
        ))

        fxid = c.lastrowid
        inserted_ids.append(fxid)
        print(f"fxdef added: {full_shortname} (fxid={c.lastrowid})")

    conn.commit()
    conn.close()

    if skipped_count > 0:
        print(f"Summary: {len(inserted_ids) - skipped_count} new fxdefs added, {skipped_count} already existed")
    else:
        print(f"Summary: {len(inserted_ids)} new fxdefs added")

    return inserted_ids
