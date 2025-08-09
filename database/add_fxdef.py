import os
import sqlite3
import json
from eeg2fx.function_registry import FEATURE_METADATA
from logging_config import logger

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))

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

    feature_metadata = FEATURE_METADATA.get(func)
    
    if not feature_metadata:
        raise ValueError(f"Unknown feature function: {func}")
    
    feature_type = feature_metadata["type"]

    if feature_type == "single_channel":
        # Single-channel feature: create a feature definition for each channel
        for chan in channels:
            fxdef_id = _add_single_fxdef(c, fxdef_spec, chan, feature_type)
            inserted_ids.append(fxdef_id)
            
    elif feature_type == "channel_pair":
        # Channel pair feature: validate and create feature definitions
        _validate_channel_pair_format(channels, func)
        
        # Process channel pairs
        channel_pairs = []
        for chan in channels:
            if "-" in chan:
                # 连字符格式: "C3-C4"
                ch1, ch2 = chan.split("-", 1)
                ch1 = ch1.strip()
                ch2 = ch2.strip()
                if not ch1 or not ch2:
                    raise ValueError(f"Invalid channel pair format: '{chan}'. Both channels must be non-empty.")
                channel_pairs.append([ch1, ch2])
            else:
                raise ValueError(f"Invalid channel pair format: '{chan}'. Channel pairs must use 'C3-C4' format.")
        
        for ch1, ch2 in channel_pairs:
            chan_str = f"{ch1}-{ch2}"
            fxdef_id = _add_single_fxdef(c, fxdef_spec, chan_str, feature_type)
            inserted_ids.append(fxdef_id)
            
    elif feature_type == "scalar":
        # Scalar feature: only needs one definition
        fxdef_id = _add_single_fxdef(c, fxdef_spec, channels[0] if isinstance(channels, list) else channels, feature_type)
        inserted_ids.append(fxdef_id)
    
    conn.commit()
    conn.close()

    if skipped_count > 0:
        logger.info(f"Summary: {len(inserted_ids) - skipped_count} new fxdefs added, {skipped_count} already existed.")
    else:
        logger.info(f"Summary: {len(inserted_ids)} new fxdefs added.")

    return inserted_ids

def _validate_channel_pair_format(channels, func_name):
    """
    Validate that channel pairs are in the correct format for channel_pair features.
    
    Args:
        channels: List of channel specifications
        func_name: Name of the feature function for error messages
    
    Raises:
        ValueError: If channel format is invalid
    """
    if not channels:
        raise ValueError(f"Channel pair feature '{func_name}' requires at least one channel pair specification")
    
    for i, chan in enumerate(channels):
        if not isinstance(chan, str):
            raise ValueError(f"Channel pair '{chan}' at index {i} must be a string, got {type(chan).__name__}")
        
        if not chan.strip():
            raise ValueError(f"Channel pair at index {i} cannot be empty")
        
        if "-" not in chan:
            raise ValueError(f"Channel pair '{chan}' at index {i} must use 'C3-C4' format (with hyphen)")
        
        # 检查连字符分割后的格式
        parts = chan.split("-", 1)
        if len(parts) != 2:
            raise ValueError(f"Channel pair '{chan}' at index {i} must have exactly one hyphen")
        
        ch1, ch2 = parts[0].strip(), parts[1].strip()
        if not ch1:
            raise ValueError(f"First channel in pair '{chan}' at index {i} cannot be empty")
        if not ch2:
            raise ValueError(f"Second channel in pair '{chan}' at index {i} cannot be empty")
        
        # 检查通道名称格式（可选：可以添加更严格的验证）
        if not ch1.replace(" ", "").isalnum():
            raise ValueError(f"Invalid channel name '{ch1}' in pair '{chan}' at index {i}")
        if not ch2.replace(" ", "").isalnum():
            raise ValueError(f"Invalid channel name '{ch2}' in pair '{chan}' at index {i}")

def _add_single_fxdef(cursor, fxdef_spec, chan, feature_type):
    """Add a single feature definition"""
    func = fxdef_spec["func"]
    pipeid = fxdef_spec["pipeid"]
    shortname = fxdef_spec["shortname"] + "_" + chan
    params = fxdef_spec.get("params", {})
    dim = fxdef_spec.get("dim", "1d")
    ver = fxdef_spec.get("ver", "v1")
    notes = fxdef_spec.get("notes", "")
    
    # Handle placeholders in notes
    if "{chan}" in notes:
        notes = notes.format(chan=chan)
    
    cursor.execute("""
        INSERT INTO fxdef (shortname, ver, dim, func, pipedef_id, chans, params, notes, feature_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (shortname, ver, dim, func, pipeid, chan, json.dumps(params), notes, feature_type))
    
    return cursor.lastrowid
