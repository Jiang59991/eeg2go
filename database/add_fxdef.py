import os
import sqlite3
import json
from typing import List, Dict, Any
from eeg2fx.function_registry import FEATURE_METADATA
from logging_config import logger

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))

def add_fxdefs(fxdef_spec: Dict[str, Any]) -> List[int]:
    """
    Insert one or more feature definitions (fxdef) into the database based on structured specification.

    Args:
        fxdef_spec (dict): A dictionary specifying the feature definition(s) to add. Example format:
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

    Returns:
        list: A list of integer IDs of the newly inserted fxdef records.

    Raises:
        ValueError: If required fields are missing or invalid.
        TypeError: If field types are incorrect.
    """
    required_fields = {
        "func": str,
        "pipeid": int,
        "shortname": str,
        "channels": list,
        "params": dict,
    }

    for key, expected_type in required_fields.items():
        if key not in fxdef_spec:
            raise ValueError(f"Missing required field: '{key}'")
        if not isinstance(fxdef_spec[key], expected_type):
            raise TypeError(f"Field '{key}' must be of type {expected_type.__name__}, but got {type(fxdef_spec[key]).__name__}")

    if not fxdef_spec["channels"]:
        raise ValueError("Field 'channels' must be a non-empty list of channel names.")
    if not all(isinstance(ch, str) and ch.strip() for ch in fxdef_spec["channels"]):
        raise ValueError("Each entry in 'channels' must be a non-empty string.")

    dim = fxdef_spec.get("dim", "1d")
    ver = fxdef_spec.get("ver", "v1")
    notes_template = fxdef_spec.get("notes", "")

    if dim not in {"scalar", "1d", "2d"}:
        raise ValueError(f"Invalid dim value: '{dim}'. Must be one of 'scalar', '1d', or '2d'.")

    func = fxdef_spec["func"]
    pipeid = fxdef_spec["pipeid"]
    base_shortname = fxdef_spec["shortname"]
    channels = fxdef_spec["channels"]
    params = fxdef_spec["params"]

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM pipedef WHERE id = ?", (pipeid,))
    if not c.fetchone():
        raise ValueError(f"Invalid pipedef_id: {pipeid} does not exist in pipedef table.")

    inserted_ids: List[int] = []
    skipped_count = 0

    feature_metadata = FEATURE_METADATA.get(func)
    
    if not feature_metadata:
        raise ValueError(f"Unknown feature function: {func}")
    
    feature_type = feature_metadata["type"]

    if feature_type == "single_channel":
        for chan in channels:
            fxdef_id = _add_single_fxdef(c, fxdef_spec, chan, feature_type)
            inserted_ids.append(fxdef_id)
            
    elif feature_type == "channel_pair":
        _validate_channel_pair_format(channels, func)
        
        channel_pairs = []
        for chan in channels:
            if "-" in chan:
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
        fxdef_id = _add_single_fxdef(c, fxdef_spec, channels[0] if isinstance(channels, list) else channels, feature_type)
        inserted_ids.append(fxdef_id)
    
    conn.commit()
    conn.close()

    if skipped_count > 0:
        logger.info(f"Summary: {len(inserted_ids) - skipped_count} new fxdefs added, {skipped_count} already existed.")
    else:
        logger.info(f"Summary: {len(inserted_ids)} new fxdefs added.")

    return inserted_ids

def _validate_channel_pair_format(channels: list, func_name: str) -> None:
    """
    Validate that channel pairs are in the correct format for channel_pair features.

    Args:
        channels (list): List of channel pair specifications (e.g., ["C3-C4", "Fz-Pz"]).
        func_name (str): Name of the feature function for error messages.

    Returns:
        None

    Raises:
        ValueError: If channel format is invalid.
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
        
        parts = chan.split("-", 1)
        if len(parts) != 2:
            raise ValueError(f"Channel pair '{chan}' at index {i} must have exactly one hyphen")
        
        ch1, ch2 = parts[0].strip(), parts[1].strip()
        if not ch1:
            raise ValueError(f"First channel in pair '{chan}' at index {i} cannot be empty")
        if not ch2:
            raise ValueError(f"Second channel in pair '{chan}' at index {i} cannot be empty")
        
        if not ch1.replace(" ", "").isalnum():
            raise ValueError(f"Invalid channel name '{ch1}' in pair '{chan}' at index {i}")
        if not ch2.replace(" ", "").isalnum():
            raise ValueError(f"Invalid channel name '{ch2}' in pair '{chan}' at index {i}")

def _add_single_fxdef(cursor: sqlite3.Cursor, fxdef_spec: Dict[str, Any], chan: str, feature_type: str) -> int:
    """
    Add a single feature definition (fxdef) entry to the database.

    Args:
        cursor (sqlite3.Cursor): SQLite cursor object for database operations.
        fxdef_spec (dict): The feature definition specification dictionary.
        chan (str): The channel or channel pair string for this fxdef.
        feature_type (str): The type of feature ("single_channel", "channel_pair", or "scalar").

    Returns:
        int: The ID of the newly inserted fxdef record.

    Raises:
        sqlite3.DatabaseError: If database insertion fails.
    """
    func = fxdef_spec["func"]
    pipeid = fxdef_spec["pipeid"]
    shortname = fxdef_spec["shortname"] + "_" + chan
    params = fxdef_spec.get("params", {})
    dim = fxdef_spec.get("dim", "1d")
    ver = fxdef_spec.get("ver", "v1")
    notes = fxdef_spec.get("notes", "")
    
    if "{chan}" in notes:
        notes = notes.format(chan=chan)
    
    cursor.execute("""
        INSERT INTO fxdef (shortname, ver, dim, func, pipedef_id, chans, params, notes, feature_type)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (shortname, ver, dim, func, pipeid, chan, json.dumps(params), notes, feature_type))
    
    return cursor.lastrowid
