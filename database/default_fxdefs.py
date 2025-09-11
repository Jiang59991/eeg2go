from database.add_fxdef import add_fxdefs
from logging_config import logger
from typing import List, Dict, Any

def create_default_features() -> None:
    """
    Create and add a set of default feature definitions to the database.

    Returns:
        None
    """
    features: List[Dict[str, Any]] = []

    features.append({
        "func": "spectral_entropy",
        "pipeid": 5,
        "shortname": "entropy",
        "channels": ["C3", "C4", "Pz"],
        "params": {},
        "dim": "1d",
        "ver": "v1",
        "notes": "Spectral entropy for {chan}"
    })

    features.append({
        "func": "bandpower",
        "pipeid": 1,
        "shortname": "bp_alpha",
        "channels": ["C3", "C4"],
        "params": {"band": "alpha"},
        "dim": "1d",
        "ver": "v1",
        "notes": "Alpha band power at {chan}"
    })

    features.append({
        "func": "zscore_stddev",
        "pipeid": 2,
        "shortname": "zstd",
        "channels": ["C3"],
        "params": {},
        "dim": "scalar",
        "ver": "v1",
        "notes": "Z-score std deviation at {chan}"
    })

    for fxdef_spec in features:
        logger.info(f"Adding features for: {fxdef_spec['shortname']}")
        add_fxdefs(fxdef_spec)

if __name__ == "__main__":
    create_default_features()
