from database.add_fxdef import add_fxdefs

def create_default_features():
    features = []

    # 1. Entropy features from pipeline 5 (e.g. "entropy_eval_base")
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

    # 2. Bandpower (alpha) features from pipeline 1 (e.g. "classic_clean_5s")
    features.append({
        "func": "bandpower",
        "pipeid": 1,
        "shortname": "bp_alpha",
        "channels": ["C3", "C4"],
        "params": {"band": "alpha"},  # Assume bandpower supports band param
        "dim": "1d",
        "ver": "v1",
        "notes": "Alpha band power at {chan}"
    })

    # 3. Z-score stddev (scalar) from pipeline 2 (e.g. "full_preprocessing_test")
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
        print(f"\nAdding features for: {fxdef_spec['shortname']}")
        add_fxdefs(fxdef_spec)

if __name__ == "__main__":
    create_default_features()
