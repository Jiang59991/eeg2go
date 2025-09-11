from database.add_pipeline import add_pipeline
from logging_config import logger
from typing import NoReturn

def create_default_pipelines() -> NoReturn:
    """
    Create and add a set of default EEG preprocessing pipelines to the database.

    Returns:
        NoReturn
    """
    pipelines = [
        {
            "shortname": "classic_clean_5s",
            "description": "Standard EEG preprocessing: bandpass filter → reref → epoch",
            "source": "default",
            "chanset": "10/20",
            "fs": 250,
            "hp": 1.0,
            "lp": 40.0,
            "epoch": 5.0,
            "sample_rating": 8.0,
            "steps": [
                ["flt", "filter", ["raw"], {"hp": 1.0, "lp": 40.0}],
                ["reref", "reref", ["flt"], {"method": "average"}],
                ["epoch", "epoch", ["reref"], {"duration": 5.0}]
            ]
        },
        {
            "shortname": "full_preprocessing_test",
            "description": "Covers all core preprocessing steps for functional verification",
            "source": "test_suite",
            "chanset": "10/20",
            "fs": 256,
            "hp": 1.0,
            "lp": 30.0,
            "epoch": 4.0,
            "sample_rating": 9.0,
            # This pipeline covers all main preprocessing steps for testing
            "steps": [
                ["flt", "filter", ["raw"], {"hp": 1.0, "lp": 30.0}],
                ["notch", "notch_filter", ["flt"], {"freq": 50.0}],
                ["resample", "resample", ["notch"], {"sfreq": 128.0}],
                ["reref", "reref", ["resample"], {"method": "average"}],
                ["ica", "ica", ["reref"], {"n_components": 20, "detect_artifacts": "none"}],
                ["epoch", "epoch", ["ica"], {"duration": 4.0}],
                ["rej", "reject_high_amplitude", ["epoch"], {"threshold_uv": 150}],
                ["z", "zscore", ["rej"], {"mode": "per_epoch"}]
            ]
        },
    ]

    for p in pipelines:
        logger.info(f"Adding pipeline: {p['shortname']}")
        add_pipeline(p)

if __name__ == "__main__":
    create_default_pipelines()
