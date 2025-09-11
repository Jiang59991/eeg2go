#!/usr/bin/env python3
import argparse
import sys
from logging_config import logger

from eeg2fx.featureset_fetcher import run_feature_set

def main() -> int:
    """
    Main entry point for running feature extraction for a single recording.

    Returns:
        int: Exit code, 0 for success, 1 for failure.
    """
    parser = argparse.ArgumentParser(description="Run feature extraction for a single recording")
    parser.add_argument("--featureset", type=int, required=True, help="Feature set ID")
    parser.add_argument("--recording", type=int, required=True, help="Recording ID")
    args = parser.parse_args()

    feature_set_id = args.featureset
    recording_id = args.recording

    logger.info(f"Starting run_feature_set for recording_id={recording_id}, feature_set_id={feature_set_id}")
    try:
        result = run_feature_set(feature_set_id, recording_id)
        logger.info(
            f"Completed run_feature_set for recording_id={recording_id}, feature_set_id={feature_set_id}"
        )
        # Print a simple success flag; see PBS logs for detailed logger output
        print("success", recording_id)
        return 0
    except Exception as e:
        logger.error(
            f"Failed run_feature_set for recording_id={recording_id}, feature_set_id={feature_set_id}: {e}"
        )
        return 1

if __name__ == "__main__":
    sys.exit(main())
