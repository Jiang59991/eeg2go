#!/usr/bin/env python3
"""
Feature Extraction Validator - Test all_available_features feature set

This script is used to:
1. Validate the code of all_available_features feature set
2. Extract all_available_features for a specific dataset
3. Generate a validation report

Usage:
python feature_extraction_validator.py [dataset_id] [max_recordings]
"""

import os
import sys
import time
import json
import sqlite3
import argparse
from datetime import datetime
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from eeg2fx.function_registry import FEATURE_FUNCS
from eeg2fx.featureset_fetcher import run_feature_set
from eeg2fx.featureset_grouping import load_fxdefs_for_set
from eeg2fx.feature_saver import get_failed_features_stats

DB_PATH = os.path.join(project_root, "database", "eeg2go.db")

FEATURESET_NAME = "all_available_features"

class FeatureSetValidator:
    """Validator for all_available_features"""

    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.validation_results = {
            "start_time": datetime.now().isoformat(),
            "feature_validation": {},
            "extraction_validation": {},
            "summary": {}
        }

    def get_featureset_id(self):
        """Get all_available_features feature set ID"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id, name, description FROM feature_sets WHERE name = ?", (FEATURESET_NAME,))
        row = c.fetchone()
        if row:
            set_id, name, description = row
            print(f"Found feature set '{name}' (ID: {set_id})")
            print(f"Description: {description}")
            c.execute("SELECT COUNT(*) FROM feature_set_items WHERE feature_set_id = ?", (set_id,))
            feature_count = c.fetchone()[0]
            print(f"Feature count: {feature_count}")
            c.execute("""
                SELECT f.id, f.shortname, f.func, f.chans, f.params, f.dim, p.shortname as pipeline_name
                FROM feature_set_items fsi
                JOIN fxdef f ON fsi.fxdef_id = f.id
                LEFT JOIN pipedef p ON f.pipedef_id = p.id
                WHERE fsi.feature_set_id = ?
                ORDER BY f.id
            """, (set_id,))
            features = c.fetchall()
            print(f"\nFeature details:")
            for i, (fxid, shortname, func, chans, params, dim, pipeline) in enumerate(features, 1):
                print(f"  {i:2d}. ID:{fxid:3d} | {shortname:20s} | {func:20s} | {chans:15s} | {dim:8s} | Pipeline:{pipeline}")
        else:
            print(f"✗ Feature set '{FEATURESET_NAME}' not found")
            print("Please create the feature set first.")
            set_id = None
        conn.close()
        return set_id

    def validate_functions(self, fxdefs):
        """Validate all feature functions in the set"""
        print("=" * 60)
        print(f"Validating feature functions in '{FEATURESET_NAME}'")
        print("=" * 60)
        validation_results = {}
        for fxdef in fxdefs:
            func_name = fxdef["func"]
            print(f"\nValidating function: {func_name}")
            result = {
                "status": "unknown",
                "error": None,
                "function_info": {}
            }
            try:
                if func_name not in FEATURE_FUNCS:
                    result["status"] = "error"
                    result["error"] = f"Function '{func_name}' not in FEATURE_FUNCS"
                    print(f"  ✗ Function '{func_name}' not in FEATURE_FUNCS")
                    continue
                func = FEATURE_FUNCS[func_name]
                if func is None:
                    result["status"] = "error"
                    result["error"] = "Function object is None"
                    print(f"  ✗ Function object is None")
                    continue
                import inspect
                sig = inspect.signature(func)
                params = list(sig.parameters.keys())
                result["function_info"] = {
                    "name": func.__name__,
                    "module": func.__module__,
                    "signature": str(sig)
                }
                if "epochs" not in params:
                    result["status"] = "error"
                    result["error"] = "Missing required 'epochs' parameter"
                    print(f"  ✗ Missing required 'epochs' parameter")
                else:
                    result["status"] = "valid"
                    print(f"  ✓ Function signature valid")
            except Exception as e:
                result["status"] = "error"
                result["error"] = str(e)
                print(f"  ✗ Validation failed: {e}")
            validation_results[func_name] = result
        self.validation_results["feature_validation"] = validation_results
        valid_count = sum(1 for r in validation_results.values() if r["status"] == "valid")
        error_count = sum(1 for r in validation_results.values() if r["status"] == "error")
        print(f"\nFeature function validation results:")
        print(f"  Valid: {valid_count}")
        print(f"  Error: {error_count}")
        print(f"  Total: {len(validation_results)}")
        return validation_results

    def get_dataset_recordings(self, dataset_id):
        """Get all recordings for a dataset"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("""
            SELECT id, filename, subject_id, duration, channels, sampling_rate
            FROM recordings 
            WHERE dataset_id = ?
            ORDER BY id
        """, (dataset_id,))
        recordings = []
        for row in c.fetchall():
            recordings.append({
                "id": row[0],
                "filename": row[1],
                "subject_id": row[2],
                "duration": row[3],
                "channels": row[4],
                "sampling_rate": row[5]
            })
        conn.close()
        return recordings

    def validate_extraction(self, feature_set_id, fxdefs, dataset_id, max_recordings=None):
        """Validate feature extraction for all_available_features"""
        print("\n" + "=" * 60)
        print(f"Validating feature extraction for '{FEATURESET_NAME}'")
        print("=" * 60)
        recordings = self.get_dataset_recordings(dataset_id)
        if not recordings:
            print(f"✗ No recordings found in dataset {dataset_id}")
            return None
        print(f"Found {len(recordings)} recordings")
        if max_recordings and len(recordings) > max_recordings:
            recordings = recordings[:max_recordings]
            print(f"Limiting to first {max_recordings} recordings")
        extraction_results = {
            "feature_set_id": feature_set_id,
            "dataset_id": dataset_id,
            "recordings_processed": 0,
            "recordings_successful": 0,
            "recordings_failed": 0,
            "total_features": len(fxdefs),
            "recording_results": [],
            "feature_results": {}
        }
        for fxdef in fxdefs:
            extraction_results["feature_results"][fxdef['func']] = {
                "successful": 0,
                "failed": 0,
                "errors": []
            }
        for i, recording in enumerate(recordings, 1):
            print(f"\nProcessing recording {i}/{len(recordings)}: {recording['filename']}")
            print(f"  ID: {recording['id']}, Duration: {recording['duration']:.1f}s, Channels: {recording['channels']}")
            start_time = time.time()
            try:
                results = run_feature_set(feature_set_id, recording['id'])
                successful_features = 0
                failed_features = 0
                for fxdef_id, result in results.items():
                    if result["value"] is not None:
                        successful_features += 1
                        for fxdef in fxdefs:
                            if fxdef['id'] == fxdef_id:
                                extraction_results["feature_results"][fxdef['func']]["successful"] += 1
                                break
                    else:
                        failed_features += 1
                        for fxdef in fxdefs:
                            if fxdef['id'] == fxdef_id:
                                extraction_results["feature_results"][fxdef['func']]["failed"] += 1
                                if "error" in result:
                                    extraction_results["feature_results"][fxdef['func']]["errors"].append(result["error"])
                                break
                processing_time = time.time() - start_time
                recording_result = {
                    "recording_id": recording['id'],
                    "filename": recording['filename'],
                    "status": "success",
                    "processing_time": processing_time,
                    "successful_features": successful_features,
                    "failed_features": failed_features,
                    "error": None
                }
                extraction_results["recordings_successful"] += 1
                print(f"  ✓ Extracted {successful_features}/{len(results)} features ({processing_time:.2f}s)")
            except Exception as e:
                processing_time = time.time() - start_time
                recording_result = {
                    "recording_id": recording['id'],
                    "filename": recording['filename'],
                    "status": "failed",
                    "processing_time": processing_time,
                    "successful_features": 0,
                    "failed_features": len(fxdefs),
                    "error": str(e)
                }
                extraction_results["recordings_failed"] += 1
                print(f"  ✗ Feature extraction failed: {e}")
            extraction_results["recording_results"].append(recording_result)
            extraction_results["recordings_processed"] += 1
        total_successful = sum(r["successful_features"] for r in extraction_results["recording_results"])
        total_failed = sum(r["failed_features"] for r in extraction_results["recording_results"])
        extraction_results["summary"] = {
            "total_successful_features": total_successful,
            "total_failed_features": total_failed,
            "success_rate": extraction_results["recordings_successful"] / extraction_results["recordings_processed"] if extraction_results["recordings_processed"] > 0 else 0,
            "feature_success_rate": total_successful / (total_successful + total_failed) if (total_successful + total_failed) > 0 else 0
        }
        print(f"\nFeature extraction validation results:")
        print(f"  Recordings processed: {extraction_results['recordings_processed']}")
        print(f"  Successful: {extraction_results['recordings_successful']}")
        print(f"  Failed: {extraction_results['recordings_failed']}")
        print(f"  Recording success rate: {extraction_results['summary']['success_rate']:.2%}")
        print(f"  Feature success rate: {extraction_results['summary']['feature_success_rate']:.2%}")
        print(f"\nFeature-wise results:")
        for func_name, stats in extraction_results["feature_results"].items():
            total = stats["successful"] + stats["failed"]
            if total > 0:
                success_rate = stats["successful"] / total
                print(f"  {func_name:20s}: {stats['successful']:3d}/{total:3d} ({success_rate:.1%})")
                if stats["errors"]:
                    print(f"    Common error: {stats['errors'][0][:50]}...")
        self.validation_results["extraction_validation"] = extraction_results
        return extraction_results

    def generate_validation_report(self, output_file=None):
        """Generate validation report"""
        print("\n" + "=" * 60)
        print("Generating validation report")
        print("=" * 60)
        self.validation_results["end_time"] = datetime.now().isoformat()
        try:
            failed_stats = get_failed_features_stats()
            self.validation_results["failed_features_stats"] = failed_stats
            print("✓ Got failed features stats")
        except Exception as e:
            print(f"✗ Failed to get failed features stats: {e}")
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"all_features_validation_report_{timestamp}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
        print(f"✓ Validation report saved to: {output_file}")
        self.print_summary()
        return output_file

    def print_summary(self):
        """Print validation summary"""
        print("\n" + "=" * 60)
        print("All Features Validation Summary")
        print("=" * 60)
        feature_validation = self.validation_results.get("feature_validation", {})
        if feature_validation:
            valid_funcs = sum(1 for r in feature_validation.values() if r["status"] == "valid")
            error_funcs = sum(1 for r in feature_validation.values() if r["status"] == "error")
            print(f"Feature function validation:")
            print(f"  ✓ Valid functions: {valid_funcs}")
            print(f"  ✗ Error functions: {error_funcs}")
            if error_funcs > 0:
                print(f"  Error function list:")
                for func_name, result in feature_validation.items():
                    if result["status"] == "error":
                        print(f"    - {func_name}: {result['error']}")
        extraction_validation = self.validation_results.get("extraction_validation", {})
        if extraction_validation:
            summary = extraction_validation.get("summary", {})
            print(f"Feature extraction validation:")
            print(f"  Recordings processed: {extraction_validation.get('recordings_processed', 0)}")
            print(f"  ✓ Successful: {extraction_validation.get('recordings_successful', 0)}")
            print(f"  ✗ Failed: {extraction_validation.get('recordings_failed', 0)}")
            print(f"  Recording success rate: {summary.get('success_rate', 0):.2%}")
            print(f"  Feature success rate: {summary.get('feature_success_rate', 0):.2%}")
            feature_results = extraction_validation.get("feature_results", {})
            if feature_results:
                print(f"  Feature-wise success rate:")
                for func_name, stats in feature_results.items():
                    total = stats["successful"] + stats["failed"]
                    if total > 0:
                        success_rate = stats["successful"] / total
                        status = "✓" if success_rate > 0.8 else "⚠" if success_rate > 0.5 else "✗"
                        print(f"    {status} {func_name:20s}: {success_rate:.1%}")
        failed_stats = self.validation_results.get("failed_features_stats", {})
        if failed_stats:
            print(f"Failed features stats:")
            print(f"  Total failed features: {failed_stats.get('total_failed', 0)}")
            if failed_stats.get('common_errors'):
                print(f"  Common errors:")
                for error, count in list(failed_stats['common_errors'].items())[:3]:
                    print(f"    - {error[:50]}... ({count} times)")

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description="EEG all_available_features extraction validation tool")
    parser.add_argument("dataset_id", type=int, help="Dataset ID")
    parser.add_argument("--max_recordings", type=int, default=None, 
                       help="Max number of recordings to process (default: all)")
    parser.add_argument("--output", type=str, default=None,
                       help="Output report file name (default: auto-generated)")
    args = parser.parse_args()
    print("EEG all_available_features extraction validation tool")
    print("=" * 60)
    print(f"Dataset ID: {args.dataset_id}")
    print(f"Max recordings: {args.max_recordings or 'all'}")
    validator = FeatureSetValidator()
    # featureset_id = validator.get_featureset_id()
    featureset_id = 5
    if not featureset_id:
        print("✗ Cannot get all_available_features feature set ID")
        return
    fxdefs = load_fxdefs_for_set(featureset_id)
    validator.validate_functions(fxdefs)
    validator.validate_extraction(featureset_id, fxdefs, args.dataset_id, args.max_recordings)
    report_file = validator.generate_validation_report(args.output)
    print(f"\nValidation complete! Detailed report saved to: {report_file}")

if __name__ == "__main__":
    main() 