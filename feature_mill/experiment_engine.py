import os
import json
import importlib
from datetime import datetime
import pandas as pd
import numpy as np
import sqlite3
from eeg2fx.featureset_fetcher import run_feature_set
from eeg2fx.featureset_grouping import load_fxdefs_for_set
from feature_mill.experiments import get_experiment_info as get_exp_info_from_package
from feature_mill.experiment_result_manager import ExperimentResultManager
from logging_config import logger
import traceback

DEFAULT_DB_PATH = "database/eeg2go.db"

result_manager = ExperimentResultManager(DEFAULT_DB_PATH)

def normalize_experiment_parameters(parameters: dict) -> dict:
    """
    Normalize experiment parameters to ensure correct numeric types.
    Args:
        parameters (dict): Experiment parameters.
    Returns:
        dict: Normalized parameters.
    """
    normalized = parameters.copy()
    numeric_params = {
        'n_features': int,
        'n_splits': int,
        'top_n': int,
        'top_n_features': int,
        'min_corr': float,
        'fdr_alpha': float,
        'outlier_threshold': float,
        'test_size': float,
    }
    for param_name, param_type in numeric_params.items():
        if param_name in normalized:
            try:
                normalized[param_name] = param_type(normalized[param_name])
            except (ValueError, TypeError) as e:
                logger.warning(f"Failed to convert parameter {param_name} to {param_type.__name__}: {e}")
                default_values = {
                    'n_features': 20,
                    'n_splits': 5,
                    'top_n': 20,
                    'top_n_features': 20,
                    'min_corr': 0.3,
                    'fdr_alpha': 0.05,
                    'outlier_threshold': 1.5,
                    'test_size': 0.2,
                }
                normalized[param_name] = default_values.get(param_name, normalized[param_name])
    return normalized

def get_recording_ids_for_dataset(dataset_id: int, db_path: str) -> list[int]:
    """
    Get all recording IDs for a specified dataset.
    Args:
        dataset_id (int): Dataset ID.
        db_path (str): Path to database.
    Returns:
        list[int]: List of recording IDs.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id FROM recordings WHERE dataset_id = ?", (dataset_id,))
    ids = [row[0] for row in c.fetchall()]
    conn.close()
    return ids

def get_fxdef_meta(fxid: int, db_path: str) -> dict:
    """
    Get feature definition metadata.
    Args:
        fxid (int): Feature definition ID.
        db_path (str): Path to database.
    Returns:
        dict: Metadata with keys 'shortname' and 'chans'.
    """
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT shortname, chans FROM fxdef WHERE id = ?", (fxid,))
    row = c.fetchone()
    conn.close()
    return {
        "shortname": row[0] if row else f"fx{fxid}",
        "chans": row[1] if row else "NA"
    }

def process_recording_result(recording_id: int, fx_values: dict, db_path: str) -> tuple[dict, int, int]:
    """
    Process the result of a single recording and return the feature row and statistics.
    Args:
        recording_id (int): Recording ID.
        fx_values (dict): Feature values for the recording.
        db_path (str): Path to database.
    Returns:
        tuple: (feature_row, successful_features_count, failed_features_count)
    """
    logger.info(f"Processing results for recording {recording_id}: got {len(fx_values)} feature values")
    feature_row = {"recording_id": recording_id}
    recording_failed_features = 0
    recording_successful_features = 0
    for fxid, fxval in fx_values.items():
        if fxval is None:
            recording_failed_features += 1
            logger.warning(f"Feature {fxid} returned None for recording {recording_id}")
            continue
        if fxval.get("value") is None:
            recording_failed_features += 1
            error_msg = fxval.get('notes', 'Unknown error')
            logger.warning(f"Feature {fxid} failed for recording {recording_id}: {error_msg}")
            continue
        recording_successful_features += 1
        fxmeta = get_fxdef_meta(int(fxid), db_path)
        chans_str = fxmeta['chans']
        if chans_str and "-" in chans_str:
            base_name = f"fx{fxid}_{fxmeta['shortname']}_{chans_str}"
        else:
            base_name = f"fx{fxid}_{fxmeta['shortname']}_{chans_str}".replace(",", "_")
        value = fxval.get("value")
        dim = fxval.get("dim", "scalar")
        if dim == "scalar":
            try:
                feature_row[base_name] = value[0] if isinstance(value, (list, np.ndarray)) else value
            except Exception as e:
                recording_failed_features += 1
                logger.warning(f"Scalar processing failed for feature {fxid}: {e}")
        elif dim == "1d":
            if isinstance(value, list) and len(value) > 0:
                if isinstance(value[0], dict) and "value" in value[0]:
                    epoch_values = []
                    for epoch_data in value:
                        if isinstance(epoch_data, dict) and "value" in epoch_data:
                            epoch_value = epoch_data["value"]
                            if isinstance(epoch_value, (int, float, np.number)):
                                epoch_values.append(float(epoch_value))
                    if len(epoch_values) > 0:
                        epoch_values = np.array(epoch_values)
                        feature_row[f"{base_name}_mean"] = np.mean(epoch_values)
                        feature_row[f"{base_name}_std"] = np.std(epoch_values)
                        feature_row[f"{base_name}_min"] = np.min(epoch_values)
                        feature_row[f"{base_name}_max"] = np.max(epoch_values)
                        feature_row[f"{base_name}_median"] = np.median(epoch_values)
                        feature_row[f"{base_name}_count"] = len(epoch_values)
                    else:
                        recording_failed_features += 1
                        logger.warning(f"No valid epoch values found for feature {fxid}")
                else:
                    numeric_values = []
                    for v in value:
                        if isinstance(v, (int, float, np.number)):
                            numeric_values.append(float(v))
                    if len(numeric_values) > 0:
                        numeric_values = np.array(numeric_values)
                        feature_row[f"{base_name}_mean"] = np.mean(numeric_values)
                        feature_row[f"{base_name}_std"] = np.std(numeric_values)
                        feature_row[f"{base_name}_min"] = np.min(numeric_values)
                        feature_row[f"{base_name}_max"] = np.max(numeric_values)
                        feature_row[f"{base_name}_median"] = np.median(numeric_values)
                        feature_row[f"{base_name}_count"] = len(numeric_values)
                    else:
                        recording_failed_features += 1
                        logger.warning(f"No valid numeric values found for feature {fxid}")
        elif dim == "2d":
            all_values = []
            for epoch_idx, vec in enumerate(value):
                for v in vec:
                    if isinstance(v, (int, float, np.number)):
                        all_values.append(float(v))
            if len(all_values) > 0:
                all_values = np.array(all_values)
                feature_row[f"{base_name}_mean"] = np.mean(all_values)
                feature_row[f"{base_name}_std"] = np.std(all_values)
                feature_row[f"{base_name}_min"] = np.min(all_values)
                feature_row[f"{base_name}_max"] = np.max(all_values)
                feature_row[f"{base_name}_median"] = np.median(all_values)
                feature_row[f"{base_name}_count"] = len(all_values)
            else:
                recording_failed_features += 1
                logger.warning(f"No valid numeric values found for 2D feature {fxid}")
        else:
            recording_failed_features += 1
            logger.warning(f"Unknown dimension '{dim}' for feature {fxid}")
    return feature_row, recording_successful_features, recording_failed_features

def check_features_exist_in_db(
    dataset_id: int,
    feature_set_id: int,
    db_path: str,
    min_coverage: float = 0.95
) -> dict:
    """
    Check if features for the dataset exist in database with coverage information.
    Args:
        dataset_id (int): Dataset ID.
        feature_set_id (int): Feature set ID.
        db_path (str): Path to database.
        min_coverage (float): Minimum coverage ratio to consider features as "available".
    Returns:
        dict: Coverage information.
    """
    logger.info(f"Checking if features exist in database: dataset_id={dataset_id}, feature_set_id={feature_set_id}")
    conn = sqlite3.connect(db_path)
    try:
        c = conn.cursor()
        c.execute("SELECT id FROM recordings WHERE dataset_id = ?", (dataset_id,))
        recording_ids = [row[0] for row in c.fetchall()]
        if not recording_ids:
            logger.warning(f"No recordings found for dataset {dataset_id}")
            return {
                'exists': False,
                'coverage_ratio': 0.0,
                'missing_count': 0,
                'total_expected': 0,
                'can_use_existing': False
            }
        c.execute("""
            SELECT fxdef_id FROM feature_set_items 
            WHERE feature_set_id = ?
        """, (feature_set_id,))
        fxdef_ids = [row[0] for row in c.fetchall()]
        if not fxdef_ids:
            logger.warning(f"No feature definitions found for feature set {feature_set_id}")
            return {
                'exists': False,
                'coverage_ratio': 0.0,
                'missing_count': 0,
                'total_expected': 0,
                'can_use_existing': False
            }
        logger.info(f"Checking {len(recording_ids)} recordings × {len(fxdef_ids)} features = {len(recording_ids) * len(fxdef_ids)} total combinations")
        c.execute("""
            SELECT COUNT(*) FROM feature_values fv
            JOIN recordings r ON fv.recording_id = r.id
            WHERE r.dataset_id = ? AND fv.fxdef_id IN ({})
        """.format(','.join('?' * len(fxdef_ids))), 
        [dataset_id] + fxdef_ids)
        actual_count = c.fetchone()[0]
        expected_count = len(recording_ids) * len(fxdef_ids)
        logger.info(f"Found {actual_count} feature values, expected {expected_count}")
        coverage_ratio = actual_count / expected_count if expected_count > 0 else 0
        missing_count = expected_count - actual_count
        can_use_existing = coverage_ratio >= min_coverage
        exists = actual_count == expected_count
        if exists:
            logger.info(f"All features exist in database (coverage: {coverage_ratio:.1%})")
        elif can_use_existing:
            logger.info(f"High coverage available (coverage: {coverage_ratio:.1%}, missing: {missing_count}) - using existing data")
        else:
            logger.info(f"Insufficient coverage (coverage: {coverage_ratio:.1%}, missing: {missing_count}) - need to compute features")
        return {
            'exists': exists,
            'coverage_ratio': coverage_ratio,
            'missing_count': missing_count,
            'total_expected': expected_count,
            'can_use_existing': can_use_existing
        }
    except Exception as e:
        logger.error(f"Error checking features in database: {e}")
        return {
            'exists': False,
            'coverage_ratio': 0.0,
            'missing_count': 0,
            'total_expected': 0,
            'can_use_existing': False
        }
    finally:
        conn.close()

def extract_features_with_coverage(
    dataset_id: int,
    feature_set_id: int,
    db_path: str,
    min_coverage: float = 0.95
) -> pd.DataFrame:
    """
    Extract feature matrix from database with coverage handling.
    Args:
        dataset_id (int): Dataset ID.
        feature_set_id (int): Feature set ID.
        db_path (str): Path to database.
        min_coverage (float): Minimum coverage ratio to consider features as available.
    Returns:
        pd.DataFrame: Feature matrix with recording-level statistics.
    """
    logger.info(f"Extracting features with coverage handling: dataset_id={dataset_id}, feature_set_id={feature_set_id}")
    coverage_info = check_features_exist_in_db(dataset_id, feature_set_id, db_path, min_coverage)
    if not coverage_info['can_use_existing']:
        raise ValueError(f"Insufficient coverage ({coverage_info['coverage_ratio']:.2%}) to use existing data. Need to compute features.")
    if coverage_info['exists']:
        logger.info("100% coverage - extracting all features from database")
        return extract_features_from_db(dataset_id, feature_set_id, db_path)
    else:
        logger.info(f"High coverage ({coverage_info['coverage_ratio']:.1%}) - extracting available features and handling missing values")
        return extract_features_from_db_partial(dataset_id, feature_set_id, db_path, coverage_info)

def extract_features_from_db_partial(
    dataset_id: int,
    feature_set_id: int,
    db_path: str,
    coverage_info: dict
) -> pd.DataFrame:
    """
    Extract feature matrix from database when some features are missing.
    Args:
        dataset_id (int): Dataset ID.
        feature_set_id (int): Feature set ID.
        db_path (str): Path to database.
        coverage_info (dict): Coverage information.
    Returns:
        pd.DataFrame: Feature matrix with recording-level statistics, missing values handled.
    """
    logger.info(f"Extracting features with partial coverage: {coverage_info['coverage_ratio']:.1%}")
    conn = sqlite3.connect(db_path)
    try:
        recording_ids = get_recording_ids_for_dataset(dataset_id, db_path)
        logger.info(f"Found {len(recording_ids)} recordings")
        if not recording_ids:
            raise ValueError(f"No recordings found for dataset {dataset_id}")
        fxdefs = load_fxdefs_for_set(feature_set_id)
        logger.info(f"Feature set {feature_set_id} contains {len(fxdefs)} feature definitions")
        feature_rows = []
        for recording_id in recording_ids:
            logger.info(f"Processing recording {recording_id}")
            query = """
                SELECT fv.fxdef_id, fv.value, fv.dim, fv.shape, fv.notes,
                       f.shortname, f.chans
                FROM feature_values fv
                JOIN fxdef f ON fv.fxdef_id = f.id
                JOIN feature_set_items fsi ON f.id = fsi.fxdef_id
                WHERE fv.recording_id = ? AND fsi.feature_set_id = ?
            """
            cursor = conn.execute(query, (recording_id, feature_set_id))
            feature_values = cursor.fetchall()
            feature_row = {"recording_id": recording_id}
            successful_features = 0
            for fxdef_id, value_json, dim, shape_json, notes, shortname, chans in feature_values:
                if value_json is None or value_json == 'null':
                    continue
                try:
                    value = json.loads(value_json) if isinstance(value_json, str) else value_json
                    shape = json.loads(shape_json) if shape_json else []
                    chans_str = chans or ""
                    if chans_str and "-" in chans_str:
                        base_name = f"fx{fxdef_id}_{shortname}_{chans_str}"
                    else:
                        base_name = f"fx{fxdef_id}_{shortname}_{chans_str}".replace(",", "_")
                    if dim == "scalar":
                        if isinstance(value, (list, np.ndarray)):
                            feature_row[base_name] = value[0]
                        else:
                            feature_row[base_name] = value
                        successful_features += 1
                    elif dim == "1d":
                        if isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], dict) and "value" in value[0]:
                                epoch_values = []
                                for epoch_data in value:
                                    if isinstance(epoch_data, dict) and "value" in epoch_data:
                                        epoch_value = epoch_data["value"]
                                        if isinstance(epoch_value, (int, float, np.number)):
                                            epoch_values.append(float(epoch_value))
                                if len(epoch_values) > 0:
                                    epoch_values = np.array(epoch_values)
                                    feature_row[f"{base_name}_mean"] = np.mean(epoch_values)
                                    feature_row[f"{base_name}_std"] = np.std(epoch_values)
                                    feature_row[f"{base_name}_min"] = np.min(epoch_values)
                                    feature_row[f"{base_name}_max"] = np.max(epoch_values)
                                    feature_row[f"{base_name}_median"] = np.median(epoch_values)
                                    feature_row[f"{base_name}_count"] = len(epoch_values)
                                    successful_features += 1
                            else:
                                numeric_values = []
                                for v in value:
                                    if isinstance(v, (int, float, np.number)):
                                        numeric_values.append(float(v))
                                if len(numeric_values) > 0:
                                    numeric_values = np.array(numeric_values)
                                    feature_row[f"{base_name}_mean"] = np.mean(numeric_values)
                                    feature_row[f"{base_name}_std"] = np.std(numeric_values)
                                    feature_row[f"{base_name}_min"] = np.min(numeric_values)
                                    feature_row[f"{base_name}_max"] = np.max(numeric_values)
                                    feature_row[f"{base_name}_median"] = np.median(numeric_values)
                                    feature_row[f"{base_name}_count"] = len(numeric_values)
                                    successful_features += 1
                    elif dim == "2d":
                        all_values = []
                        for epoch_idx, vec in enumerate(value):
                            for v in vec:
                                if isinstance(v, (int, float, np.number)):
                                    all_values.append(float(v))
                        if len(all_values) > 0:
                            all_values = np.array(all_values)
                            feature_row[f"{base_name}_mean"] = np.mean(all_values)
                            feature_row[f"{base_name}_std"] = np.std(all_values)
                            feature_row[f"{base_name}_min"] = np.min(all_values)
                            feature_row[f"{base_name}_max"] = np.max(all_values)
                            feature_row[f"{base_name}_median"] = np.median(all_values)
                            feature_row[f"{base_name}_count"] = len(all_values)
                            successful_features += 1
                except Exception as e:
                    logger.warning(f"Error processing feature {fxdef_id} for recording {recording_id}: {e}")
                    continue
            if successful_features > 0:
                feature_rows.append(feature_row)
                logger.info(f"Recording {recording_id}: {successful_features} features processed")
            else:
                logger.warning(f"Recording {recording_id}: no features processed")
        df = pd.DataFrame(feature_rows)
        logger.info(f"Feature matrix extracted from database (partial coverage): {df.shape}")
        total_possible_features = len(recording_ids) * len(fxdefs)
        actual_features = df.shape[1] - 1
        logger.info(f"Coverage achieved: {actual_features}/{total_possible_features} ({actual_features/total_possible_features:.1%})")
        return df
    except Exception as e:
        logger.error(f"Error extracting features from database: {e}")
        raise
    finally:
        conn.close()

def extract_feature_matrix_direct(
    dataset_id: int,
    feature_set_id: int,
    db_path: str,
    min_coverage: float = 0.95
) -> pd.DataFrame:
    """
    Extract feature matrix directly from database, with recording-level aggregation and coverage handling.
    Args:
        dataset_id (int): Dataset ID.
        feature_set_id (int): Feature set ID.
        db_path (str): Path to database.
        min_coverage (float): Minimum coverage ratio to consider existing data as usable.
    Returns:
        pd.DataFrame: Feature matrix with recording-level statistics.
    """
    logger.info(f"Extracting feature matrix with coverage handling: dataset_id={dataset_id}, feature_set_id={feature_set_id}")
    coverage_info = check_features_exist_in_db(dataset_id, feature_set_id, db_path, min_coverage)
    if coverage_info['can_use_existing']:
        logger.info(f"Using existing data with {coverage_info['coverage_ratio']:.1%} coverage")
        return extract_features_with_coverage(dataset_id, feature_set_id, db_path, min_coverage)
    logger.info(f"Insufficient coverage ({coverage_info['coverage_ratio']:.1%}), computing missing features...")
    recording_ids = get_recording_ids_for_dataset(dataset_id, db_path)
    logger.info(f"Found {len(recording_ids)} recordings")
    if not recording_ids:
        raise ValueError(f"No recordings found for dataset {dataset_id}")
    try:
        fxdefs = load_fxdefs_for_set(feature_set_id)
        logger.info(f"Feature set {feature_set_id} contains {len(fxdefs)} feature definitions")
        fxdef_ids = [fx["id"] for fx in fxdefs]
        logger.info(f"Feature definition IDs: {fxdef_ids}")
    except Exception as e:
        logger.error(f"Failed to load feature set definitions: {e}")
        raise
    logger.info(f"Scheduling {len(recording_ids)} recording tasks for parallel processing...")
    use_local_mode = os.getenv('USE_LOCAL_EXECUTOR', 'false').lower() == 'true'
    if not use_local_mode:
        recording_tasks = []
        for recording_id in recording_ids:
            logger.info(f"Scheduling run_feature_set task for recording {recording_id}")
            try:
                from task_queue.tasks import run_feature_set_task
                task = run_feature_set_task.apply_async(
                    args=[feature_set_id, recording_id],
                    queue='recordings'
                )
                recording_tasks.append((recording_id, task))
            except ImportError:
                logger.warning(f"Celery not available, using direct run_feature_set for recording {recording_id}")
                fx_values = run_feature_set(feature_set_id, recording_id)
                recording_tasks.append((recording_id, fx_values))
    if not use_local_mode:
        logger.info(f"All {len(recording_tasks)} recording tasks scheduled, waiting for completion...")
    feature_rows = []
    failed_count = 0
    failed_features_count = 0
    successful_features_count = 0
    if use_local_mode:
        logger.info("Local mode: processing recordings sequentially...")
        for recording_id in recording_ids:
            logger.info(f"Processing recording {recording_id} in local mode")
            try:
                fx_values = run_feature_set(feature_set_id, recording_id)
                if not fx_values:
                    logger.warning(f"run_feature_set returned None for recording {recording_id}")
                    failed_count += 1
                    continue
                feature_row, recording_successful_features, recording_failed_features = process_recording_result(
                    recording_id, fx_values, db_path
                )
                if recording_successful_features > 0:
                    feature_rows.append(feature_row)
                    successful_features_count += recording_successful_features
                else:
                    failed_count += 1
                    logger.warning(f"Recording {recording_id} had no successful features")
                failed_features_count += recording_failed_features
            except Exception as e:
                logger.error(f"Failed to process recording {recording_id}: {e}")
                failed_count += 1
                continue
    else:
        completed_count = 0
        while completed_count < len(recording_tasks):
            for i, (recording_id, task_or_result) in enumerate(recording_tasks):
                if hasattr(task_or_result, 'ready') and task_or_result.ready() and not hasattr(task_or_result, '_counted'):
                    completed_count += 1
                    task_or_result._counted = True
                    try:
                        result = task_or_result.result
                        if result.get('success'):
                            logger.info(f"run_feature_set task completed for recording {recording_id}")
                            fx_values = result.get('result', {})
                        else:
                            error_msg = result.get('error', 'Unknown error')
                            logger.error(f"run_feature_set task failed for recording {recording_id}: {error_msg}")
                            failed_count += 1
                            continue
                    except Exception as e:
                        logger.error(f"Task execution failed for recording {recording_id}: {e}")
                        failed_count += 1
                        continue
                elif not hasattr(task_or_result, 'ready') and not hasattr(task_or_result, '_counted'):
                    completed_count += 1
                    task_or_result._counted = True
                    fx_values = task_or_result
                else:
                    continue
                feature_row, recording_successful_features, recording_failed_features = process_recording_result(
                    recording_id, fx_values, db_path
                )
                if recording_successful_features > 0:
                    feature_rows.append(feature_row)
                    successful_features_count += recording_successful_features
                else:
                    failed_count += 1
                    logger.warning(f"Recording {recording_id} had no successful features")
                failed_features_count += recording_failed_features
            import time
            time.sleep(0.1)
    logger.info(f"All {len(recording_tasks)} recording tasks completed.")
    logger.info(f"Feature extraction completed:")
    logger.info(f"  Successful recordings: {len(feature_rows)}")
    logger.info(f"  Failed recordings: {failed_count}")
    logger.info(f"  Total successful features: {successful_features_count}")
    logger.info(f"  Total failed features: {failed_features_count}")
    if not feature_rows:
        raise ValueError("No successful feature extractions found")
    df = pd.DataFrame(feature_rows)
    logger.info(f"Final feature matrix shape: {df.shape}")
    return df

def get_relevant_metadata(
    dataset_id: int,
    db_path: str,
    target_vars: list = None
) -> pd.DataFrame:
    """
    Get metadata relevant to the experiment.
    Args:
        dataset_id (int): Dataset ID.
        db_path (str): Path to database.
        target_vars (list): List of target variables, if None get all available metadata.
    Returns:
        pd.DataFrame: Metadata dataframe.
    """
    conn = sqlite3.connect(db_path)
    base_fields = [
        "r.id as recording_id",
        "r.subject_id",
        "r.filename"
    ]
    dynamic_fields = []
    if target_vars:
        for var in target_vars:
            if var in ['age', 'age_days', 'age_group', 'age_class']:
                dynamic_fields.extend([
                    "s.age",
                    "rm.age_days"
                ])
            elif var == 'sex':
                dynamic_fields.extend([
                    "s.sex"
                ])
            elif var == 'race':
                dynamic_fields.append("s.race")
            elif var == 'ethnicity':
                dynamic_fields.append("s.ethnicity")
            elif var == 'visit_count':
                dynamic_fields.append("s.visit_count")
            elif var == 'icd10_count':
                dynamic_fields.append("s.icd10_count")
            elif var == 'medication_count':
                dynamic_fields.append("s.medication_count")
            elif var in ['seizure', 'spindles', 'status', 'normal', 'abnormal']:
                dynamic_fields.append(f"rm.{var}")
    else:
        dynamic_fields = [
            "s.age", "s.sex", "s.race", "s.ethnicity",
            "s.visit_count", "s.icd10_count", "s.medication_count",
            "rm.age_days", "rm.seizure", "rm.spindles", 
            "rm.status", "rm.normal", "rm.abnormal"
        ]
    all_fields = base_fields + list(set(dynamic_fields))
    fields_str = ", ".join(all_fields)
    query = f"""
        SELECT {fields_str}
        FROM recordings r
        LEFT JOIN subjects s ON r.subject_id = s.subject_id
        LEFT JOIN recording_metadata rm ON r.id = rm.recording_id
        WHERE r.dataset_id = ?
        ORDER BY r.id
    """
    df_meta = pd.read_sql_query(query, conn, params=(dataset_id,))
    conn.close()
    age_target_vars = [var for var in target_vars if var in ['age_group', 'age_class']] if target_vars else []
    if age_target_vars:
        if 'age' in df_meta.columns and 'age_days' in df_meta.columns:
            age_data = df_meta['age_days'].fillna(df_meta['age'] * 365.25)
            for age_var in age_target_vars:
                if age_var == 'age_group':
                    df_meta['age_group'] = pd.cut(age_data, 
                                                bins=[0, 2*365.25, 12*365.25, 18*365.25, 65*365.25, float('inf')],
                                                labels=['infant', 'child', 'adolescent', 'adult', 'elderly'],
                                                include_lowest=True)
                elif age_var == 'age_class':
                    df_meta['age_class'] = pd.cut(age_data, 
                                                bins=[0, 2*365.25, 12*365.25, 18*365.25, 65*365.25, float('inf')],
                                                labels=['infant', 'child', 'adolescent', 'adult', 'elderly'],
                                                include_lowest=True)
            logger.info(f"Created age columns from age/age_days data: {age_target_vars}")
        elif 'age' in df_meta.columns:
            for age_var in age_target_vars:
                if age_var == 'age_group':
                    df_meta['age_group'] = pd.cut(df_meta['age'], 
                                                bins=[0, 2, 12, 18, 65, float('inf')],
                                                labels=['infant', 'child', 'adolescent', 'adult', 'elderly'],
                                                include_lowest=True)
                elif age_var == 'age_class':
                    df_meta['age_class'] = pd.cut(df_meta['age'], 
                                                bins=[0, 2, 12, 18, 65, float('inf')],
                                                labels=['infant', 'child', 'adolescent', 'adult', 'elderly'],
                                                include_lowest=True)
            logger.info(f"Created age columns from age data: {age_target_vars}")
    logger.info(f"Loaded {len(df_meta)} metadata records, fields: {list(df_meta.columns)}")
    return df_meta

def run_experiment(
    experiment_type: str,
    dataset_id: int,
    feature_set_id: int,
    output_dir: str,
    extra_args: dict = None,
    db_path: str = DEFAULT_DB_PATH,
    min_coverage: float = 0.95
) -> dict:
    """
    Main function to run experiments.
    Args:
        experiment_type (str): Experiment type, must correspond to a module in feature_mill/experiments folder.
        dataset_id (int): Dataset ID.
        feature_set_id (int): Feature set ID.
        output_dir (str): Result save path.
        extra_args (dict): Additional parameters passed to experiment function.
        db_path (str): Database path.
        min_coverage (float): Minimum coverage ratio to use existing data.
    Returns:
        dict: Experiment result summary.
    """
    start_time = datetime.now()
    logger.info(f"Starting experiment: {experiment_type}")
    logger.info(f"Dataset ID: {dataset_id}, Feature Set ID: {feature_set_id}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Minimum coverage threshold: {min_coverage:.1%}")
    os.makedirs(output_dir, exist_ok=True)
    logger.info(f"Extracting feature matrix...")
    try:
        df_feat = extract_feature_matrix_direct(dataset_id, feature_set_id, db_path, min_coverage)
        logger.info(f"Feature matrix extraction completed: {df_feat.shape}")
    except Exception as e:
        logger.error(f"Feature matrix extraction failed: {e}")
        raise RuntimeError(f"Feature matrix extraction failed: {e}")
    logger.info(f"Loading metadata...")
    try:
        logger.info(f"extra_args received: {extra_args}")
        actual_params = extra_args
        if extra_args and 'parameters' in extra_args:
            actual_params = extra_args['parameters']
            logger.info(f"Found nested parameters, using: {actual_params}")
        if actual_params and 'target_var' in actual_params:
            target_vars = [actual_params['target_var']]
            logger.info(f"Frontend sent target_var: {actual_params['target_var']}, converted to target_vars: {target_vars}")
        elif actual_params and 'target_vars' in actual_params:
            target_vars = actual_params['target_vars']
            logger.info(f"Frontend sent target_vars: {target_vars}")
        else:
            target_vars = ['age', 'sex']
            logger.info(f"No target variables specified, using defaults: {target_vars}")
        df_meta = get_relevant_metadata(dataset_id, db_path, target_vars)
        logger.info(f"Metadata loading completed: {df_meta.shape}")
    except Exception as e:
        logger.error(f"Metadata loading failed: {e}")
        raise RuntimeError(f"Metadata loading failed: {e}")
    params = {
        "experiment_type": experiment_type,
        "dataset_id": dataset_id,
        "feature_set_id": feature_set_id,
        "output_dir": output_dir,
        "extra_args": extra_args or {},
        "min_coverage": min_coverage,
        "run_time": start_time.isoformat(),
        "feature_matrix_shape": df_feat.shape,
        "metadata_shape": df_meta.shape,
        "target_variables": target_vars
    }
    params_path = os.path.join(output_dir, "parameters.json")
    with open(params_path, "w", encoding='utf-8') as f:
        json.dump(params, f, indent=2, ensure_ascii=False)
    try:
        module_name = f"feature_mill.experiments.{experiment_type}"
        logger.info(f"Importing experiment module: {module_name}")
        module = importlib.import_module(module_name)
        if not hasattr(module, "run"):
            raise AttributeError(f"Module '{experiment_type}' does not have 'run' function")
        experiment_kwargs = normalize_experiment_parameters(actual_params) if actual_params else {}
        experiment_kwargs['target_vars'] = target_vars
        experiment_kwargs['db_path'] = db_path
        logger.info(f"Running experiment: {experiment_type} with target_vars: {target_vars}")
        logger.info(f"Normalized parameters: {experiment_kwargs}")
        summary = module.run(df_feat, df_meta, output_dir, **experiment_kwargs)
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, "w", encoding='utf-8') as f:
            f.write(summary if isinstance(summary, str) else str(summary))
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        experiment_result_id = None
        try:
            logger.info(f"Saving experiment result to database...")
            summary_str = summary
            if isinstance(summary, dict):
                summary_str = json.dumps(summary, ensure_ascii=False, default=str)
            elif summary is None:
                summary_str = "{}"
            experiment_result_id = result_manager.save_experiment_result(
                experiment_type=experiment_type,
                dataset_id=dataset_id,
                feature_set_id=feature_set_id,
                parameters=extra_args or {},
                output_dir=output_dir,
                summary=summary_str,
                duration=duration
            )
            logger.info(f"Experiment result saved with ID: {experiment_result_id}")
        except Exception as e:
            logger.error(f"Failed to save experiment result to database: {e}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            experiment_result_id = None
        logger.info(f"准备保存特征级别结果，experiment_result_id: {experiment_result_id}")
        try:
            if experiment_type == 'correlation':
                logger.info("开始保存correlation实验的特征级别结果...")
                associations_files = [f for f in os.listdir(output_dir) if f.startswith('associations_') and f.endswith('.csv')]
                if not associations_files:
                    associations_files = [f for f in os.listdir(output_dir) if f.startswith('correlation_') and f.endswith('.csv')]
                logger.info(f"找到结果文件: {associations_files}")
                for file in associations_files:
                    if 'summary' in file:
                        logger.info(f"跳过汇总文件: {file}")
                        continue
                    target_var = file.replace('associations_', '').replace('correlation_', '').replace('.csv', '')
                    result_file = os.path.join(output_dir, file)
                    logger.info(f"处理文件: {file}, 目标变量: {target_var}, 文件路径: {result_file}")
                    if os.path.exists(result_file):
                        try:
                            logger.info(f"读取 CSV 文件: {file}")
                            associations_df = pd.read_csv(result_file)
                            logger.info(f"文件包含 {len(associations_df)} 行数据")
                            logger.info(f"列名: {list(associations_df.columns)}")
                            if 'correlation' in associations_df.columns:
                                associations_df['abs_correlation'] = associations_df['correlation'].abs()
                                associations_df = associations_df.sort_values('abs_correlation', ascending=False)
                            logger.info(f"开始保存 {len(associations_df)} 个特征结果到数据库...")
                            saved_count = 0
                            required_columns = ['feature', 'correlation', 'p_value', 'q_value']
                            missing_columns = [col for col in required_columns if col not in associations_df.columns]
                            if missing_columns:
                                logger.error(f"CSV 文件缺少必要的列: {missing_columns}")
                                logger.error(f"可用的列: {list(associations_df.columns)}")
                                continue
                            current_experiment_result_id = experiment_result_id
                            if current_experiment_result_id is None:
                                logger.warning("experiment_result_id 为 None，尝试从数据库中找到对应的记录")
                                try:
                                    conn = sqlite3.connect(db_path)
                                    c = conn.cursor()
                                    c.execute("""
                                        SELECT id FROM experiment_results 
                                        WHERE experiment_type = ? AND dataset_id = ? AND feature_set_id = ?
                                        ORDER BY run_time DESC LIMIT 1
                                    """, (experiment_type, dataset_id, feature_set_id))
                                    result = c.fetchone()
                                    if result:
                                        current_experiment_result_id = result[0]
                                        logger.info(f"找到对应的 experiment_result_id: {current_experiment_result_id}")
                                    else:
                                        logger.error("在数据库中没有找到对应的实验记录")
                                        continue
                                    conn.close()
                                except Exception as e:
                                    logger.error(f"查找 experiment_result_id 失败: {e}")
                                    continue
                            if current_experiment_result_id is None:
                                logger.error("无法获取有效的 experiment_result_id，跳过特征结果保存")
                                continue
                            for rank, (_, row) in enumerate(associations_df.iterrows(), 1):
                                feature_name = str(row['feature']) if pd.notna(row['feature']) else ''
                                correlation = float(row['correlation']) if pd.notna(row['correlation']) else 0.0
                                p_value = float(row['p_value']) if pd.notna(row['p_value']) else 1.0
                                q_value = float(row['q_value']) if pd.notna(row['q_value']) else 1.0
                                significant = bool(row['significant']) if pd.notna(row['significant']) else False
                                logger.debug(f"处理特征 {rank}: {feature_name}, r={correlation:.3f}, p={p_value:.6f}, q={q_value:.6f}")
                                if not feature_name or feature_name == '' or feature_name == 'nan':
                                    logger.debug(f"跳过无效特征名称: {feature_name}")
                                    continue
                                fxdef_id = result_manager._extract_fxdef_id(feature_name)
                                logger.debug(f"提取的fxdef_id: {fxdef_id}")
                                if q_value < 0.001:
                                    significance = '***'
                                elif q_value < 0.01:
                                    significance = '**'
                                elif q_value < 0.05:
                                    significance = '*'
                                else:
                                    significance = 'ns'
                                try:
                                    conn = sqlite3.connect(db_path)
                                    c = conn.cursor()
                                    c.execute("""
                                        INSERT INTO experiment_feature_results (
                                            experiment_result_id, fxdef_id, feature_name, target_variable,
                                            result_type, metric_name, metric_value, metric_unit,
                                            significance_level, rank_position, additional_data
                                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                    """, (
                                        current_experiment_result_id, fxdef_id, feature_name, target_var,
                                        'correlation', 'correlation_coefficient', correlation, 'correlation',
                                        significance, rank, json.dumps({
                                            'p_value': p_value,
                                            'q_value': q_value,
                                            'significant': significant,
                                            'abs_correlation': abs(correlation)
                                        })
                                    ))
                                    conn.commit()
                                    conn.close()
                                    saved_count += 1
                                    logger.debug(f"成功保存特征结果: {feature_name}, r={correlation:.3f}, q={q_value:.3f}, rank={rank}")
                                except Exception as e:
                                    logger.error(f"保存特征结果失败: {feature_name}, 错误: {e}")
                                    continue
                            logger.info(f"成功保存 {saved_count}/{len(associations_df)} 个特征结果到数据库")
                            csv_content = associations_df.to_csv(index=False)
                            conn = sqlite3.connect(db_path)
                            c = conn.cursor()
                            c.execute("""
                                INSERT INTO experiment_feature_results (
                                    experiment_result_id, fxdef_id, feature_name, target_variable,
                                    result_type, metric_name, metric_value, metric_unit,
                                    significance_level, rank_position, additional_data
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                current_experiment_result_id, None, f'csv_data_{target_var}', target_var,
                                'correlation', 'csv_content', len(associations_df), 'rows',
                                'data', 0, json.dumps({
                                    'csv_content': csv_content,
                                    'file_path': result_file,
                                    'total_rows': len(associations_df),
                                    'columns': list(associations_df.columns)
                                })
                            ))
                            conn.commit()
                            conn.close()
                            logger.info(f"成功保存 CSV 内容到数据库")
                        except Exception as e:
                            logger.error(f"处理文件 {file} 时出错: {e}")
                            import traceback
                            logger.error(f"异常详情: {traceback.format_exc()}")
                            continue
                    else:
                        logger.warning(f"结果文件不存在: {result_file}")
                if isinstance(summary, dict) and 'frontend_summary' in summary:
                    frontend_data = summary['frontend_summary']
                    logger.info("保存前端摘要数据...")
                    current_experiment_result_id = experiment_result_id
                    if current_experiment_result_id is None:
                        try:
                            conn = sqlite3.connect(db_path)
                            c = conn.cursor()
                            c.execute("""
                                SELECT id FROM experiment_results 
                                WHERE experiment_type = ? AND dataset_id = ? AND feature_set_id = ?
                                ORDER BY run_time DESC LIMIT 1
                            """, (experiment_type, dataset_id, feature_set_id))
                            result = c.fetchone()
                            if result:
                                current_experiment_result_id = result[0]
                                logger.info(f"找到对应的 experiment_result_id: {current_experiment_result_id}")
                            conn.close()
                        except Exception as e:
                            logger.error(f"查找 experiment_result_id 失败: {e}")
                    if current_experiment_result_id is not None:
                        if 'overall_significant_features' in frontend_data:
                            conn = sqlite3.connect(db_path)
                            c = conn.cursor()
                            c.execute("""
                                INSERT INTO experiment_feature_results (
                                    experiment_result_id, fxdef_id, feature_name, target_variable,
                                    result_type, metric_name, metric_value, metric_unit
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                current_experiment_result_id, None, 'overall_significant_associations', 'associations_summary',
                                'correlation', 'significant_features_count', frontend_data['overall_significant_features'], 'count'
                            ))
                            conn.commit()
                            conn.close()
                        if 'target_variables' in frontend_data:
                            for target_var, target_data in frontend_data['target_variables'].items():
                                conn = sqlite3.connect(db_path)
                                c = conn.cursor()
                                c.execute("""
                                    INSERT INTO experiment_feature_results (
                                        experiment_result_id, fxdef_id, feature_name, target_variable,
                                        result_type, metric_name, metric_value, metric_unit, additional_data
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    current_experiment_result_id, None, f'associations_{target_var}', target_var,
                                    'correlation', 'significant_associations', target_data['significant_count'], 'count',
                                    json.dumps({
                                        'target_type': target_data['type'],
                                        'total_features': target_data['total_features'],
                                        'significant_ratio': target_data['significant_ratio']
                                    })
                                ))
                                conn.commit()
                                conn.close()
                        logger.info("关联性分析前端摘要数据已保存到数据库")
                    else:
                        logger.error("无法获取有效的 experiment_result_id，跳过前端摘要数据保存")
            elif experiment_type == 'classification':
                if isinstance(summary, dict) and 'frontend_summary' in summary:
                    frontend_data = summary['frontend_summary']
                    if 'overall_performance' in frontend_data:
                        overall_perf = frontend_data['overall_performance']
                        conn = sqlite3.connect(db_path)
                        c = conn.cursor()
                        c.execute("""
                            INSERT INTO experiment_feature_results (
                                experiment_result_id, fxdef_id, feature_name, target_variable,
                                result_type, metric_name, metric_value, metric_unit
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            experiment_result_id, None, 'overall_classification_performance', 'classification_summary',
                            'classification', 'average_f1_score', overall_perf.get('average_f1', 0), 'score'
                        ))
                        conn.commit()
                        conn.close()
                    if 'target_performance' in frontend_data:
                        for target_var, target_data in frontend_data['target_performance'].items():
                            conn = sqlite3.connect(db_path)
                            c = conn.cursor()
                            c.execute("""
                                INSERT INTO experiment_feature_results (
                                    experiment_result_id, fxdef_id, feature_name, target_variable,
                                    result_type, metric_name, metric_value, metric_unit, additional_data
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                experiment_result_id, None, f'classification_{target_var}', target_var,
                                'classification', 'model_performance', target_data.get('f1_score', 0), 'score',
                                json.dumps({
                                    'model': target_data.get('model', 'Unknown'),
                                    'accuracy': target_data.get('accuracy', 0),
                                    'roc_auc': target_data.get('roc_auc', 0),
                                    'cv_mean': target_data.get('cv_mean', 0),
                                    'cv_std': target_data.get('cv_std', 0)
                                })
                            ))
                            conn.commit()
                            conn.close()
                    logger.info("分类分析前端摘要数据已保存到数据库")
                classification_file = os.path.join(output_dir, 'classification_results.csv')
                if os.path.exists(classification_file):
                    try:
                        classification_df = pd.read_csv(classification_file)
                        for _, row in classification_df.iterrows():
                            feature_name = row.get('feature')
                            importance_score = row.get('importance_score')
                            target_var = row.get('target_variable', 'N/A')
                            rank_position = row.get('rank_position', None)
                            fxdef_id = result_manager._extract_fxdef_id(feature_name)
                            if fxdef_id is not None:
                                conn = sqlite3.connect(db_path)
                                c = conn.cursor()
                                c.execute("""
                                    INSERT INTO experiment_feature_results (
                                        experiment_result_id, fxdef_id, feature_name, target_variable,
                                        result_type, metric_name, metric_value, metric_unit, rank_position
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    experiment_result_id, fxdef_id, feature_name, target_var,
                                    'classification_importance', 'importance_score', importance_score, 'score', rank_position
                                ))
                                conn.commit()
                                conn.close()
                        logger.info("分类特征重要性结果已保存到数据库")
                    except Exception as e:
                        logger.warning(f"保存分类结果失败: {e}")
            elif experiment_type == 'feature_statistics':
                if isinstance(summary, dict) and 'frontend_summary' in summary:
                    frontend_data = summary['frontend_summary']
                    if 'overall_health' in frontend_data:
                        overall_health = frontend_data['overall_health']
                        conn = sqlite3.connect(db_path)
                        c = conn.cursor()
                        c.execute("""
                            INSERT INTO experiment_feature_results (
                                experiment_result_id, fxdef_id, feature_name, target_variable,
                                result_type, metric_name, metric_value, metric_unit
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            experiment_result_id, None, 'overall_dataset_health', 'statistics_summary',
                            'statistics', 'health_score', overall_health.get('health_score', 0), 'score'
                        ))
                        conn.commit()
                        conn.close()
                    if 'quality_distribution' in frontend_data:
                        quality_dist = frontend_data['quality_distribution']
                        for grade, count in quality_dist.items():
                            conn = sqlite3.connect(db_path)
                            c = conn.cursor()
                            c.execute("""
                                INSERT INTO experiment_feature_results (
                                    experiment_result_id, fxdef_id, feature_name, target_variable,
                                    result_type, metric_name, metric_value, metric_unit
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                experiment_result_id, None, f'quality_grade_{grade}', 'quality_distribution',
                                'statistics', 'feature_count', count, 'count'
                            ))
                            conn.commit()
                            conn.close()
                    if 'quality_issues' in frontend_data:
                        quality_issues = frontend_data['quality_issues']
                        for issue_type, count in quality_issues.items():
                            conn = sqlite3.connect(db_path)
                            c = conn.cursor()
                            c.execute("""
                                INSERT INTO experiment_feature_results (
                                    experiment_result_id, fxdef_id, feature_name, target_variable,
                                    result_type, metric_name, metric_value, metric_unit
                                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                            """, (
                                experiment_result_id, None, f'quality_issue_{issue_type}', 'quality_issues',
                                'statistics', 'issue_count', count, 'count'
                            ))
                            conn.commit()
                            conn.close()
                    logger.info("特征统计前端摘要数据已保存到数据库")
                stats_file = os.path.join(output_dir, 'feature_statistics.csv')
                if os.path.exists(stats_file):
                    try:
                        stats_df = pd.read_csv(stats_file)
                        for _, row in stats_df.iterrows():
                            feature_name = row.get('feature')
                            stat_type = row.get('statistic_type', 'unknown')
                            stat_value = row.get('value', 0)
                            fxdef_id = result_manager._extract_fxdef_id(feature_name)
                            if fxdef_id is not None:
                                conn = sqlite3.connect(db_path)
                                c = conn.cursor()
                                c.execute("""
                                    INSERT INTO experiment_feature_results (
                                        experiment_result_id, fxdef_id, feature_name, target_variable,
                                        result_type, metric_name, metric_value, metric_unit
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    experiment_result_id, fxdef_id, feature_name, 'statistics',
                                    'feature_statistic', stat_type, stat_value, 'value'
                                ))
                                conn.commit()
                                conn.close()
                        logger.info("特征统计结果已保存到数据库")
                    except Exception as e:
                        logger.warning(f"保存特征统计结果失败: {e}")
        except Exception as e:
            logger.error(f"保存特征级别结果时出错: {e}")
            import traceback
            logger.error(f"异常详情: {traceback.format_exc()}")
        return {
            'status': 'success',
            'output_dir': output_dir,
            'summary': summary,
            'duration': duration
        }
    except ModuleNotFoundError:
        error_msg = f"Experiment module '{experiment_type}' not found. Please add it under 'feature_mill/experiments/'."
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Experiment '{experiment_type}' failed: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)

def save_experiment_result(
    db_path,
    experiment_type,
    dataset_id,
    feature_set_id,
    output_dir,
    extra_args,
    summary=None,
    duration=None
) -> None:
    """
    Save experiment results to database.
    Args:
        db_path (str): Path to database.
        experiment_type (str): Experiment type.
        dataset_id (int): Dataset ID.
        feature_set_id (int): Feature set ID.
        output_dir (str): Output directory.
        extra_args (dict): Extra arguments.
        summary (str): Result summary.
        duration (float): Duration in seconds.
    Returns:
        None
    """
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        c.execute("""
            CREATE TABLE IF NOT EXISTS experiment_results (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                experiment_type TEXT,
                dataset_id INTEGER,
                feature_set_id INTEGER,
                run_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
                duration_seconds REAL,
                parameters TEXT,
                result_path TEXT,
                result_summary TEXT,
                notes TEXT
            )
        """)
        c.execute("""
            INSERT INTO experiment_results (
                experiment_type, dataset_id, feature_set_id,
                duration_seconds, parameters, result_path, result_summary
            )
            VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            experiment_type,
            dataset_id,
            feature_set_id,
            duration,
            json.dumps(extra_args or {}, ensure_ascii=False),
            output_dir,
            summary if summary else ""
        ))
        conn.commit()
        conn.close()
        logger.info("Experiment results saved to database")
    except Exception as e:
        logger.warning(f"Failed to save to database: {e}")

def list_experiments() -> list:
    """
    List all available experiment modules.
    Returns:
        list: List of experiment module names.
    """
    experiments_dir = os.path.join(os.path.dirname(__file__), "experiments")
    experiments = []
    if os.path.exists(experiments_dir):
        for file in os.listdir(experiments_dir):
            if file.endswith('.py') and not file.startswith('__'):
                experiment_name = file[:-3]
                experiments.append(experiment_name)
    return experiments

def get_experiment_info(experiment_type: str) -> dict:
    """
    Get experiment module information.
    Args:
        experiment_type (str): Experiment type.
    Returns:
        dict: Information about the experiment module.
    """
    try:
        return get_exp_info_from_package(experiment_type)
    except ValueError:
        try:
            module = importlib.import_module(f"feature_mill.experiments.{experiment_type}")
            info = {
                "name": experiment_type,
                "module": module.__name__,
                "has_run_function": hasattr(module, "run"),
                "docstring": module.__doc__ or "No documentation"
            }
            if hasattr(module, "run"):
                info["run_docstring"] = module.run.__doc__ or "No function documentation"
            return info
        except ModuleNotFoundError:
            return {"error": f"Experiment module '{experiment_type}' not found"}
        except Exception as e:
            return {"error": f"Failed to get experiment info: {e}"} 