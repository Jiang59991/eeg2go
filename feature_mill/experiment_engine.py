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
from logging_config import logger  # 使用全局logger
import traceback

DEFAULT_DB_PATH = "database/eeg2go.db"

# 初始化实验结果管理器
result_manager = ExperimentResultManager(DEFAULT_DB_PATH)


def get_recording_ids_for_dataset(dataset_id: int, db_path: str) -> list[int]:
    """Get all recording IDs for a specified dataset"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT id FROM recordings WHERE dataset_id = ?", (dataset_id,))
    ids = [row[0] for row in c.fetchall()]
    conn.close()
    return ids


def get_fxdef_meta(fxid: int, db_path: str) -> dict:
    """Get feature definition metadata"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    c.execute("SELECT shortname, chans FROM fxdef WHERE id = ?", (fxid,))
    row = c.fetchone()
    conn.close()
    return {
        "shortname": row[0] if row else f"fx{fxid}",
        "chans": row[1] if row else "NA"
    }

def process_recording_result(recording_id: int, fx_values: dict, db_path: str):
    """处理单个recording的结果，返回特征行和统计信息"""
    logger.info(f"Processing results for recording {recording_id}: got {len(fx_values)} feature values")
    
    # Build feature row with recording-level aggregation
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
        
        # Get feature metadata
        fxmeta = get_fxdef_meta(int(fxid), db_path)
        # 修改特征名称生成逻辑，确保连字符格式的一致性
        chans_str = fxmeta['chans']
        if chans_str and "-" in chans_str:
            # 双通道特征：保持连字符格式
            base_name = f"fx{fxid}_{fxmeta['shortname']}_{chans_str}"
        else:
            # 单通道特征：替换逗号为下划线
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
            # 处理1D时间序列数据 - 计算recording级别的统计量
            if isinstance(value, list) and len(value) > 0:
                # 检查是否是包含字典的列表格式（结构化数据）
                if isinstance(value[0], dict) and "value" in value[0]:
                    # 提取所有epoch的数值
                    epoch_values = []
                    for epoch_data in value:
                        if isinstance(epoch_data, dict) and "value" in epoch_data:
                            epoch_value = epoch_data["value"]
                            if isinstance(epoch_value, (int, float, np.number)):
                                epoch_values.append(float(epoch_value))
                    
                    if len(epoch_values) > 0:
                        # 计算recording级别的统计量
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
                    # 普通的1D数组 - 计算统计量
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
            # 处理2D数据 - 计算recording级别的统计量
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


def check_features_exist_in_db(dataset_id: int, feature_set_id: int, db_path: str, min_coverage: float = 0.95) -> dict:
    """
    Check if features for the dataset exist in database with coverage information
    
    Args:
        dataset_id: Dataset ID
        feature_set_id: Feature set ID
        db_path: Database path
        min_coverage: Minimum coverage ratio to consider features as "available" (default: 0.95)
    
    Returns:
        dict: Coverage information with keys:
            - exists: bool - True if coverage >= min_coverage
            - coverage_ratio: float - Actual coverage ratio
            - missing_count: int - Number of missing feature values
            - total_expected: int - Total expected feature values
            - can_use_existing: bool - Whether existing data can be used
    """
    logger.info(f"Checking if features exist in database: dataset_id={dataset_id}, feature_set_id={feature_set_id}")
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Get all recording IDs for the dataset
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
        
        # Get all feature definition IDs for the feature set
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
        
        # Check if all combinations exist in feature_values table
        c.execute("""
            SELECT COUNT(*) FROM feature_values fv
            JOIN recordings r ON fv.recording_id = r.id
            WHERE r.dataset_id = ? AND fv.fxdef_id IN ({})
        """.format(','.join('?' * len(fxdef_ids))), 
        [dataset_id] + fxdef_ids)
        
        actual_count = c.fetchone()[0]
        expected_count = len(recording_ids) * len(fxdef_ids)
        
        logger.info(f"Found {actual_count} feature values, expected {expected_count}")
        
        # Calculate coverage ratio
        coverage_ratio = actual_count / expected_count if expected_count > 0 else 0
        missing_count = expected_count - actual_count
        
        # Determine if we can use existing data
        can_use_existing = coverage_ratio >= min_coverage
        exists = actual_count == expected_count  # 100% coverage
        
        if exists:
            logger.info(f"✅ All features exist in database (coverage: {coverage_ratio:.2%})")
        elif can_use_existing:
            logger.info(f"✅ High coverage features available (coverage: {coverage_ratio:.2%}, missing: {missing_count}) - can use existing data")
        else:
            logger.info(f"❌ Insufficient coverage in database (coverage: {coverage_ratio:.2%}, missing: {missing_count}) - need to compute features")
        
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


def extract_features_with_coverage(dataset_id: int, feature_set_id: int, db_path: str, 
                                  min_coverage: float = 0.95) -> pd.DataFrame:
    """
    Extract feature matrix from database with coverage handling
    
    Args:
        dataset_id: Dataset ID
        feature_set_id: Feature set ID
        db_path: Database path
        min_coverage: Minimum coverage ratio to consider features as available
    
    Returns:
        pd.DataFrame: Feature matrix with recording-level statistics
    """
    logger.info(f"Extracting features with coverage handling: dataset_id={dataset_id}, feature_set_id={feature_set_id}")
    
    # Check coverage
    coverage_info = check_features_exist_in_db(dataset_id, feature_set_id, db_path, min_coverage)
    
    if not coverage_info['can_use_existing']:
        raise ValueError(f"Insufficient coverage ({coverage_info['coverage_ratio']:.2%}) to use existing data. Need to compute features.")
    
    if coverage_info['exists']:
        # 100% coverage - use existing data directly
        logger.info("100% coverage - extracting all features from database")
        return extract_features_from_db(dataset_id, feature_set_id, db_path)
    else:
        # High coverage but some missing - extract what we have and handle missing values
        logger.info(f"High coverage ({coverage_info['coverage_ratio']:.2%}) - extracting available features and handling missing values")
        return extract_features_from_db_partial(dataset_id, feature_set_id, db_path, coverage_info)


def extract_features_from_db_partial(dataset_id: int, feature_set_id: int, db_path: str, 
                                   coverage_info: dict) -> pd.DataFrame:
    """
    Extract feature matrix from database when some features are missing
    
    Args:
        dataset_id: Dataset ID
        feature_set_id: Feature set ID
        db_path: Database path
        coverage_info: Coverage information from check_features_exist_in_db
    
    Returns:
        pd.DataFrame: Feature matrix with recording-level statistics, missing values handled
    """
    logger.info(f"Extracting features with partial coverage: {coverage_info['coverage_ratio']:.2%}")
    
    conn = sqlite3.connect(db_path)
    
    try:
        # Get all recording IDs for the dataset
        recording_ids = get_recording_ids_for_dataset(dataset_id, db_path)
        logger.info(f"Found {len(recording_ids)} recordings")
        
        if not recording_ids:
            raise ValueError(f"No recordings found for dataset {dataset_id}")
        
        # Get feature set definitions
        fxdefs = load_fxdefs_for_set(feature_set_id)
        logger.info(f"Feature set {feature_set_id} contains {len(fxdefs)} feature definitions")
        
        # Build feature matrix from database
        feature_rows = []
        
        for recording_id in recording_ids:
            logger.info(f"Processing recording {recording_id}")
            
            # Get all feature values for this recording and feature set
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
            
            # Build feature row
            feature_row = {"recording_id": recording_id}
            successful_features = 0
            
            for fxdef_id, value_json, dim, shape_json, notes, shortname, chans in feature_values:
                if value_json is None or value_json == 'null':
                    continue
                
                try:
                    # Parse value and shape
                    value = json.loads(value_json) if isinstance(value_json, str) else value_json
                    shape = json.loads(shape_json) if shape_json else []
                    
                    # Generate feature name
                    chans_str = chans or ""
                    if chans_str and "-" in chans_str:
                        base_name = f"fx{fxdef_id}_{shortname}_{chans_str}"
                    else:
                        base_name = f"fx{fxdef_id}_{shortname}_{chans_str}".replace(",", "_")
                    
                    # Process based on dimension
                    if dim == "scalar":
                        if isinstance(value, (list, np.ndarray)):
                            feature_row[base_name] = value[0]
                        else:
                            feature_row[base_name] = value
                        successful_features += 1
                        
                    elif dim == "1d":
                        # Process 1D data - calculate recording-level statistics
                        if isinstance(value, list) and len(value) > 0:
                            if isinstance(value[0], dict) and "value" in value[0]:
                                # Structured epoch data
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
                                # Simple array data
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
                        # Process 2D data - calculate recording-level statistics
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
        
        # Convert to DataFrame
        df = pd.DataFrame(feature_rows)
        logger.info(f"Feature matrix extracted from database (partial coverage): {df.shape}")
        
        # Log coverage information
        total_possible_features = len(recording_ids) * len(fxdefs)
        actual_features = df.shape[1] - 1  # Subtract recording_id column
        logger.info(f"Coverage achieved: {actual_features}/{total_possible_features} ({actual_features/total_possible_features:.2%})")
        
        return df
        
    except Exception as e:
        logger.error(f"Error extracting features from database: {e}")
        raise
    finally:
        conn.close()


def extract_feature_matrix_direct(dataset_id: int, feature_set_id: int, db_path: str, 
                                 min_coverage: float = 0.95) -> pd.DataFrame:
    """
    Extract feature matrix directly from database, with recording-level aggregation and coverage handling
    
    Args:
        dataset_id: Dataset ID
        feature_set_id: Feature set ID
        db_path: Database path
        min_coverage: Minimum coverage ratio to consider existing data as usable (default: 0.95)
    
    Returns:
        pd.DataFrame: Feature matrix with recording-level statistics
    """
    logger.info(f"Extracting feature matrix with coverage handling: dataset_id={dataset_id}, feature_set_id={feature_set_id}")
    
    # First, check if features exist in database with sufficient coverage
    coverage_info = check_features_exist_in_db(dataset_id, feature_set_id, db_path, min_coverage)
    
    if coverage_info['can_use_existing']:
        logger.info(f"Using existing data with {coverage_info['coverage_ratio']:.2%} coverage")
        return extract_features_with_coverage(dataset_id, feature_set_id, db_path, min_coverage)
    
    logger.info(f"Insufficient coverage ({coverage_info['coverage_ratio']:.2%}), computing missing features...")
    
    # Get recording IDs
    recording_ids = get_recording_ids_for_dataset(dataset_id, db_path)
    logger.info(f"Found {len(recording_ids)} recordings")
    
    if not recording_ids:
        raise ValueError(f"No recordings found for dataset {dataset_id}")
    
    # Get feature set definitions
    try:
        fxdefs = load_fxdefs_for_set(feature_set_id)
        logger.info(f"Feature set {feature_set_id} contains {len(fxdefs)} feature definitions")
        fxdef_ids = [fx["id"] for fx in fxdefs]
        logger.info(f"Feature definition IDs: {fxdef_ids}")
    except Exception as e:
        logger.error(f"Failed to load feature set definitions: {e}")
        raise
    
    # Schedule all recording tasks for parallel processing
    logger.info(f"Scheduling {len(recording_ids)} recording tasks for parallel processing...")
    
    # 检查是否使用本地模式
    use_local_mode = os.getenv('USE_LOCAL_EXECUTOR', 'false').lower() == 'true'
    
    # 本地模式下不需要预先调度任务，直接处理
    if not use_local_mode:
        # Celery模式：调度所有recording任务
        recording_tasks = []
        for recording_id in recording_ids:
            logger.info(f"Scheduling run_feature_set task for recording {recording_id}")
            
            try:
                from task_queue.tasks import run_feature_set_task
                
                # 调度run_feature_set任务
                task = run_feature_set_task.apply_async(
                    args=[feature_set_id, recording_id],
                    queue='recordings'
                )
                recording_tasks.append((recording_id, task))
                
            except ImportError:
                logger.warning(f"Celery not available, using direct run_feature_set for recording {recording_id}")
                # 如果Celery不可用，直接执行
                fx_values = run_feature_set(feature_set_id, recording_id)
                recording_tasks.append((recording_id, fx_values))
    
    if not use_local_mode:
        logger.info(f"All {len(recording_tasks)} recording tasks scheduled, waiting for completion...")
    
    # 等待所有任务完成并收集结果
    feature_rows = []
    failed_count = 0
    failed_features_count = 0
    successful_features_count = 0
    
    if use_local_mode:
        # 本地模式：直接顺序处理所有recording
        logger.info("Local mode: processing recordings sequentially...")
        for recording_id in recording_ids:
            logger.info(f"Processing recording {recording_id} in local mode")
            
            try:
                # 直接调用run_feature_set函数
                fx_values = run_feature_set(feature_set_id, recording_id)
                
                if not fx_values:
                    logger.warning(f"run_feature_set returned None for recording {recording_id}")
                    failed_count += 1
                    continue
                
                # 处理recording结果
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
        # Celery模式：等待异步任务完成
        completed_count = 0
        while completed_count < len(recording_tasks):
            # 检查已完成的任务
            for i, (recording_id, task_or_result) in enumerate(recording_tasks):
                if hasattr(task_or_result, 'ready') and task_or_result.ready() and not hasattr(task_or_result, '_counted'):
                    # 这是Celery任务
                    completed_count += 1
                    task_or_result._counted = True
                    
                    try:
                        # 使用result属性而不是get()方法，避免死锁
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
                    # 这是直接执行的结果
                    completed_count += 1
                    task_or_result._counted = True
                    fx_values = task_or_result
                    
                else:
                    # 任务还未完成或已处理
                    continue
                
                # 处理recording结果
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
        
        # 短暂休眠，避免过度占用CPU
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
    
    # Convert to DataFrame
    df = pd.DataFrame(feature_rows)
    logger.info(f"Final feature matrix shape: {df.shape}")
    
    return df


def get_relevant_metadata(dataset_id: int, db_path: str, target_vars: list = None) -> pd.DataFrame:
    """
    Get metadata relevant to the experiment
    
    Args:
        dataset_id: Dataset ID
        db_path: Database path
        target_vars: List of target variables, if None get all available metadata
    
    Returns:
        pd.DataFrame: Metadata dataframe
    """
    conn = sqlite3.connect(db_path)
    
    # Base query fields
    base_fields = [
        "r.id as recording_id",
        "r.subject_id",
        "r.filename"
    ]
    
    # Dynamically add fields based on target variables
    dynamic_fields = []
    if target_vars:
        for var in target_vars:
            if var in ['age', 'age_days']:
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
        # If no target variables specified, get all common fields
        dynamic_fields = [
            "s.age", "s.sex", "s.race", "s.ethnicity",
            "s.visit_count", "s.icd10_count", "s.medication_count",
            "rm.age_days", "rm.seizure", "rm.spindles", 
            "rm.status", "rm.normal", "rm.abnormal"
        ]
    
    # Remove duplicates and build query
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
):
    """
    Main function to run experiments
    
    Args:
        experiment_type (str): Experiment type, must correspond to a module in feature_mill/experiments folder
        dataset_id (int): Dataset ID
        feature_set_id (int): Feature set ID
        output_dir (str): Result save path
        extra_args (dict): Additional parameters passed to experiment function
        db_path (str): Database path
        min_coverage (float): Minimum coverage ratio to use existing data (default: 0.95)
    
    Returns:
        dict: Experiment result summary
    """
    start_time = datetime.now()
    logger.info(f"Starting experiment: {experiment_type}")
    logger.info(f"Dataset ID: {dataset_id}, Feature Set ID: {feature_set_id}")
    logger.info(f"Output directory: {output_dir}")
    logger.info(f"Minimum coverage threshold: {min_coverage:.1%}")
    
    # Step 1: Prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 2: Extract feature matrix
    logger.info(f"Extracting feature matrix...")
    try:
        df_feat = extract_feature_matrix_direct(dataset_id, feature_set_id, db_path, min_coverage)
        logger.info(f"Feature matrix extraction completed: {df_feat.shape}")
    except Exception as e:
        logger.error(f"Feature matrix extraction failed: {e}")
        raise RuntimeError(f"Feature matrix extraction failed: {e}")
    
    # Step 3: Load metadata
    logger.info(f"Loading metadata...")
    try:
        # Get target variables from extra arguments
        target_vars = extra_args.get('target_vars', ['age', 'sex']) if extra_args else ['age', 'sex']
        df_meta = get_relevant_metadata(dataset_id, db_path, target_vars)
        logger.info(f"Metadata loading completed: {df_meta.shape}")
    except Exception as e:
        logger.error(f"Metadata loading failed: {e}")
        raise RuntimeError(f"Metadata loading failed: {e}")
    
    # Step 4: Save experiment parameters
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
    
    # Step 5: Run experiment
    try:
        # Dynamically import experiment module
        module_name = f"feature_mill.experiments.{experiment_type}"
        logger.info(f"Importing experiment module: {module_name}")
        
        module = importlib.import_module(module_name)
        if not hasattr(module, "run"):
            raise AttributeError(f"Module '{experiment_type}' does not have 'run' function")
        
        # Prepare parameters for experiment function
        experiment_kwargs = extra_args or {}
        
        logger.info(f"Running experiment: {experiment_type}")
        summary = module.run(df_feat, df_meta, output_dir, **experiment_kwargs)
        
        # Step 6: Save result summary
        summary_path = os.path.join(output_dir, "summary.txt")
        with open(summary_path, "w", encoding='utf-8') as f:
            f.write(summary if isinstance(summary, str) else str(summary))
        
        # Step 7: Save to database using result manager
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        # 保存实验运行记录
        try:
            logger.info(f"Saving experiment result to database...")
            experiment_result_id = result_manager.save_experiment_result(
                experiment_type=experiment_type,
                dataset_id=dataset_id,
                feature_set_id=feature_set_id,
                parameters=extra_args or {},
                output_dir=output_dir,
                summary=summary,
                duration=duration
            )
            logger.info(f"Experiment result saved with ID: {experiment_result_id}")
        except Exception as e:
            logger.error(f"Failed to save experiment result to database: {e}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            # 不抛出异常，继续执行，因为实验本身已经成功完成
        
        # 根据实验类型保存特征级别的详细结果
        try:
            if experiment_type == 'correlation':
                # 读取相关性分析结果文件
                correlation_files = [f for f in os.listdir(output_dir) if f.startswith('correlation_') and f.endswith('.csv')]
                for file in correlation_files:
                    if 'summary' not in file:  # 跳过汇总文件
                        target_var = file.replace('correlation_', '').replace('.csv', '')
                        result_file = os.path.join(output_dir, file)
                        if os.path.exists(result_file):
                            try:
                                correlation_df = pd.read_csv(result_file)
                                correlation_results = {
                                    target_var: {
                                        'top_results': correlation_df,
                                        'total_features': len(correlation_df),
                                        'significant_count': len(correlation_df[correlation_df['significant'] == True])
                                    }
                                }
                                result_manager.save_correlation_results(
                                    experiment_result_id, correlation_results, [target_var]
                                )
                            except Exception as e:
                                logger.warning(f"保存相关性结果失败 {file}: {e}")

            elif experiment_type == 'classification':
                # 读取分类分析结果文件
                classification_file = os.path.join(output_dir, 'classification_results.csv')
                if os.path.exists(classification_file):
                    try:
                        classification_df = pd.read_csv(classification_file)
                        # 假设有列: feature, importance_score, target_variable
                        for _, row in classification_df.iterrows():
                            feature_name = row.get('feature')
                            importance_score = row.get('importance_score')
                            target_var = row.get('target_variable', 'N/A')
                            rank_position = row.get('rank_position', None)
                            fxdef_id = result_manager._extract_fxdef_id(feature_name)
                            if fxdef_id is not None:  # 只处理有效的fxdef_id
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

            elif experiment_type == 'feature_selection':
                # 读取特征选择结果文件
                selection_file = os.path.join(output_dir, 'feature_selection_results.csv')
                if os.path.exists(selection_file):
                    try:
                        selection_df = pd.read_csv(selection_file)
                        # 假设有列: feature, selection_score, method, target_variable
                        for _, row in selection_df.iterrows():
                            feature_name = row.get('feature')
                            selection_score = row.get('selection_score')
                            method = row.get('method', 'N/A')
                            target_var = row.get('target_variable', 'N/A')
                            rank_position = row.get('rank_position', None)
                            fxdef_id = result_manager._extract_fxdef_id(feature_name)
                            if fxdef_id is not None:  # 只处理有效的fxdef_id
                                conn = sqlite3.connect(db_path)
                                c = conn.cursor()
                                c.execute("""
                                    INSERT INTO experiment_feature_results (
                                        experiment_result_id, fxdef_id, feature_name, target_variable,
                                        result_type, metric_name, metric_value, metric_unit, rank_position, additional_data
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    experiment_result_id, fxdef_id, feature_name, target_var,
                                    'selection_score', 'selection_score', selection_score, 'score', rank_position,
                                    json.dumps({'method': method})
                                ))
                                conn.commit()
                                conn.close()
                        logger.info("特征选择结果已保存到数据库")
                    except Exception as e:
                        logger.warning(f"保存特征选择结果失败: {e}")

            elif experiment_type == 'feature_statistics':
                # 读取特征统计结果文件
                try:
                    # 读取特征重要性排名
                    importance_file = os.path.join(output_dir, 'feature_importance_ranking.csv')
                    if os.path.exists(importance_file):
                        importance_df = pd.read_csv(importance_file)
                        for _, row in importance_df.iterrows():
                            feature_name = row['feature']
                            importance_score = row['importance_score']
                            rank_position = row.name + 1
                            fxdef_id = result_manager._extract_fxdef_id(feature_name)
                            if fxdef_id is not None:  # 只处理有效的fxdef_id
                                conn = sqlite3.connect(db_path)
                                c = conn.cursor()
                                c.execute("""
                                    INSERT INTO experiment_feature_results (
                                        experiment_result_id, fxdef_id, feature_name, target_variable,
                                        result_type, metric_name, metric_value, metric_unit, rank_position
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    experiment_result_id, fxdef_id, feature_name, 'feature_importance',
                                    'statistics', 'importance_score', importance_score, 'score', rank_position
                                ))
                                conn.commit()
                                conn.close()
                        logger.info(f"特征统计结果已保存到数据库，共{len(importance_df)}个特征")
                    # 读取基本统计信息
                    basic_stats_file = os.path.join(output_dir, 'feature_basic_statistics.csv')
                    if os.path.exists(basic_stats_file):
                        basic_stats_df = pd.read_csv(basic_stats_file, index_col=0)
                        for feature_name, stats in basic_stats_df.iterrows():
                            fxdef_id = result_manager._extract_fxdef_id(feature_name)
                            if fxdef_id is not None and 'std' in stats:  # 只处理有效的fxdef_id
                                conn = sqlite3.connect(db_path)
                                c = conn.cursor()
                                c.execute("""
                                    INSERT INTO experiment_feature_results (
                                        experiment_result_id, fxdef_id, feature_name, target_variable,
                                        result_type, metric_name, metric_value, metric_unit
                                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                                """, (
                                    experiment_result_id, fxdef_id, feature_name, 'feature_variability',
                                    'statistics', 'standard_deviation', stats['std'], 'value'
                                ))
                                conn.commit()
                                conn.close()
                        logger.info("特征基本统计信息已保存到数据库")
                except Exception as e:
                    logger.warning(f"保存特征统计结果失败: {e}")
        except Exception as e:
            logger.error(f"Failed to save feature-level results: {e}")
            logger.error(f"Exception traceback: {traceback.format_exc()}")
            # 不抛出异常，继续执行，因为实验本身已经成功完成
        
        logger.info(f"Experiment completed! Results saved to: {output_dir}")
        logger.info(f"Total duration: {duration:.2f} seconds")
        
        return {
            "status": "success",
            "summary": summary,
            "output_dir": output_dir,
            "duration": duration,
            "experiment_result_id": experiment_result_id if 'experiment_result_id' in locals() else None
        }
        
    except ModuleNotFoundError:
        error_msg = f"Experiment module '{experiment_type}' not found. Please add it under 'feature_mill/experiments/'."
        logger.error(error_msg)
        raise ValueError(error_msg)
    except Exception as e:
        error_msg = f"Experiment '{experiment_type}' failed: {e}"
        logger.error(error_msg)
        raise RuntimeError(error_msg)


def save_experiment_result(db_path, experiment_type, dataset_id, feature_set_id, 
                          output_dir, extra_args, summary=None, duration=None):
    """
    Save experiment results to database
    """
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # Create experiment results table if not exists
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
        
        # Insert experiment result
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


def list_experiments():
    """
    List all available experiment modules
    """
    experiments_dir = os.path.join(os.path.dirname(__file__), "experiments")
    experiments = []
    
    if os.path.exists(experiments_dir):
        for file in os.listdir(experiments_dir):
            if file.endswith('.py') and not file.startswith('__'):
                experiment_name = file[:-3]  # Remove .py extension
                experiments.append(experiment_name)
    
    return experiments


def get_experiment_info(experiment_type: str):
    """
    Get experiment module information
    """
    try:
        # 首先尝试从 experiments 包获取信息
        return get_exp_info_from_package(experiment_type)
    except ValueError:
        # 如果失败，回退到动态导入
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