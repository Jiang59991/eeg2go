import os
import json
import importlib
import logging
from datetime import datetime
import pandas as pd
import numpy as np
import sqlite3
from eeg2fx.featureset_fetcher import run_feature_set
from eeg2fx.featureset_grouping import load_fxdefs_for_set

DEFAULT_DB_PATH = "database/eeg2go.db"

# Setup basic logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def setup_logging(log_file=None, log_level=logging.INFO):
    """
    Setup logging configuration with both console and file output
    
    Args:
        log_file: Path to log file. If None, only console output is used
        log_level: Logging level (default: INFO)
    """
    # Clear existing handlers
    logger = logging.getLogger(__name__)
    for handler in logger.handlers[:]:
        logger.removeHandler(handler)
    
    # Create formatter
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    
    # Console handler
    console_handler = logging.StreamHandler()
    console_handler.setLevel(log_level)
    console_handler.setFormatter(formatter)
    logger.addHandler(console_handler)
    
    # File handler (if specified)
    if log_file:
        # Ensure log directory exists
        log_dir = os.path.dirname(log_file)
        if log_dir and not os.path.exists(log_dir):
            os.makedirs(log_dir)
        
        file_handler = logging.FileHandler(log_file, mode='a', encoding='utf-8')
        file_handler.setLevel(log_level)
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    
    logger.setLevel(log_level)
    return logger


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


def extract_feature_matrix_direct(dataset_id: int, feature_set_id: int, db_path: str) -> pd.DataFrame:
    """
    Extract feature matrix directly from database, with recording-level aggregation
    
    Args:
        dataset_id: Dataset ID
        feature_set_id: Feature set ID
        db_path: Database path
    
    Returns:
        pd.DataFrame: Feature matrix with recording-level statistics
    """
    logger.info(f"Extracting feature matrix directly from database: dataset_id={dataset_id}, feature_set_id={feature_set_id}")
    
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
    
    # Extract features
    feature_rows = []
    failed_count = 0
    failed_features_count = 0
    successful_features_count = 0
    
    # 添加调试：记录第一个recording的详细信息
    first_recording_debug = True
    
    for i, recording_id in enumerate(recording_ids):
        try:
            logger.info(f"Processing recording {i+1}/{len(recording_ids)}: recording_id={recording_id}")
            
            # Get all feature values for this recording
            fx_values = run_feature_set(feature_set_id, recording_id)
            logger.info(f"Recording {recording_id}: got {len(fx_values)} feature values")
            
            # 添加调试：查看第一个recording的特征数据
            if first_recording_debug:
                logger.info("=== FIRST RECORDING FEATURE DATA DEBUG ===")
                for fxid, fxval in list(fx_values.items())[:3]:  # 只查看前3个特征
                    logger.info(f"Feature {fxid}:")
                    logger.info(f"  Type: {type(fxval)}")
                    logger.info(f"  Keys: {list(fxval.keys()) if isinstance(fxval, dict) else 'Not a dict'}")
                    logger.info(f"  Value: {fxval.get('value')}")
                    logger.info(f"  Dim: {fxval.get('dim')}")
                    logger.info(f"  Notes: {fxval.get('notes')}")
                    
                    # 详细查看value的结构
                    value = fxval.get('value')
                    if value is not None:
                        logger.info(f"  Value type: {type(value)}")
                        logger.info(f"  Value length: {len(value) if isinstance(value, (list, np.ndarray)) else 'N/A'}")
                        if isinstance(value, list) and len(value) > 0:
                            logger.info(f"  First element type: {type(value[0])}")
                            logger.info(f"  First element: {value[0]}")
                            if isinstance(value[0], dict):
                                logger.info(f"  First element keys: {list(value[0].keys())}")
                logger.info("=== END FIRST RECORDING DEBUG ===")
                first_recording_debug = False
            
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
                base_name = f"fx{fxid}_{fxmeta['shortname']}_{fxmeta['chans']}".replace(",", "_")
                
                value = fxval.get("value")
                dim = fxval.get("dim", "scalar")
                
                # 添加调试：查看特征处理过程
                if i == 0:  # 只对第一个recording输出详细信息
                    logger.info(f"Processing feature {fxid}: dim={dim}, value_type={type(value).__name__}")
                    if isinstance(value, list) and len(value) > 0:
                        logger.info(f"  Value length: {len(value)}")
                        logger.info(f"  First element: {value[0]}")
                
                if dim == "scalar":
                    try:
                        feature_row[base_name] = value[0] if isinstance(value, (list, np.ndarray)) else value
                        if i == 0:
                            logger.info(f"  Scalar feature {base_name}: {feature_row[base_name]}")
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
                                
                                if i == 0:
                                    logger.info(f"  Extracted {len(epoch_values)} epoch values for feature {base_name}")
                                    logger.info(f"  Recording-level stats: mean={feature_row[f'{base_name}_mean']:.4f}, std={feature_row[f'{base_name}_std']:.4f}")
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
            
            if recording_successful_features > 0:
                feature_rows.append(feature_row)
                successful_features_count += recording_successful_features
            else:
                failed_count += 1
                logger.warning(f"Recording {recording_id} had no successful features")
            
            failed_features_count += recording_failed_features
            
        except Exception as e:
            failed_count += 1
            logger.error(f"Failed to process recording {recording_id}: {e}")
            continue
    
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
    log_file: str = None
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
        log_file (str): Path to log file. If None, only console output is used
    
    Returns:
        dict: Experiment result summary
    """
    # Setup logging with file output if specified
    if log_file:
        setup_logging(log_file=log_file, log_level=logging.INFO)
    
    start_time = datetime.now()
    logger.info(f"Starting experiment: {experiment_type}")
    logger.info(f"Dataset ID: {dataset_id}, Feature Set ID: {feature_set_id}")
    logger.info(f"Output directory: {output_dir}")
    if log_file:
        logger.info(f"Log file: {log_file}")
    
    # Step 1: Prepare output directory
    os.makedirs(output_dir, exist_ok=True)
    
    # Step 2: Extract feature matrix
    logger.info(f"Extracting feature matrix...")
    try:
        df_feat = extract_feature_matrix_direct(dataset_id, feature_set_id, db_path)
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
        "run_time": start_time.isoformat(),
        "feature_matrix_shape": df_feat.shape,
        "metadata_shape": df_meta.shape,
        "target_variables": target_vars,
        "log_file": log_file
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
        
        # Step 7: Save to database
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        save_experiment_result(
            db_path=db_path,
            experiment_type=experiment_type,
            dataset_id=dataset_id,
            feature_set_id=feature_set_id,
            output_dir=output_dir,
            extra_args=extra_args,
            summary=summary,
            duration=duration
        )
        
        logger.info(f"Experiment completed! Results saved to: {output_dir}")
        logger.info(f"Total duration: {duration:.2f} seconds")
        
        return {
            "status": "success",
            "summary": summary,
            "output_dir": output_dir,
            "duration": duration
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