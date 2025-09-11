#!/usr/bin/env python3
import os
import sys
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, Optional, List

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .common import TaskStatus
from .models import TaskManager
from logging_config import logger

class SimpleLocalExecutor:
    """
    Simple local executor that executes all tasks sequentially in the current thread.
    """

    def __init__(self) -> None:
        """
        Initialize the SimpleLocalExecutor.
        """
        logger.info("SimpleLocalExecutor initialized - no threading, direct execution")

    def execute_feature_extraction(
        self, 
        task_id: int, 
        parameters: Dict[str, Any], 
        dataset_id: int, 
        feature_set_id: int
    ) -> Dict[str, Any]:
        """
        Execute a feature extraction task directly.

        Args:
            task_id (int): Task ID.
            parameters (Dict[str, Any]): Task parameters.
            dataset_id (int): Dataset ID.
            feature_set_id (int): Feature set ID.

        Returns:
            Dict[str, Any]: Result dictionary containing task execution details.
        """
        logger.info(f"=== Start executing feature extraction task {task_id} ===")
        logger.info(f"Task parameters: dataset_id={dataset_id}, feature_set_id={feature_set_id}")

        task_manager = TaskManager()
        task_manager.update_task_status(task_id, TaskStatus.RUNNING)

        recording_ids = self._get_recording_ids_for_dataset(dataset_id)
        if not recording_ids:
            raise ValueError(f"No recordings found for dataset {dataset_id}")

        logger.info(f"Found {len(recording_ids)} recordings to process")
        task_manager.update_task_progress(task_id, 0.0, 0, len(recording_ids))

        results = {
            'dataset_id': dataset_id,
            'feature_set_id': feature_set_id,
            'total_recordings': len(recording_ids),
            'processed_recordings': 0,
            'successful_recordings': 0,
            'failed_recordings': 0,
            'errors': []
        }

        successful_count = 0
        failed_count = 0

        for i, recording_id in enumerate(recording_ids):
            logger.info(f"Processing recording {recording_id} ({i+1}/{len(recording_ids)})")
            try:
                from eeg2fx.featureset_fetcher import run_feature_set
                result = run_feature_set(feature_set_id, recording_id)
                if result:
                    successful_count += 1
                    logger.info(f"Recording {recording_id} processed successfully")
                else:
                    failed_count += 1
                    error_msg = "run_feature_set returned None"
                    results['errors'].append(f"Recording {recording_id}: {error_msg}")
                    logger.warning(f"Recording {recording_id} failed: {error_msg}")
            except Exception as e:
                failed_count += 1
                error_msg = str(e)
                results['errors'].append(f"Recording {recording_id}: {error_msg}")
                logger.error(f"Recording {recording_id} failed with exception: {error_msg}")

            processed_count = i + 1
            progress = (processed_count / len(recording_ids)) * 100
            task_manager.update_task_progress(task_id, progress, processed_count, len(recording_ids))

        results['processed_recordings'] = successful_count + failed_count
        results['successful_recordings'] = successful_count
        results['failed_recordings'] = failed_count

        logger.info(f"Feature extraction completed: {successful_count} successful, {failed_count} failed")
        task_manager.update_task_status(task_id, TaskStatus.COMPLETED, result=results)
        return results

    def execute_experiment(
        self, 
        task_id: int, 
        parameters: Dict[str, Any], 
        dataset_id: int, 
        feature_set_id: int, 
        experiment_type: str
    ) -> Dict[str, Any]:
        """
        Execute an experiment task directly.

        Args:
            task_id (int): Task ID.
            parameters (Dict[str, Any]): Task parameters.
            dataset_id (int): Dataset ID.
            feature_set_id (int): Feature set ID.
            experiment_type (str): Experiment type.

        Returns:
            Dict[str, Any]: Result dictionary containing experiment execution details.
        """
        logger.info(f"Start executing experiment task {task_id}")

        task_manager = TaskManager()
        task_manager.update_task_status(task_id, TaskStatus.RUNNING)

        # Generate unique output directory
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = parameters.get('output_dir', f'experiments/{experiment_type}_{task_id}_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")

        extra_args = parameters.copy()
        for key in ['dataset_id', 'feature_set_id', 'experiment_type', 'output_dir']:
            extra_args.pop(key, None)

        logger.info(f"Calling run_experiment with parameters: experiment_type={experiment_type}, dataset_id={dataset_id}, feature_set_id={feature_set_id}")

        from feature_mill.experiment_engine import run_experiment
        result = run_experiment(
            experiment_type=experiment_type,
            dataset_id=dataset_id,
            feature_set_id=feature_set_id,
            output_dir=output_dir,
            extra_args=extra_args
        )

        logger.info(f"run_experiment completed successfully for task {task_id}")
        logger.info(f"Result: {result}")

        result['output_dir'] = output_dir
        logger.info(f"Updating task {task_id} status to COMPLETED...")
        task_manager.update_task_status(task_id, TaskStatus.COMPLETED, result=result)

        # Update output_dir in database
        logger.info(f"Updating output_dir in database for task {task_id}...")
        conn = sqlite3.connect(task_manager.db_path)
        c = conn.cursor()
        c.execute("UPDATE tasks SET output_dir = ? WHERE id = ?", (output_dir, task_id))
        conn.commit()
        conn.close()

        logger.info(f"Experiment task {task_id} completed successfully, output_dir: {output_dir}")
        return result

    def _get_recording_ids_for_dataset(self, dataset_id: int) -> List[int]:
        """
        Get all recording IDs for a given dataset.

        Args:
            dataset_id (int): Dataset ID.

        Returns:
            List[int]: List of recording IDs.
        """
        db_path = os.getenv('DATABASE_PATH', 'database/eeg2go.db')
        logger.info(f"Getting recording IDs for dataset {dataset_id} from database...")
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("SELECT id FROM recordings WHERE dataset_id = ?", (dataset_id,))
            recording_ids = [row[0] for row in c.fetchall()]
            conn.close()
            logger.info(f"Successfully got {len(recording_ids)} recording IDs for dataset {dataset_id}")
            return recording_ids
        except Exception as e:
            logger.error(f"Failed to get recording IDs for dataset {dataset_id}: {e}")
            return []

    def submit_task(
        self, 
        task_id: int, 
        task_type: str, 
        parameters: Dict[str, Any], 
        dataset_id: Optional[int] = None, 
        feature_set_id: Optional[int] = None,
        experiment_type: Optional[str] = None
    ) -> Any:
        """
        Submit a task to the simple local executor.

        Args:
            task_id (int): Task ID.
            task_type (str): Task type.
            parameters (Dict[str, Any]): Task parameters.
            dataset_id (Optional[int]): Dataset ID.
            feature_set_id (Optional[int]): Feature set ID.
            experiment_type (Optional[str]): Experiment type.

        Returns:
            Any: The result of the executed task.
        """
        logger.info(f"Task {task_id} submitted to simple local executor")
        try:
            if task_type == 'feature_extraction':
                return self.execute_feature_extraction(
                    task_id, parameters, dataset_id, feature_set_id
                )
            elif task_type == 'experiment':
                return self.execute_experiment(
                    task_id, parameters, dataset_id, feature_set_id, experiment_type
                )
            else:
                raise ValueError(f"Unknown task type: {task_type}")
        except Exception as e:
            error_message = str(e)
            logger.error(f"Task {task_id} failed: {error_message}")
            task_manager = TaskManager()
            task_manager.update_task_status(task_id, TaskStatus.FAILED, error_message=error_message)
            raise

_simple_local_executor = None

def get_simple_local_executor() -> SimpleLocalExecutor:
    """
    Get the global SimpleLocalExecutor instance.

    Returns:
        SimpleLocalExecutor: The global executor instance.
    """
    global _simple_local_executor
    if _simple_local_executor is None:
        _simple_local_executor = SimpleLocalExecutor()
    return _simple_local_executor
