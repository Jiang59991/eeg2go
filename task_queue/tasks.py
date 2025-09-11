import os
import sys
import sqlite3
import json
from datetime import datetime
from typing import Dict, Any, Optional, List
from celery import current_task
from celery.utils.log import get_task_logger
import traceback

# Add project root directory to Python path
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if project_root not in sys.path:
    sys.path.insert(0, project_root)

from .celery_app import celery_app
from .common import TaskStatus
from .models import TaskManager
from eeg2fx.featureset_fetcher import run_feature_set
from feature_mill.experiment_engine import run_experiment

from logging_config import logger

def get_task_manager() -> TaskManager:
    """
    Get a TaskManager instance using the database path from environment variable.

    Returns:
        TaskManager: Instance of TaskManager.
    """
    db_path = os.getenv('DATABASE_PATH', 'database/eeg2go.db')
    return TaskManager(db_path)

@celery_app.task(bind=True, name='task_queue.tasks.feature_extraction_task')
def feature_extraction_task(
    self, 
    task_id: int, 
    parameters: Dict[str, Any], 
    dataset_id: int, 
    feature_set_id: int
) -> Dict[str, Any]:
    """
    Celery task for feature extraction. Schedules parallel feature extraction tasks for all recordings in a dataset.

    Args:
        self: Celery task instance.
        task_id (int): Task ID in the database.
        parameters (Dict[str, Any]): Task parameters.
        dataset_id (int): Dataset ID.
        feature_set_id (int): Feature set ID.

    Returns:
        Dict[str, Any]: Information about the scheduled task group.
    """
    logger.info(f"=== Start executing feature extraction task {task_id} ===")
    logger.info(f"Task parameters: dataset_id={dataset_id}, feature_set_id={feature_set_id}")
    logger.info(f"Parameter details: {parameters}")
    try:
        logger.info(f"Updating task status to running...")
        task_manager = get_task_manager()
        task_manager.update_task_status(task_id, TaskStatus.RUNNING)
        logger.info(f"Getting all recording IDs in dataset {dataset_id}...")
        recording_ids = _get_recording_ids_for_dataset(dataset_id)
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
        logger.info(f"Scheduling {len(recording_ids)} recording tasks for parallel processing...")
        from celery import group
        job = group([
            run_feature_set_task.s(feature_set_id, recording_id)
            for recording_id in recording_ids
        ])
        logger.info(f"All {len(recording_ids)} recording tasks scheduled, waiting for completion...")
        result_group = job.apply_async(queue='recordings')
        logger.info(f"Task group created with ID: {result_group.id}")
        task_manager.update_task_status(task_id, TaskStatus.RUNNING, result={
            'task_group_id': result_group.id,
            'total_recordings': len(recording_ids),
            'status': 'scheduled'
        })
        logger.info(f"=== Feature extraction task {task_id} scheduled successfully ===")
        return {
            'task_group_id': result_group.id,
            'total_recordings': len(recording_ids),
            'status': 'scheduled'
        }
    except Exception as e:
        error_message = str(e)
        logger.error(f"Exception occurred during feature extraction task {task_id}: {error_message}")
        task_manager = get_task_manager()
        task_manager.update_task_status(task_id, TaskStatus.FAILED, error_message=error_message)
        logger.error(f"=== Feature extraction task {task_id} failed ===")
        raise

@celery_app.task(bind=True, name='task_queue.tasks.experiment_task')
def experiment_task(
    self, 
    task_id: int, 
    parameters: Dict[str, Any], 
    dataset_id: int, 
    feature_set_id: int, 
    experiment_type: str
) -> Dict[str, Any]:
    """
    Celery task for running an experiment.

    Args:
        self: Celery task instance.
        task_id (int): Task ID in the database.
        parameters (Dict[str, Any]): Task parameters.
        dataset_id (int): Dataset ID.
        feature_set_id (int): Feature set ID.
        experiment_type (str): Type of experiment.

    Returns:
        Dict[str, Any]: Result of the experiment, including output directory.
    """
    logger.info(f"Start executing experiment task {task_id}")
    try:
        logger.info(f"Updating task {task_id} status to RUNNING...")
        task_manager = get_task_manager()
        task_manager.update_task_status(task_id, TaskStatus.RUNNING)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = parameters.get('output_dir', f'experiments/{experiment_type}_{task_id}_{timestamp}')
        os.makedirs(output_dir, exist_ok=True)
        logger.info(f"Created output directory: {output_dir}")
        extra_args = parameters.get('parameters', {}).copy()
        for key in ['dataset_id', 'feature_set_id', 'experiment_type', 'output_dir']:
            extra_args.pop(key, None)
        logger.info(f"Calling run_experiment with parameters: experiment_type={experiment_type}, dataset_id={dataset_id}, feature_set_id={feature_set_id}")
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
        logger.info(f"Updating output_dir in database for task {task_id}...")
        conn = sqlite3.connect(task_manager.db_path)
        c = conn.cursor()
        c.execute("UPDATE tasks SET output_dir = ? WHERE id = ?", (output_dir, task_id))
        conn.commit()
        conn.close()
        logger.info(f"Experiment task {task_id} completed successfully, output_dir: {output_dir}")
        return result
    except Exception as e:
        error_message = str(e)
        logger.error(f"Exception in experiment task {task_id}: {error_message}")
        logger.error(f"Exception type: {type(e).__name__}")
        logger.error(f"Exception traceback: {traceback.format_exc()}")
        try:
            task_manager = get_task_manager()
            task_manager.update_task_status(task_id, TaskStatus.FAILED, error_message=error_message)
            logger.info(f"Task {task_id} status updated to FAILED")
        except Exception as update_error:
            logger.error(f"Failed to update task status to FAILED: {update_error}")
        logger.error(f"Experiment task {task_id} failed: {error_message}")
        raise

def _get_recording_ids_for_dataset(dataset_id: int) -> List[int]:
    """
    Get all recording IDs in the dataset.

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

@celery_app.task(bind=True, name='task_queue.tasks.run_feature_set_task')
def run_feature_set_task(
    self, 
    feature_set_id: int, 
    recording_id: int
) -> Dict[str, Any]:
    """
    Celery task wrapper for run_feature_set function.

    Args:
        self: Celery task instance.
        feature_set_id (int): Feature set ID.
        recording_id (int): Recording ID.

    Returns:
        Dict[str, Any]: Result of the feature extraction for a single recording.
    """
    logger.info(f"=== Start run_feature_set task: feature_set_id={feature_set_id}, recording_id={recording_id} ===")
    try:
        # Mark as running in Celery worker
        import os
        os.environ['CELERY_WORKER_RUNNING'] = '1'
        from eeg2fx.featureset_fetcher import run_feature_set
        result = run_feature_set(feature_set_id, recording_id)
        logger.info(f"=== run_feature_set task completed: feature_set_id={feature_set_id}, recording_id={recording_id} ===")
        return {
            'feature_set_id': feature_set_id,
            'recording_id': recording_id,
            'success': True,
            'result': result
        }
    except Exception as e:
        error_message = str(e)
        logger.error(f"run_feature_set task failed: feature_set_id={feature_set_id}, recording_id={recording_id}, error={error_message}")
        return {
            'feature_set_id': feature_set_id,
            'recording_id': recording_id,
            'success': False,
            'error': error_message
        }
    finally:
        # Clean up environment variable
        import os
        os.environ.pop('CELERY_WORKER_RUNNING', None)