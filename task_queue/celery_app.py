from celery import Celery
import os


REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = os.getenv('REDIS_PORT', '6379')
REDIS_DB = os.getenv('REDIS_DB', '0')

celery_app = Celery(
    "eeg2go",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
    include=['task_queue.tasks']
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    
    timezone="UTC",
    enable_utc=True,

    task_track_started=True,
    task_time_limit=7200,
    task_soft_time_limit=6000,
    
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    
    result_expires=7200,
    
    task_routes={
        'task_queue.tasks.feature_extraction_task': {'queue': 'feature_extraction'},
        'task_queue.tasks.experiment_task': {'queue': 'experiments'},
        'task_queue.tasks.run_feature_set_task': {'queue': 'recordings'},
        'task_queue.tasks.*': {'queue': 'default'},
    },
    
    task_default_queue='default',
    task_default_exchange='default',
    task_default_routing_key='default',
)

celery_app.autodiscover_tasks(['task_queue'])