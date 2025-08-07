from celery import Celery
import os

# 从环境变量获取Redis配置，如果没有则使用默认值
REDIS_HOST = os.getenv('REDIS_HOST', 'localhost')
REDIS_PORT = os.getenv('REDIS_PORT', '6379')
REDIS_DB = os.getenv('REDIS_DB', '0')

celery_app = Celery(
    "eeg2go",
    broker=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
    backend=f"redis://{REDIS_HOST}:{REDIS_PORT}/{REDIS_DB}",
    include=['task_queue.tasks']  # 包含任务模块
)

# Celery配置
celery_app.conf.update(
    # 序列化配置
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    
    # 时区配置
    timezone="UTC",
    enable_utc=True,
    
    # 任务配置 - 增加超时时间
    task_track_started=True,
    task_time_limit=7200,  # 2小时超时（从1小时增加到2小时）
    task_soft_time_limit=6000,  # 100分钟软超时（从50分钟增加到100分钟）
    
    # Worker配置
    worker_prefetch_multiplier=1,
    worker_max_tasks_per_child=1000,
    
    # 结果配置
    result_expires=7200,  # 结果保存2小时（从1小时增加到2小时）
    
    # 路由配置
    task_routes={
        'task_queue.tasks.feature_extraction_task': {'queue': 'feature_extraction'},
        'task_queue.tasks.experiment_task': {'queue': 'experiments'},
        'task_queue.tasks.run_feature_set_task': {'queue': 'recordings'},
        'task_queue.tasks.*': {'queue': 'default'},
    },
    
    # 队列配置
    task_default_queue='default',
    task_default_exchange='default',
    task_default_routing_key='default',
)

# 自动发现任务
celery_app.autodiscover_tasks(['task_queue'])