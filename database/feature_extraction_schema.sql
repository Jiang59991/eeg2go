-- 特征提取任务表
CREATE TABLE IF NOT EXISTS feature_extraction_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER NOT NULL,
    feature_set_id INTEGER NOT NULL,
    task_name TEXT,
    status TEXT DEFAULT 'running',  -- 'running', 'completed', 'failed'
    total_recordings INTEGER DEFAULT 0,
    processed_recordings INTEGER DEFAULT 0,
    failed_recordings INTEGER DEFAULT 0,
    start_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    end_time TIMESTAMP,
    duration_seconds REAL,
    result_file TEXT,
    notes TEXT,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id),
    FOREIGN KEY (feature_set_id) REFERENCES feature_sets(id)
);

-- 创建索引
CREATE INDEX IF NOT EXISTS idx_feature_extraction_tasks_dataset ON feature_extraction_tasks(dataset_id);
CREATE INDEX IF NOT EXISTS idx_feature_extraction_tasks_feature_set ON feature_extraction_tasks(feature_set_id);
CREATE INDEX IF NOT EXISTS idx_feature_extraction_tasks_status ON feature_extraction_tasks(status);
CREATE INDEX IF NOT EXISTS idx_feature_extraction_tasks_time ON feature_extraction_tasks(start_time); 