-- 重建experiment相关表格和tasks表格的脚本
-- 注意：此脚本会删除现有数据，请谨慎使用

-- 删除相关的视图
DROP VIEW IF EXISTS feature_experiment_summary;
DROP VIEW IF EXISTS feature_correlation_history;
DROP VIEW IF EXISTS feature_importance_history;

-- 删除experiment相关表格
DROP TABLE IF EXISTS experiment_feature_results;
DROP TABLE IF EXISTS experiment_metadata;
DROP TABLE IF EXISTS experiment_results;
DROP TABLE IF EXISTS experiment_definitions;

-- 删除tasks表格
DROP TABLE IF EXISTS tasks;

-- 重新创建experiment_definitions表格
CREATE TABLE experiment_definitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    type TEXT NOT NULL,  -- 'correlation', 'classification', 'feature_selection', 'feature_statistics'
    description TEXT,
    default_parameters TEXT,  -- Default parameters in JSON format
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 重新创建experiment_results表格
CREATE TABLE experiment_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_type TEXT NOT NULL,
    experiment_name TEXT,
    dataset_id INTEGER,
    feature_set_id INTEGER,
    run_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    duration_seconds REAL,
    parameters TEXT,  -- Experiment parameters in JSON format
    output_dir TEXT,
    status TEXT DEFAULT 'completed',  -- 'completed', 'failed'
    summary TEXT,
    notes TEXT,
    task_id INTEGER,  -- 关联到tasks表的ID
    FOREIGN KEY (dataset_id) REFERENCES datasets(id),
    FOREIGN KEY (feature_set_id) REFERENCES feature_sets(id),
    FOREIGN KEY (task_id) REFERENCES tasks(id)
);

-- 重新创建experiment_metadata表格
CREATE TABLE experiment_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_result_id INTEGER,
    key TEXT NOT NULL,
    value TEXT,
    value_type TEXT DEFAULT 'string',  -- 'string', 'number', 'boolean', 'json'
    FOREIGN KEY (experiment_result_id) REFERENCES experiment_results(id)
);

-- 重新创建experiment_feature_results表格
CREATE TABLE experiment_feature_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_result_id INTEGER,
    fxdef_id INTEGER,
    feature_name TEXT,  -- Actual feature name (e.g. fx20_bp_alpha_O1_mean)
    target_variable TEXT,  -- Target variable (e.g. 'age', 'sex')
    result_type TEXT,  -- 'correlation', 'classification_importance', 'selection_score', 'statistic'
    metric_name TEXT,  -- Metric name (e.g. 'correlation_coefficient', 'p_value', 'accuracy', 'importance_score')
    metric_value REAL,
    metric_unit TEXT,  -- Unit (e.g. 'correlation', 'percentage', 'score')
    significance_level TEXT,  -- Significance level (e.g. 'p<0.001', 'p<0.01', 'p<0.05', 'ns')
    rank_position INTEGER,  -- Rank position in results
    confidence_interval_lower REAL,
    confidence_interval_upper REAL,
    additional_data TEXT,  -- Additional data in JSON format
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_result_id) REFERENCES experiment_results(id),
    FOREIGN KEY (fxdef_id) REFERENCES fxdef(id)
);

-- 重新创建tasks表格
CREATE TABLE tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type TEXT NOT NULL,           -- 'feature_extraction', 'experiment', 'pipeline', etc.
    status TEXT DEFAULT 'pending',     -- 'pending', 'running', 'completed', 'failed'
    parameters TEXT,                   -- Store task parameters in JSON format
    result TEXT,                       -- Store results in JSON format (for completed tasks)
    error_message TEXT,                -- Error message
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    priority INTEGER DEFAULT 0,        -- Task priority
    dataset_id INTEGER,                -- Associated dataset ID
    feature_set_id INTEGER,            -- Associated feature set ID
    experiment_type TEXT,              -- Experiment type (for experiment tasks)
    progress REAL DEFAULT 0.0,         -- Task progress (0-100)
    processed_count INTEGER DEFAULT 0, -- Number of processed items
    total_count INTEGER DEFAULT 0,     -- Total number of items to process
    notes TEXT,                        -- Additional notes
    output_dir TEXT,                   -- Output directory for results
    duration_seconds REAL,             -- Task duration in seconds
    FOREIGN KEY (dataset_id) REFERENCES datasets(id),
    FOREIGN KEY (feature_set_id) REFERENCES feature_sets(id)
);

-- 创建索引以提高查询性能
CREATE INDEX idx_experiment_results_type ON experiment_results(experiment_type);
CREATE INDEX idx_experiment_results_dataset ON experiment_results(dataset_id);
CREATE INDEX idx_experiment_results_feature_set ON experiment_results(feature_set_id);
CREATE INDEX idx_experiment_results_time ON experiment_results(run_time);
CREATE INDEX idx_experiment_results_task ON experiment_results(task_id);

CREATE INDEX idx_experiment_feature_results_experiment ON experiment_feature_results(experiment_result_id);
CREATE INDEX idx_experiment_feature_results_fxdef ON experiment_feature_results(fxdef_id);
CREATE INDEX idx_experiment_feature_results_feature ON experiment_feature_results(feature_name);
CREATE INDEX idx_experiment_feature_results_target ON experiment_feature_results(target_variable);
CREATE INDEX idx_experiment_feature_results_type ON experiment_feature_results(result_type);
CREATE INDEX idx_experiment_feature_results_metric ON experiment_feature_results(metric_name);

-- Tasks表索引
CREATE INDEX idx_tasks_type ON tasks(task_type);
CREATE INDEX idx_tasks_status ON tasks(status);
CREATE INDEX idx_tasks_dataset ON tasks(dataset_id);
CREATE INDEX idx_tasks_feature_set ON tasks(feature_set_id);
CREATE INDEX idx_tasks_created ON tasks(created_at);
CREATE INDEX idx_tasks_experiment_type ON tasks(experiment_type);

-- 插入默认的experiment定义
INSERT INTO experiment_definitions (name, type, description, default_parameters) VALUES
('correlation', 'correlation', 'Analyze correlation between EEG features and subject metadata', 
 '{"target_vars": ["age", "sex"], "method": "pearson", "min_corr": 0.3, "top_n": 20}'),
('classification', 'classification', 'Perform classification tasks using EEG features', 
 '{"target_var": "age_group", "age_threshold": 65, "test_size": 0.2, "n_splits": 5}'),
('feature_selection', 'feature_selection', 'Select most important features using multiple methods', 
 '{"target_var": "age", "n_features": 20, "variance_threshold": 0.01, "correlation_threshold": 0.95}'),
('feature_statistics', 'feature_statistics', 'Comprehensive statistical analysis of EEG features', 
 '{"outlier_method": "iqr", "outlier_threshold": 1.5, "top_n_features": 20}');

-- 重新创建视图
-- Create view: feature experiment result summary
CREATE VIEW feature_experiment_summary AS
SELECT 
    efr.fxdef_id,
    efr.feature_name,
    efr.target_variable,
    efr.result_type,
    efr.metric_name,
    efr.metric_value,
    efr.significance_level,
    efr.rank_position,
    er.experiment_type,
    er.dataset_id,
    er.feature_set_id,
    er.run_time,
    er.parameters,
    fd.shortname as feature_shortname,
    fd.chans as feature_channels
FROM experiment_feature_results efr
JOIN experiment_results er ON efr.experiment_result_id = er.id
LEFT JOIN fxdef fd ON efr.fxdef_id = fd.id;

-- Create view: feature correlation history
CREATE VIEW feature_correlation_history AS
SELECT 
    efr.fxdef_id,
    efr.feature_name,
    efr.target_variable,
    efr.metric_value as correlation_coefficient,
    efr.significance_level,
    efr.rank_position,
    er.dataset_id,
    er.feature_set_id,
    er.run_time,
    er.parameters
FROM experiment_feature_results efr
JOIN experiment_results er ON efr.experiment_result_id = er.id
WHERE efr.result_type = 'correlation' AND efr.metric_name = 'correlation_coefficient'
ORDER BY er.run_time DESC;

-- Create view: feature importance history
CREATE VIEW feature_importance_history AS
SELECT 
    efr.fxdef_id,
    efr.feature_name,
    efr.target_variable,
    efr.metric_value as importance_score,
    efr.rank_position,
    er.dataset_id,
    er.feature_set_id,
    er.run_time,
    er.parameters
FROM experiment_feature_results efr
JOIN experiment_results er ON efr.experiment_result_id = er.id
WHERE efr.result_type IN ('classification_importance', 'selection_score')
ORDER BY er.run_time DESC;

-- 完成提示
SELECT 'Experiment tables and tasks table have been successfully rebuilt!' as status;
