-- 实验管理系统数据库模式
-- 支持特征级别的实验结果追踪和查询

-- 删除现有表（如果存在）
DROP TABLE IF EXISTS experiment_feature_results;
DROP TABLE IF EXISTS experiment_metadata;
DROP TABLE IF EXISTS experiment_results;
DROP TABLE IF EXISTS experiment_definitions;

-- 删除现有视图（如果存在）
DROP VIEW IF EXISTS feature_experiment_summary;
DROP VIEW IF EXISTS feature_correlation_history;
DROP VIEW IF EXISTS feature_importance_history;

-- 实验定义表
CREATE TABLE experiment_definitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    type TEXT NOT NULL,  -- 'correlation', 'classification', 'feature_selection', 'feature_statistics'
    description TEXT,
    default_parameters TEXT,  -- JSON格式的默认参数
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- 实验运行记录表
CREATE TABLE experiment_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_type TEXT NOT NULL,
    experiment_name TEXT,
    dataset_id INTEGER,
    feature_set_id INTEGER,
    run_time TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    duration_seconds REAL,
    parameters TEXT,  -- JSON格式的实验参数
    output_dir TEXT,
    status TEXT DEFAULT 'completed',  -- 'running', 'completed', 'failed'
    summary TEXT,
    notes TEXT,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id),
    FOREIGN KEY (feature_set_id) REFERENCES feature_sets(id)
);

-- 实验元数据表（存储实验的详细元信息）
CREATE TABLE experiment_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_result_id INTEGER,
    key TEXT NOT NULL,
    value TEXT,
    value_type TEXT DEFAULT 'string',  -- 'string', 'number', 'boolean', 'json'
    FOREIGN KEY (experiment_result_id) REFERENCES experiment_results(id)
);

-- 特征级别实验结果表（核心表）
CREATE TABLE experiment_feature_results (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_result_id INTEGER,
    fxdef_id INTEGER,
    feature_name TEXT,  -- 实际的特征名称（如 fx20_bp_alpha_O1_mean）
    target_variable TEXT,  -- 目标变量（如 'age', 'sex'）
    result_type TEXT,  -- 'correlation', 'classification_importance', 'selection_score', 'statistic'
    metric_name TEXT,  -- 指标名称（如 'correlation_coefficient', 'p_value', 'accuracy', 'importance_score'）
    metric_value REAL,
    metric_unit TEXT,  -- 单位（如 'correlation', 'percentage', 'score'）
    significance_level TEXT,  -- 显著性水平（如 'p<0.001', 'p<0.01', 'p<0.05', 'ns'）
    rank_position INTEGER,  -- 在结果中的排名位置
    confidence_interval_lower REAL,
    confidence_interval_upper REAL,
    additional_data TEXT,  -- JSON格式的额外数据
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY (experiment_result_id) REFERENCES experiment_results(id),
    FOREIGN KEY (fxdef_id) REFERENCES fxdef(id)
);

-- 创建索引以提高查询性能
CREATE INDEX idx_experiment_results_type ON experiment_results(experiment_type);
CREATE INDEX idx_experiment_results_dataset ON experiment_results(dataset_id);
CREATE INDEX idx_experiment_results_feature_set ON experiment_results(feature_set_id);
CREATE INDEX idx_experiment_results_time ON experiment_results(run_time);

CREATE INDEX idx_experiment_feature_results_experiment ON experiment_feature_results(experiment_result_id);
CREATE INDEX idx_experiment_feature_results_fxdef ON experiment_feature_results(fxdef_id);
CREATE INDEX idx_experiment_feature_results_feature ON experiment_feature_results(feature_name);
CREATE INDEX idx_experiment_feature_results_target ON experiment_feature_results(target_variable);
CREATE INDEX idx_experiment_feature_results_type ON experiment_feature_results(result_type);
CREATE INDEX idx_experiment_feature_results_metric ON experiment_feature_results(metric_name);

-- 插入默认实验定义
INSERT INTO experiment_definitions (name, type, description, default_parameters) VALUES
('correlation', 'correlation', 'Analyze correlation between EEG features and subject metadata', 
 '{"target_vars": ["age", "sex"], "method": "pearson", "min_corr": 0.3, "top_n": 20}'),
('classification', 'classification', 'Perform classification tasks using EEG features', 
 '{"target_var": "age_group", "age_threshold": 65, "test_size": 0.2, "n_splits": 5}'),
('feature_selection', 'feature_selection', 'Select most important features using multiple methods', 
 '{"target_var": "age", "n_features": 20, "variance_threshold": 0.01, "correlation_threshold": 0.95}'),
('feature_statistics', 'feature_statistics', 'Comprehensive statistical analysis of EEG features', 
 '{"outlier_method": "iqr", "outlier_threshold": 1.5, "top_n_features": 20}');

-- 创建视图：特征实验结果汇总
DROP VIEW IF EXISTS feature_experiment_summary;
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

-- 创建视图：特征相关性历史记录
DROP VIEW IF EXISTS feature_correlation_history;
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

-- 创建视图：特征重要性历史记录
DROP VIEW IF EXISTS feature_importance_history;
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