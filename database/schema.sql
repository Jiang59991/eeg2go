DROP TABLE IF EXISTS datasets;
DROP TABLE IF EXISTS subjects;
DROP TABLE IF EXISTS recordings;
DROP TABLE IF EXISTS recording_events;
DROP TABLE IF EXISTS recording_metadata;
DROP TABLE IF EXISTS pipedef;
DROP TABLE IF EXISTS pipe_nodes;
DROP TABLE IF EXISTS pipe_edges;
DROP TABLE IF EXISTS fxdef;
DROP TABLE IF EXISTS feature_sets;
DROP TABLE IF EXISTS feature_set_items;
DROP TABLE IF EXISTS feature_values;
DROP TABLE IF EXISTS experiment_definitions;
DROP TABLE IF EXISTS experiment_results;
DROP TABLE IF EXISTS experiment_metadata;
DROP TABLE IF EXISTS experiment_feature_results;
DROP TABLE IF EXISTS experiments;
DROP TABLE IF EXISTS pipe_steps;
DROP VIEW IF EXISTS feature_experiment_summary;
DROP VIEW IF EXISTS feature_correlation_history;
DROP VIEW IF EXISTS feature_importance_history;
DROP TABLE IF EXISTS tasks;
DROP TABLE IF EXISTS feature_extraction_tasks;

CREATE TABLE datasets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE,
    description TEXT,
    source_type TEXT,
    path TEXT
);

CREATE TABLE subjects (
    subject_id TEXT PRIMARY KEY,
    dataset_id INTEGER,
    sex TEXT,
    age INTEGER,
    race TEXT,
    ethnicity TEXT,
    visit_count INTEGER,
    icd10_count INTEGER,
    medication_count INTEGER,
    notes TEXT,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id)
);

CREATE TABLE recordings (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER,
    subject_id TEXT,
    filename TEXT,
    path TEXT,
    duration REAL,
    channels INTEGER,
    sampling_rate REAL,
    original_reference TEXT,
    recording_type TEXT,         -- "continuous" or "epoched"
    eeg_ground TEXT,
    placement_scheme TEXT,       -- e.g., "10-20"
    manufacturer TEXT,
    powerline_frequency TEXT,    -- "50", "60", "n/a"
    software_filters TEXT,
    has_events BOOLEAN DEFAULT FALSE,
    event_types TEXT,            
    FOREIGN KEY (dataset_id) REFERENCES datasets(id),
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);

CREATE TABLE recording_events (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    recording_id INTEGER,
    event_type TEXT,        -- e.g., 'seizure', 'stimulus', 'sleep_stage'
    onset REAL,             -- in seconds
    duration REAL,          -- optional, in seconds
    value TEXT,             -- Can store specific info, e.g. 'REM', 'Stage2', 'left button'
    FOREIGN KEY (recording_id) REFERENCES recordings(id)
);

CREATE INDEX idx_event_type ON recording_events (event_type);

CREATE TABLE recording_metadata (
    recording_id INTEGER PRIMARY KEY,
    age_days INTEGER,
    sex TEXT,
    start_time TEXT,
    end_time TEXT,
    seizure TEXT,
    spindles TEXT,
    status TEXT,
    normal TEXT,
    abnormal TEXT,
    FOREIGN KEY (recording_id) REFERENCES recordings(id)
);

CREATE TABLE pipedef (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shortname TEXT UNIQUE,
    description TEXT,
    source TEXT,
    chanset TEXT,
    fs REAL,
    hp REAL,
    lp REAL,
    epoch REAL,
    steps TEXT,
    output_node TEXT NOT NULL
);

CREATE TABLE pipe_nodes (
    nodeid TEXT PRIMARY KEY,
    func TEXT NOT NULL,
    params TEXT
);

CREATE TABLE pipe_edges (
    pipedef_id INTEGER,
    from_node TEXT,
    to_node TEXT,
    FOREIGN KEY (pipedef_id) REFERENCES pipedef(id),
    FOREIGN KEY (from_node) REFERENCES pipe_nodes(nodeid),
    FOREIGN KEY (to_node) REFERENCES pipe_nodes(nodeid)
);

CREATE TABLE fxdef (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    shortname TEXT,
    ver TEXT,
    dim TEXT,               
    func TEXT,              
    pipedef_id INTEGER,
    chans TEXT,       
    params TEXT,            
    notes TEXT,
    feature_type TEXT DEFAULT 'single_channel',
    FOREIGN KEY (pipedef_id) REFERENCES pipedef(id)
);

CREATE TABLE feature_sets (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT,
    description TEXT
);

CREATE TABLE feature_set_items (
    feature_set_id TEXT,
    fxdef_id INTEGER,
    FOREIGN KEY (feature_set_id) REFERENCES feature_sets(id),
    FOREIGN KEY (fxdef_id) REFERENCES fxdef(id)
);

CREATE TABLE feature_values (
    fxdef_id INTEGER,
    recording_id INTEGER,
    value TEXT,
    dim TEXT,
    shape TEXT,
    notes TEXT,
    PRIMARY KEY (fxdef_id, recording_id),
    FOREIGN KEY (fxdef_id) REFERENCES fxdef(id),
    FOREIGN KEY (recording_id) REFERENCES recordings(id)
);

-- Experiment definition table
CREATE TABLE experiment_definitions (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    name TEXT UNIQUE NOT NULL,
    type TEXT NOT NULL,  -- 'correlation', 'classification', 'feature_selection', 'feature_statistics'
    description TEXT,
    default_parameters TEXT,  -- Default parameters in JSON format
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

-- Experiment run record table (new version, replaces old experiments table)
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
    status TEXT DEFAULT 'completed',  -- 'running', 'completed', 'failed'
    summary TEXT,
    notes TEXT,
    FOREIGN KEY (dataset_id) REFERENCES datasets(id),
    FOREIGN KEY (feature_set_id) REFERENCES feature_sets(id)
);

-- Experiment metadata table (stores detailed experiment metadata)
CREATE TABLE experiment_metadata (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    experiment_result_id INTEGER,
    key TEXT NOT NULL,
    value TEXT,
    value_type TEXT DEFAULT 'string',  -- 'string', 'number', 'boolean', 'json'
    FOREIGN KEY (experiment_result_id) REFERENCES experiment_results(id)
);

-- Feature-level experiment results table (core table)
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

-- Create indexes to improve query performance
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

-- Insert default experiment definitions
INSERT INTO experiment_definitions (name, type, description, default_parameters) VALUES
('correlation', 'correlation', 'Analyze correlation between EEG features and subject metadata', 
 '{"target_vars": ["age", "sex"], "method": "pearson", "min_corr": 0.3, "top_n": 20}'),
('classification', 'classification', 'Perform classification tasks using EEG features', 
 '{"target_var": "age_group", "age_threshold": 65, "test_size": 0.2, "n_splits": 5}'),
('feature_selection', 'feature_selection', 'Select most important features using multiple methods', 
 '{"target_var": "age", "n_features": 20, "variance_threshold": 0.01, "correlation_threshold": 0.95}'),
('feature_statistics', 'feature_statistics', 'Comprehensive statistical analysis of EEG features', 
 '{"outlier_method": "iqr", "outlier_threshold": 1.5, "top_n_features": 20}');

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

CREATE TABLE tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    task_type TEXT NOT NULL,           -- 'feature_extraction', 'experiment', etc.
    status TEXT DEFAULT 'pending',     -- 'pending', 'running', 'completed', 'failed'
    parameters TEXT,                   -- Store task parameters in JSON format
    result TEXT,                       -- Store results in JSON format
    error_message TEXT,                -- Error message
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    started_at TIMESTAMP,
    completed_at TIMESTAMP,
    priority INTEGER DEFAULT 0,        -- Task priority
    dataset_id INTEGER,                -- Associated dataset ID (instead of recording_id)
    feature_set_id INTEGER,            -- Associated feature set ID (for feature extraction)
    experiment_type TEXT,              -- Experiment type (for experiment tasks)
    progress REAL DEFAULT 0.0,         -- Task progress (0-100)
    processed_count INTEGER DEFAULT 0, -- Number of processed items
    total_count INTEGER DEFAULT 0,     -- Total number of items to process
    notes TEXT,                        -- Additional notes
    execution_mode TEXT DEFAULT 'local', -- 'local' or 'pbs' execution mode
    pbs_job_id TEXT,                   -- PBS job ID for PBS-executed tasks
    queue_name TEXT,                   -- PBS queue name used for the task
    FOREIGN KEY (dataset_id) REFERENCES datasets(id),
    FOREIGN KEY (feature_set_id) REFERENCES feature_sets(id)
);

CREATE INDEX IF NOT EXISTS idx_tasks_type ON tasks(task_type);
CREATE INDEX IF NOT EXISTS idx_tasks_status ON tasks(status);
CREATE INDEX IF NOT EXISTS idx_tasks_dataset ON tasks(dataset_id);
CREATE INDEX IF NOT EXISTS idx_tasks_feature_set ON tasks(feature_set_id);
CREATE INDEX IF NOT EXISTS idx_tasks_created ON tasks(created_at);


CREATE TABLE feature_extraction_tasks (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    dataset_id INTEGER,
    feature_set_id INTEGER,
    task_name TEXT,
    status TEXT DEFAULT 'pending',
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