DROP TABLE IF EXISTS experiment_results;
DROP TABLE IF EXISTS experiments;
DROP TABLE IF EXISTS feature_values;
DROP TABLE IF EXISTS feature_set_items;
DROP TABLE IF EXISTS feature_sets;
DROP TABLE IF EXISTS fxdef;
DROP TABLE IF EXISTS pipe_edges;
DROP TABLE IF EXISTS pipe_nodes;
DROP TABLE IF EXISTS pipedef;
DROP TABLE IF EXISTS recording_metadata;
DROP TABLE IF EXISTS recordings;
DROP TABLE IF EXISTS subjects;
DROP TABLE IF EXISTS datasets;

DROP TABLE IF EXISTS pipe_steps;

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
    FOREIGN KEY (dataset_id) REFERENCES datasets(id),
    FOREIGN KEY (subject_id) REFERENCES subjects(subject_id)
);

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
    source TEXT,  -- 标记用户？ default?
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

CREATE TABLE experiments (
    id TEXT PRIMARY KEY,
    name TEXT,
    feature_set_id TEXT,         -- 外键 → feature_sets.id
    label TEXT,               -- 标签字段名，如 sex / age / diagnosis
    method TEXT,              -- 如 'ttest', 'pearson'
    notes TEXT,
    created_at TEXT,
    FOREIGN KEY (feature_set_id) REFERENCES feature_sets(id)
);

CREATE TABLE experiment_results (
    experiment_id TEXT,
    fxdef_id INTEGER,
    stat TEXT,               -- 'mean_1', 'mean_2', 't', 'p', 'corr'
    value REAL,
    label_value TEXT,        -- 'male', 'female' or NULL（用于分组）
    FOREIGN KEY (experiment_id) REFERENCES experiments(id),
    FOREIGN KEY (fxdef_id) REFERENCES fxdef(id)
);