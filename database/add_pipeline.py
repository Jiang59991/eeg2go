"""
TODO: 
1. 传入参数异常检测（字段缺失/不符合规范）
2. 判重 ： add之前先检测数据库中是否已经存在完全相同配置的pipeline

"""
import sqlite3
import json
import hashlib

DB_PATH = "database/eeg2go.db"

def make_nodeid(func, params):
    key = f"{func}:{json.dumps(params, sort_keys=True)}"
    return f"{func}_{hashlib.md5(key.encode()).hexdigest()[:8]}"

def add_pipeline(pipeline_def):
    conn = sqlite3.connect(DB_PATH)

    """
    将一个 pipeline 插入数据库（pipedef, pipe_nodes, pipe_edges）
    
    reference pipeline_def:
    {
        "shortname": "basic_clean_5s",
        "description": "Filter → reref → epoch",
        "source": "default",
        "chanset": "10/20",
        "fs": 250.0,
        "hp": 1.0,
        "lp": 40.0,
        "epoch": 5.0,
        "steps": [
            ["flt", "filter", ["raw"], {"hp": 1.0, "lp": 40.0}],
            ["reref", "reref", ["flt"], {}],
            ["epoch", "epoch", ["reref"], {"duration": 5.0}]
        ]
    }
    """
    c = conn.cursor()

    # Check if pipeline with same shortname already exists
    c.execute("SELECT id FROM pipedef WHERE shortname = ?", (pipeline_def["shortname"],))
    existing = c.fetchone()
    if existing:
        print(f"Pipeline '{pipeline_def['shortname']}' already exists (id={existing[0]}) - skipping")
        conn.close()
        return existing[0]

    steps = pipeline_def["steps"]
    node_map = {}  # step_name → nodeid

    # create raw node（if not exists）
    raw_nodeid = "raw"
    c.execute("SELECT 1 FROM pipe_nodes WHERE nodeid = ?", (raw_nodeid,))
    if not c.fetchone():
        c.execute("INSERT INTO pipe_nodes (nodeid, func, params) VALUES (?, ?, ?)",
                  (raw_nodeid, "raw", json.dumps({})))

    node_map["raw"] = raw_nodeid

    # insert nodes
    for step_name, func, inputnames, params in steps:
        nodeid = make_nodeid(func, params)
        node_map[step_name] = nodeid

        # check whether node exist
        c.execute("SELECT 1 FROM pipe_nodes WHERE nodeid = ?", (nodeid,))
        if not c.fetchone():
            c.execute("INSERT INTO pipe_nodes (nodeid, func, params) VALUES (?, ?, ?)",
                      (nodeid, func, json.dumps(params, sort_keys=True)))

    # insert pipeline definition（pipedef）
    output_node = node_map[steps[-1][0]]
    c.execute("""
        INSERT INTO pipedef (shortname, description, source, chanset, fs, hp, lp, epoch, steps, output_node)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (
        pipeline_def["shortname"], pipeline_def["description"], pipeline_def["source"],
        pipeline_def["chanset"], pipeline_def["fs"], pipeline_def["hp"],
        pipeline_def["lp"], pipeline_def["epoch"], json.dumps(steps), output_node
    ))
    pipedef_id = c.lastrowid

    # insert edges
    for step_name, func, inputnames, params in steps:
        to_node = node_map[step_name]
        for inputname in inputnames:
            from_node = node_map[inputname]
            c.execute("""
                INSERT INTO pipe_edges (pipedef_id, from_node, to_node)
                VALUES (?, ?, ?)
            """, (pipedef_id, from_node, to_node))

    conn.commit()
    conn.close()
    print(f"Pipeline '{pipeline_def['shortname']}' inserted (id={pipedef_id})")
    return pipedef_id

if __name__ == "__main__":

    pipeline = {
        "shortname": "basic_clean_5s",
        "description": "Filter → reref → epoch",
        "source": "TUH_demo",
        "chanset": "10/20",
        "fs": 250.0,
        "hp": 1.0,
        "lp": 40.0,
        "epoch": 5.0,
        "steps": [
            ["flt", "filter", ["raw"], {"hp": 1.0, "lp": 40.0}],
            ["reref", "reref", ["flt"], {}],
            ["epoch", "epoch", ["reref"], {"duration": 5.0}]
        ]
    }

    add_pipeline(pipeline)
