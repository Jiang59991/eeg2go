"""
TODO: 
1. 传入参数异常检测（字段缺失/不符合规范）
2. 判重 ： add之前先检测数据库中是否已经存在完全相同配置的pipeline

"""
import sqlite3
import json
import hashlib
from logging_config import logger

DB_PATH = "database/eeg2go.db"

STEP_REGISTRY = {
    "filter": {
        "params": {
            "hp": {"type": "float", "min": 0, "max": 100.0, "default": 1.0, "required": True},
            "lp": {"type": "float", "min": 0, "max": 200.0, "default": 40.0, "required": True}
        },
        "input_type": "raw",
        "output_type": "raw"
    },
    "reref": {
        "params": {
            "method": {"type": "str", "required": True, "default": "average", "options": ["average", "linked_mastoid"]}
        },
        "input_type": "raw",
        "output_type": "raw"
    },
    "notch_filter": {
        "params": {
            "freq": {"type": "float", "min": 1.0, "max": 1000.0, "default": 50.0, "required": True}
        },
        "input_type": "raw",
        "output_type": "raw"
    },
    "resample": {
        "params": {
            "sfreq": {"type": "float", "min": 1.0, "max": 1000.0, "default": 128.0, "required": True}
        },
        "input_type": "raw",
        "output_type": "raw"
    },
    # "pick_channels": {
    #     "params": {
    #         "include": {"type": "list", "required": True}
    #     },
    #     "input_type": "raw",
    #     "output_type": "raw"
    # },
    "ica": {
        "params": {
            "n_components": {"type": "int", "min": 1, "max": 100, "default": 20, "required": True},
            "detect_artifacts": {"type": "str", "required": True, "default": "none", "options": ["eog", "ecg", "auto","none"]}
        },
        "input_type": "raw",
        "output_type": "raw"
    },
    "zscore": {
        "params": {
            "mode": {"type": "str", "required": True, "default": "per_epoch", "options": ["per_epoch", "global"]}
        },
        "input_type": "epochs",
        "output_type": "epochs"
    },
    "epoch": {
        "params": {
            "duration": {"type": "float", "min": 0.1, "max": 60.0, "default": 5.0, "required": True}
        },
        "input_type": "raw",
        "output_type": "epochs"
    },
    "epoch_by_event": {
        "params": {
            "event_type": {"type": "str", "required": True, "description": "Event type to use for epoching, e.g., 'seizure', 'spindle', 'normal'"},
            "tmin": {"type": "float", "min": -5.0, "max": 0.0, "default": -0.2, "required": True},
            "tmax": {"type": "float", "min": 0.1, "max": 10.0, "default": 1.0, "required": True}
        },
        "input_type": "raw",
        "output_type": "epochs"
    },
    "reject_high_amplitude": {
        "params": {
            "threshold_uv": {"type": "int", "min": 5, "max": 1000, "default": 150, "required": True}
        },
        "input_type": "epochs",
        "output_type": "epochs"
    }
}

def validate_pipeline(pipeline_steps, step_registry=STEP_REGISTRY):
    """
    pipeline_steps: List[List]，每个内部列表包含 [step_name, func, inputnames, params]
    step_registry: 步骤注册表
    """
    # 1. 检查每步格式
    for step in pipeline_steps:
        if len(step) != 4:
            raise ValueError(f"Invalid step format: {step}")
    
    # 2. 检查步骤顺序和唯一性
    # 2.1 检查步骤顺序：根据注册表的input_type和output_type检查步骤顺序
    # 只检查主链路（即每一步的第一个输入），确保数据类型流转正确
    prev_output_type = "raw"
    for idx, step in enumerate(pipeline_steps):
        step_name, func, inputnames, params = step
        if func not in step_registry:
            raise ValueError(f"Unknown step: {func}")
        expected_input_type = step_registry[func]["input_type"]
        # 主链路：第一个输入
        if idx == 0:
            # 第一节点的输入类型必须为raw，且inputnames应为["raw"]
            if expected_input_type != "raw":
                raise ValueError(f"Step {step_name} ({func}) must have input_type 'raw' as the first step.")
            if inputnames != ["raw"]:
                raise ValueError(f"First step '{step_name}' inputnames must be ['raw']")
        else:
            # 检查主链路输入类型
            # 取inputnames[0]，找到其output_type
            main_input = inputnames[0]
            # 找到main_input对应的上一步
            prev_step = None
            for s in pipeline_steps:
                if s[0] == main_input:
                    prev_step = s
                    break
            if prev_step is None:
                raise ValueError(f"Step '{step_name}' input '{main_input}' not found in previous steps.")
            prev_func = prev_step[1]
            prev_output_type = step_registry[prev_func]["output_type"]
            if expected_input_type != prev_output_type:
                raise ValueError(
                    f"Step '{step_name}' ({func}) input_type '{expected_input_type}' does not match previous output_type '{prev_output_type}' (from '{main_input}')."
                )
    # 2.2 其它顺序和唯一性检查
    func_list = [step[1] for step in pipeline_steps]

    # 检查 epoch 必须存在且唯一
    if func_list.count("epoch") == 0:
        raise ValueError("Pipeline must contain an 'epoch' step.")
    if func_list.count("epoch") > 1:
        raise ValueError("Pipeline can only contain one 'epoch' step.")
    
    if "reject_high_amplitude" in func_list and "zscore" in func_list and func_list.index("reject_high_amplitude") > func_list.index("zscore"):
        raise ValueError("'reject_high_amplitude' step is recommended to appear before 'zscore' step in pipeline.")

    # 3. 检查参数
    for step in pipeline_steps:
        step_name, func, inputnames, params = step
        # 检查func是否在注册表
        if func not in step_registry:
            raise ValueError(f"Unknown step: {func}")
        
        # 检查参数
        for pname, pinfo in step_registry[func]["params"].items():
            if pinfo.get("required") and pname not in params:
                raise ValueError(f"Missing required param '{pname}' for step '{func}'")
            if pname in params:
                v = params[pname]
                if pinfo["type"] == "float":
                    v = float(v)
                    if "min" in pinfo and v < pinfo["min"]:
                        raise ValueError(f"Param '{pname}' for step '{func}' too small")
                    if "max" in pinfo and v > pinfo["max"]:
                        raise ValueError(f"Param '{pname}' for step '{func}' too large")
                    
        # 特殊检查：filter步骤的lp和hp参数大小关系
        if func == "filter":
            if "lp" in params and "hp" in params:
                lp_val = float(params["lp"])
                hp_val = float(params["hp"])
                if lp_val > 0 and hp_val > 0 and lp_val <= hp_val:
                    raise ValueError(f"Filter step: low-pass frequency ({lp_val}) must be greater than high-pass frequency ({hp_val})")
        
    return True

def make_nodeid(func, params):
    key = f"{func}:{json.dumps(params, sort_keys=True)}"
    return f"{func}_{hashlib.md5(key.encode()).hexdigest()[:8]}"

def add_pipeline(pipeline_def):
    # 检查必填字段
    required_fields = ["shortname", "chanset"]
    for field in required_fields:
        if field not in pipeline_def or not pipeline_def[field]:
            raise ValueError(f"Missing required field: {field}")

    # 步骤格式兼容：前端传来的是dict list，转为list list
    steps = pipeline_def["steps"]
    if steps and isinstance(steps[0], dict):
        steps = [[s["step_name"], s["func"], s["inputnames"], s["params"]] for s in steps]

    # 校验 pipeline steps 合法性
    validate_pipeline(steps, STEP_REGISTRY)

    # 自动推断参数
    inferred = infer_pipeline_params(steps)
    fs = inferred["fs"]
    hp = inferred["hp"]
    lp = inferred["lp"]
    epoch = inferred["epoch"]

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Check if pipeline with same shortname already exists
    c.execute("SELECT id FROM pipedef WHERE shortname = ?", (pipeline_def["shortname"],))
    existing = c.fetchone()
    if existing:
        conn.close()
        raise ValueError(f"Pipeline with shortname '{pipeline_def['shortname']}' already exists.")

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
        pipeline_def["shortname"],
        pipeline_def.get("description", None),
        pipeline_def.get("source", None),
        pipeline_def["chanset"],
        fs, hp, lp, epoch,
        json.dumps(steps),
        output_node
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
    logger.info(f"Pipeline '{pipeline_def['shortname']}' inserted (id={pipedef_id}).")
    return pipedef_id

def infer_pipeline_params(steps):
    params = {
        "fs": None,
        "hp": None,
        "lp": None,
        "epoch": None,
        "output_type": "raw"
    }
    # 兼容dict和list格式
    for step in steps:
        if isinstance(step, dict):
            func = step["func"]
            step_params = step["params"]
        else:
            _, func, _, step_params = step
        if func == "resample" and "sfreq" in step_params:
            params["fs"] = float(step_params["sfreq"])
        if func == "filter":
            if "hp" in step_params:
                params["hp"] = float(step_params["hp"])
            if "lp" in step_params:
                params["lp"] = float(step_params["lp"])
        if func == "epoch" and "duration" in step_params:
            params["epoch"] = float(step_params["duration"])
            params["output_type"] = "epochs"
    # 没有resample时fs为None（默认）
    return params

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
        "sample_rating": 8.0,
        "steps": [
            ["flt", "filter", ["raw"], {"hp": 1.0, "lp": 40.0}],
            ["reref", "reref", ["flt"], {"method": "average"}],
            ["epoch", "epoch", ["reref"], {"duration": 5.0}]
        ]
    }

    add_pipeline(pipeline)
