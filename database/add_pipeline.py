import sqlite3
import json
import hashlib
from logging_config import logger
from typing import List, Dict, Any, Optional, Union

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
            "event_type": {"type": "str", "required": True, "description": "Event type (e.g. 'sleep_stage')"},
            "subepoch_len": {"type": "float", "min": 1.0, "max": 60.0, "default": 10.0, "required": True},
            "drop_partial": {"type": "bool", "default": True, "required": False},
            "min_overlap": {"type": "float", "min": 0.0, "max": 1.0, "default": 0.8, "required": False},
            "include_values": {"type": "list", "required": False}
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

def validate_pipeline(
    pipeline_steps: List[List[Any]],
    step_registry: Dict[str, Any] = STEP_REGISTRY
) -> bool:
    """
    Validate the pipeline steps for correctness, order, and parameter validity.

    Args:
        pipeline_steps: List of steps, each as [step_name, func, inputnames, params].
        step_registry: Registry of available steps and their specifications.

    Returns:
        True if the pipeline is valid, otherwise raises ValueError.
    """
    for step in pipeline_steps:
        if len(step) != 4:
            raise ValueError(f"Invalid step format: {step}")

    prev_output_type = "raw"
    for idx, step in enumerate(pipeline_steps):
        step_name, func, inputnames, params = step
        if func not in step_registry:
            raise ValueError(f"Unknown step: {func}")
        expected_input_type = step_registry[func]["input_type"]
        if idx == 0:
            if expected_input_type != "raw":
                raise ValueError(f"Step {step_name} ({func}) must have input_type 'raw' as the first step.")
            if inputnames != ["raw"]:
                raise ValueError(f"First step '{step_name}' inputnames must be ['raw']")
        else:
            main_input = inputnames[0]
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
    func_list = [step[1] for step in pipeline_steps]

    num_epoch = func_list.count("epoch") + func_list.count("epoch_by_event")
    if num_epoch == 0:
        raise ValueError("Pipeline must contain an 'epoch' step.")
    if num_epoch > 1:
        raise ValueError("Pipeline can only contain one 'epoch' step.")

    if "reject_high_amplitude" in func_list and "zscore" in func_list and func_list.index("reject_high_amplitude") > func_list.index("zscore"):
        raise ValueError("'reject_high_amplitude' step is recommended to appear before 'zscore' step in pipeline.")

    for step in pipeline_steps:
        step_name, func, inputnames, params = step
        if func not in step_registry:
            raise ValueError(f"Unknown step: {func}")

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
                elif pinfo["type"] == "int":
                    v = int(v)
                    if "min" in pinfo and v < pinfo["min"]:
                        raise ValueError(f"Param '{pname}' for step '{func}' too small")
                    if "max" in pinfo and v > pinfo["max"]:
                        raise ValueError(f"Param '{pname}' for step '{func}' too large")
                elif pinfo["type"] == "bool":
                    if isinstance(v, str):
                        if v.lower() in ("1", "true", "yes"):
                            v = True
                        elif v.lower() in ("0", "false", "no"):
                            v = False
                        else:
                            raise ValueError(f"Param '{pname}' for step '{func}' expects bool")
                    else:
                        v = bool(v)

        if func == "filter":
            if "lp" in params and "hp" in params:
                lp_val = float(params["lp"])
                hp_val = float(params["hp"])
                if lp_val > 0 and hp_val > 0 and lp_val <= hp_val:
                    raise ValueError(f"Filter step: low-pass frequency ({lp_val}) must be greater than high-pass frequency ({hp_val})")

    return True

def make_nodeid(func: str, params: Dict[str, Any]) -> str:
    """
    Generate a unique node ID for a pipeline step.

    Args:
        func: Name of the function/step.
        params: Parameters for the step.

    Returns:
        A unique node ID string.
    """
    key = f"{func}:{json.dumps(params, sort_keys=True)}"
    return f"{func}_{hashlib.md5(key.encode()).hexdigest()[:8]}"

def add_pipeline(pipeline_def: Dict[str, Any]) -> int:
    """
    Add a new pipeline definition to the database.

    Args:
        pipeline_def: Dictionary containing pipeline definition and steps.

    Returns:
        The ID of the newly inserted pipeline definition.
    """
    required_fields = ["shortname", "chanset"]
    for field in required_fields:
        if field not in pipeline_def or not pipeline_def[field]:
            raise ValueError(f"Missing required field: {field}")

    steps = pipeline_def["steps"]
    if steps and isinstance(steps[0], dict):
        steps = [[s["step_name"], s["func"], s["inputnames"], s["params"]] for s in steps]

    validate_pipeline(steps, STEP_REGISTRY)

    inferred = infer_pipeline_params(steps)
    fs = inferred["fs"]
    hp = inferred["hp"]
    lp = inferred["lp"]
    epoch = inferred["epoch"]

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("SELECT id FROM pipedef WHERE shortname = ?", (pipeline_def["shortname"],))
    existing = c.fetchone()
    if existing:
        conn.close()
        raise ValueError(f"Pipeline with shortname '{pipeline_def['shortname']}' already exists.")

    node_map: Dict[str, str] = {}

    raw_nodeid = "raw"
    c.execute("SELECT 1 FROM pipe_nodes WHERE nodeid = ?", (raw_nodeid,))
    if not c.fetchone():
        c.execute("INSERT INTO pipe_nodes (nodeid, func, params) VALUES (?, ?, ?)",
                  (raw_nodeid, "raw", json.dumps({})))
    node_map["raw"] = raw_nodeid

    for step_name, func, inputnames, params in steps:
        nodeid = make_nodeid(func, params)
        node_map[step_name] = nodeid
        c.execute("SELECT 1 FROM pipe_nodes WHERE nodeid = ?", (nodeid,))
        if not c.fetchone():
            c.execute("INSERT INTO pipe_nodes (nodeid, func, params) VALUES (?, ?, ?)",
                      (nodeid, func, json.dumps(params, sort_keys=True)))

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

def infer_pipeline_params(
    steps: List[Union[List[Any], Dict[str, Any]]]
) -> Dict[str, Optional[float]]:
    """
    Infer key parameters (fs, hp, lp, epoch) from the pipeline steps.

    Args:
        steps: List of steps, each as a list or dict.

    Returns:
        Dictionary with inferred parameters.
    """
    params = {
        "fs": None,
        "hp": None,
        "lp": None,
        "epoch": None,
        "output_type": "raw"
    }
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
        if func == "epoch_by_event" and "subepoch_len" in step_params:
            params["epoch"] = float(step_params["subepoch_len"])
            params["output_type"] = "epochs"
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
