import json
import hashlib
from collections import defaultdict
from graphviz import Digraph
import sqlite3
import os
from logging_config import logger
from typing import Dict, Any, List

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))

def _hash_id(prefix: str, fields: list) -> str:
    """
    Generate a short, stable hash-based node ID.

    Args:
        prefix (str): Prefix for the node ID.
        fields (list): Fields to be hashed.

    Returns:
        str: Short hash-based node ID.
    """
    h = hashlib.sha1(json.dumps(fields, sort_keys=True).encode()).hexdigest()
    return f"{prefix}_{h[:8]}"

def load_pipeline_structure(pipeid: int) -> Dict[Any, Dict[str, Any]]:
    """
    Load the structure of a pipeline (nodes and edges) and return as a DAG dictionary.

    Args:
        pipeid (int): Pipeline definition ID.

    Returns:
        Dict: {nodeid: {"func": ..., "params": ..., "inputnodes": [...]}}
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT DISTINCT nodeid, func, params
        FROM pipe_nodes
        JOIN pipe_edges ON pipe_nodes.nodeid IN (pipe_edges.from_node, pipe_edges.to_node)
        WHERE pipedef_id = ?
    """, (pipeid,))
    raw_nodes = {row[0]: {"func": row[1], "params": json.loads(row[2] or "{}")} for row in c.fetchall()}

    input_map = defaultdict(list)
    c.execute("SELECT from_node, to_node FROM pipe_edges WHERE pipedef_id = ?", (pipeid,))
    for from_n, to_n in c.fetchall():
        input_map[to_n].append(from_n)

    conn.close()
    for nid in raw_nodes:
        raw_nodes[nid]["inputnodes"] = input_map[nid]

    logger.debug(f"Loaded pipeline structure for pipeid={pipeid}: {raw_nodes}")
    return raw_nodes

def build_feature_dag(fxdef_list: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    """
    Construct a merged pipeline-feature DAG from a list of feature definitions.

    Args:
        fxdef_list (List[Dict]): List of feature definition dicts.

    Returns:
        Dict: DAG structure {nodeid: {func, inputnodes, params, meta}}
    """
    dag = {}

    for fx in fxdef_list:
        pipeid = fx["pipeid"]
        inputnodeid = fx["inputnodeid"]
        fxfunc = fx["func"]
        fxparams = fx["params"]
        chan = fx["chans"]
        fxid = fx["id"]

        logger.debug(f"Building DAG for pipeid={pipeid}")
        pipeline = load_pipeline_structure(pipeid)
        for nid, node in pipeline.items():
            if nid in dag:
                existing_inputs = set(dag[nid]["inputnodes"])
                new_inputs = set(node["inputnodes"])
                dag[nid]["inputnodes"] = list(existing_inputs.union(new_inputs))
            else:
                dag[nid] = {
                    "func": node["func"],
                    "inputnodes": node["inputnodes"],
                    "params": node["params"],
                    "meta": {}
                }

        feature_id = _hash_id(fxfunc, [fxfunc, fxparams])
        if feature_id not in dag:
            dag[feature_id] = {
                "func": "feature - " + fxfunc,
                "inputnodes": [inputnodeid],
                "params": fxparams,
                "meta": {}
            }

        split_id = f"{feature_id}__{chan}"
        if split_id not in dag:
            dag[split_id] = {
                "func": "split_channel",
                "inputnodes": [feature_id],
                "params": {"chan": chan},
                "meta": {"fxdef_id": fxid, "chan": chan}
            }

    return dag

def visualize_dag(dag: Dict[str, Dict[str, Any]], output_path: str = "dag_view", view: bool = False) -> None:
    """
    Render a DAG to PDF using graphviz.

    Args:
        dag (Dict): The DAG to visualize.
        output_path (str): Output file path (without extension).
        view (bool): Whether to open the PDF after rendering.

    Returns:
        None
    """
    dot = Digraph(comment="Feature DAG", format="pdf")

    for nid, node in dag.items():
        func = node["func"]
        params_str = json.dumps(node["params"], sort_keys=True, separators=(",", ":"))
        label = f"{func}\n{params_str}"
        dot.node(nid, label)

    edge_set = set()
    for nid, node in dag.items():
        for parent in node["inputnodes"]:
            edge_key = (parent, nid)
            if edge_key not in edge_set:
                dot.edge(parent, nid)
                edge_set.add(edge_key)

    outpath = dot.render(output_path, view=False)
    logger.info(f"DAG rendered to: {outpath} (PDF)")

def load_fxdefs_for_set(feature_set_id: int) -> List[Dict[str, Any]]:
    """
    Load all feature definitions for a given feature set ID.

    Args:
        feature_set_id (int): The feature set ID.

    Returns:
        List[Dict]: List of feature definition dicts, each with id, pipeid, func, chans, params, inputnodeid.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("""
        SELECT fx.id, fx.pipedef_id, fx.func, fx.chans, fx.params, pd.output_node
        FROM feature_set_items AS fs
        JOIN fxdef AS fx ON fs.fxdef_id = fx.id
        JOIN pipedef AS pd ON fx.pipedef_id = pd.id
        WHERE fs.feature_set_id = ?
    """, (feature_set_id,))
    rows = c.fetchall()
    conn.close()

    fxdefs = []
    for row in rows:
        fxdefs.append({
            "id": row[0],
            "pipeid": row[1],
            "func": row[2],
            "chans": row[3],
            "params": json.loads(row[4] or "{}"),
            "inputnodeid": row[5]
        })
    return fxdefs

if __name__ == "__main__":
    feature_set_id = 1
    fxdefs = load_fxdefs_for_set(feature_set_id)
    dag = build_feature_dag(fxdefs)
    visualize_dag(dag, output_path=f"dag_{feature_set_id}", view=False)
