import json
import hashlib
from collections import defaultdict
from graphviz import Digraph
import sqlite3
import os
from logging_config import logger

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))

def _hash_id(prefix, fields):
    """Generate short stable hash-based node ID"""
    h = hashlib.sha1(json.dumps(fields, sort_keys=True).encode()).hexdigest()
    return f"{prefix}_{h[:8]}"

def load_pipeline_structure(pipeid):
    """Load pipe_nodes + pipe_edges for one pipeline into DAG format"""
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

def build_feature_dag(fxdef_list):
    """
    Construct merged pipeline-feature DAG.
    Output: Dict[nodeid] = {func, inputnodes, params, meta}
    """
    dag = {}

    for fx in fxdef_list:
        pipeid = fx["pipeid"]
        inputnodeid = fx["inputnodeid"] # output_node of the pipeline (final output where feature is computed)
        fxfunc = fx["func"]
        fxparams = fx["params"]
        chan = fx["chans"]
        fxid = fx["id"]

        # Add pipeline nodes
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

        # Add feature node
        feature_id = _hash_id(fxfunc, [fxfunc, fxparams])
        if feature_id not in dag:
            dag[feature_id] = {
                "func": "feature - " + fxfunc,
                "inputnodes": [inputnodeid],
                "params": fxparams,
                "meta": {}
            }

        # Add split node
        split_id = f"{feature_id}__{chan}"
        if split_id not in dag:
            dag[split_id] = {
                "func": "split_channel",
                "inputnodes": [feature_id],
                "params": {"chan": chan},
                "meta": {"fxdef_id": fxid, "chan": chan}
            }

    return dag

def visualize_dag(dag, output_path="dag_view", view=False):
    """
    Render DAG to PDF using graphviz.
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

def load_fxdefs_for_set(feature_set_id):
    """
    From feature_set_id, load full fxdef records with:
    id, pipeid, inputnodeid, func, chans, params
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
            "inputnodeid": row[5]  # = pipedef.output_node
        })
    return fxdefs

if __name__ == "__main__":
    feature_set_id = 1
    fxdefs = load_fxdefs_for_set(feature_set_id)
    dag = build_feature_dag(fxdefs)
    visualize_dag(dag, output_path=f"dag_{feature_set_id}", view=False)
