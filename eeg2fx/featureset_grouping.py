import json
import hashlib
from collections import defaultdict, deque
from graphviz import Digraph
import sqlite3
import os
from logging_config import logger

def toposort(graph):
    """拓扑排序"""
    indegree = defaultdict(int)
    for node_id in graph:
        inputnodes = graph[node_id].get("inputnodes", [])
        # 兼容 inputnodes 既可能是列表，也可能是单个值或 None
        if inputnodes is None:
            continue
        elif isinstance(inputnodes, list):
            indegree[node_id] += len(inputnodes)
        else:
            indegree[node_id] += 1
    
    queue = deque([n for n in graph if indegree[n] == 0])
    sorted_nodes = []
    
    while queue:
        node = queue.popleft()
        sorted_nodes.append(node)
        for target in graph:
            inputnodes = graph[target].get("inputnodes", [])
            if inputnodes is None:
                continue
            elif isinstance(inputnodes, list):
                if node in inputnodes:
                    indegree[target] -= 1
                    if indegree[target] == 0:
                        queue.append(target)
            else:
                if node == inputnodes:
                    indegree[target] -= 1
                    if indegree[target] == 0:
                        queue.append(target)
    
    return sorted_nodes

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))

def _hash_id(prefix, fields):
    """Generate short stable hash-based node ID"""
    h = hashlib.sha1(json.dumps(fields, sort_keys=True).encode()).hexdigest()
    return f"{prefix}_{h[:8]}"

def get_node_hash(node_id, node_info):
    """获取单个节点的哈希值"""
    inputnodes = node_info.get("inputnodes")
    input_nodes_list = [inputnodes] if inputnodes else []
    
    node_data = {
        "node_id": node_id,
        "func": node_info["func"],
        "params": node_info["params"],
        "input_nodes": sorted(input_nodes_list)
    }
    return hashlib.md5(json.dumps(node_data, sort_keys=True).encode()).hexdigest()

def load_pipeline_structure(pipeid):
    """Load pipe_nodes + pipe_edges for one pipeline into DAG format with node hashes and parent info"""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    c.execute("""
        SELECT DISTINCT nodeid, func, params
        FROM pipe_nodes
        JOIN pipe_edges ON pipe_nodes.nodeid IN (pipe_edges.from_node, pipe_edges.to_node)
        WHERE pipedef_id = ?
    """, (pipeid,))
    raw_nodes = {row[0]: {"func": row[1], "params": json.loads(row[2] or "{}")} for row in c.fetchall()}

    c.execute("SELECT from_node, to_node FROM pipe_edges WHERE pipedef_id = ?", (pipeid,))
    for from_n, to_n in c.fetchall():
        raw_nodes[to_n]["inputnodes"] = from_n

    conn.close()
    
    # 按拓扑排序顺序计算节点哈希和父节点信息
    execution_order = toposort(raw_nodes)
    
    # 存储每个节点的哈希值
    node_hashes = {}
    
    for nid in execution_order:
        node_info = raw_nodes[nid]
        
        # 计算当前节点的哈希
        current_hash = get_node_hash(nid, node_info)
        
        # 存储节点哈希和父节点信息（单个pipeline中最多只有一个父节点）
        raw_nodes[nid]["node_hash"] = current_hash
        inputnodes = node_info.get("inputnodes")
        raw_nodes[nid]["parent_node"] = inputnodes if inputnodes else None
        raw_nodes[nid]["parent_hash"] = node_hashes.get(inputnodes, "") if inputnodes else ""
        raw_nodes[nid]["upstream_hash"] = hashlib.md5((raw_nodes[nid]["parent_hash"] + current_hash).encode()).hexdigest()

        node_hashes[nid] = raw_nodes[nid]["upstream_hash"]

    logger.debug(f"Loaded pipeline structure for pipeid={pipeid}: {raw_nodes}")
    return raw_nodes

def build_feature_dag(fxdef_list):
    """
    Construct merged pipeline-feature DAG with upstream path information.
    Output: Dict[nodeid] = {func, inputnodes, params, meta, upstream_info}
    """
    dag = {}
    pipeline_nodes = {}  # 记录每个pipeline的节点
    
    # 逐个pipeline处理，同时收集节点和构建上游路径信息
    for fx in fxdef_list:
        pipeid = fx["pipeid"]
        
        # 按需加载pipeline结构（实现去重）
        if pipeid not in pipeline_nodes:
            pipeline_nodes[pipeid] = load_pipeline_structure(pipeid)
        
        pipeline = pipeline_nodes[pipeid]
        
        # 处理pipeline中的节点
        for nid, node in pipeline.items():
            inputnodes = node.get("inputnodes")
            
            new_path = (
                pipeid,
                node["node_hash"],
                node["parent_hash"],
                node["upstream_hash"]
            )
            new_upstream_path = (
                node["node_hash"],
                node["parent_hash"],
                inputnodes,
                node["upstream_hash"]
            )

            if nid not in dag:
                dag[nid] = {
                    "func": node["func"],
                    "inputnodes": [inputnodes] if inputnodes else [],
                    "params": node["params"],
                    "pipeline_paths": {new_path},  # 包含的pipeline路径
                    "fxdef_ids": [fx["id"]],                   
                    "upstream_paths": {new_upstream_path}     # 使用set防止重复
                }
            else:
                dag[nid]["pipeline_paths"].add(new_path)
                dag[nid]["fxdef_ids"].append(fx["id"])
                dag[nid]["upstream_paths"].add(new_upstream_path)

            if nid == "raw":
                print(f"[debug] nid: {nid}, inputnodes: {inputnodes}, new_path: {new_path}, new_upstream_path: {new_upstream_path}")


        inputnodeid = fx["inputnodeid"] # output_node of the pipeline (final output where feature is computed)
        fxfunc = fx["func"]
        fxparams = fx["params"]
        chan = fx["chans"]
        
        # Add feature node
        feature_id = _hash_id(fxfunc, [fxfunc, fxparams])
        parent_hash = pipeline[inputnodeid]["upstream_hash"]  # 使用node_hash而不是parent_hash
        feature_node_upstream_hash = hashlib.md5((parent_hash + feature_id).encode()).hexdigest()
        
        feature_node_path = (
            fx["pipeid"],
            feature_id,
            parent_hash,
            feature_node_upstream_hash
        )
        feature_node_upstream_path = (
            feature_id,
            parent_hash,
            inputnodeid,
            feature_node_upstream_hash
        )

        if feature_id not in dag:
            dag[feature_id] = {
                "func": fxfunc,
                "inputnodes": [inputnodeid],
                "params": fxparams,
                "meta": {},
                "pipeline_paths": {feature_node_path},  # 包含的pipeline路径
                "fxdef_ids": [fx["id"]],               # 直接添加当前fxdef_id
                "upstream_paths": {feature_node_upstream_path}     # 使用set防止重复
            }
        else:
            dag[feature_id]["pipeline_paths"].add(feature_node_path)
            dag[feature_id]["upstream_paths"].add(feature_node_upstream_path)
            dag[feature_id]["fxdef_ids"].append(fx["id"])

        # Add split node
        split_id = f"{feature_id}__{chan}"
        split_upstream_hash = hashlib.md5((feature_id + split_id).encode()).hexdigest()
        
        if split_id not in dag:
            # 构建分割节点的pipeline路径信息
            node_path = (
                fx["pipeid"],
                split_id,
                feature_node_upstream_hash,
                split_upstream_hash
            )
            upstream_path = (
                split_id,
                feature_node_upstream_hash,
                feature_id,
                split_upstream_hash
            )
            
            dag[split_id] = {
                "func": "split_channel",
                "inputnodes": [feature_id],
                "params": {"chan": chan},
                "meta": {"fxdef_id": fx["id"], "chan": chan},
                "pipeline_paths": {node_path},  # 包含的pipeline路径
                "fxdef_ids": [fx["id"]],               # 直接添加当前fxdef_id
                "upstream_paths": {upstream_path}    # 使用set防止重复
            }
        else:
            dag[split_id]["pipeline_paths"].add(node_path)
            dag[split_id]["upstream_paths"].add(upstream_path)
            dag[split_id]["fxdef_ids"].append(fx["id"])

    return dag

def visualize_dag(dag, output_path="dag_view", view=False):
    """
    Render DAG to PDF using graphviz.
    """
    dot = Digraph(comment="Feature DAG", format="pdf")

    for nid, node in dag.items():
        func = node["func"]
        params_str = json.dumps(node["params"], sort_keys=True, separators=(",", ":"))
        
        # 添加节点信息到标签
        pipeline_paths = node.get("pipeline_paths", set())
        fxdef_ids = node.get("fxdef_ids", [])
        upstream_paths = node.get("upstream_paths", set())
        pipeline_count = len(pipeline_paths)
        fxdef_count = len(fxdef_ids)
        upstream_count = len(upstream_paths)
        
        info_str = f"Pipelines: {pipeline_count}, FxDefs: {fxdef_count}, Upstream: {upstream_count}"
        
        label = f"{func}\n{params_str}\n{info_str}"
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
