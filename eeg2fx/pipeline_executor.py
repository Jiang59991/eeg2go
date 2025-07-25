import os
import json
import importlib
import hashlib
from eeg2fx.steps import load_recording
from eeg2fx.feature.common import standardize_channel_name
from eeg2fx.function_registry import PREPROCESSING_FUNCS, FEATURE_FUNCS, UTILITY_FUNCS
from logging_config import logger

def resolve_function(func_name):
    if func_name in PREPROCESSING_FUNCS:
        return PREPROCESSING_FUNCS[func_name]
    if func_name in FEATURE_FUNCS:
        return FEATURE_FUNCS[func_name]
    if func_name in UTILITY_FUNCS:
        return UTILITY_FUNCS[func_name]
    raise ValueError(f"Function '{func_name}' is not registered in function_registry.")

def generate_cache_key_with_upstream_info(func_name, params, node_id, upstream_info):
    """
    基于上游路径信息生成缓存键
    """
    cache_data = {
        "func": func_name,
        "params": params,
        "node_id": node_id,
        "upstream_paths": upstream_info.get("paths", [])
    }
    
    return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()

def generate_cache_key_with_node_info(func_name, params, node_id, node_hash, parent_hash):
    """
    基于节点哈希和父节点哈希生成缓存键
    """
    cache_data = {
        "func": func_name,
        "params": params,
        "node_id": node_id,
        "node_hash": node_hash,
        "parent_hash": parent_hash
    }
    
    return hashlib.md5(json.dumps(cache_data, sort_keys=True).encode()).hexdigest()

def execute_dag_nodes(dag, recording_id, value_cache=None):
    """
    按DAG顺序执行所有节点
    """
    execution_order = toposort(dag)
    node_outputs = {}
    value_cache = value_cache or {}
    
    for node_id in execution_order:
        node = dag[node_id]
        func_name = node["func"]
        params = node["params"]
        input_ids = node["inputnodes"]
        upstream_info = node.get("upstream_info", {})
        
        # 获取输入
        inputs = [node_outputs[inid] for inid in input_ids]
        
        # 生成缓存键（优先使用新的方式）
        pipeline_paths = node.get("pipeline_paths", [])
        if pipeline_paths:
            # 使用新的节点哈希和父节点哈希方式
            path_info = pipeline_paths[0]  # 取第一个路径
            node_hash = path_info.get("node_hash", "")
            parent_hash = path_info.get("parent_hash", "")
            
            cache_key = generate_cache_key_with_node_info(
                func_name, params, node_id, node_hash, parent_hash
            )
        else:
            # 回退到旧的方式
            cache_key = generate_cache_key_with_upstream_info(
                func_name, params, node_id, {"paths": []}
            )
        
        # 检查缓存
        if cache_key in value_cache:
            result = value_cache[cache_key]
            logger.info(f"[CACHE HIT] Node {node_id} ({func_name})")
        else:
            # 执行函数
            func = resolve_function(func_name)
            
            if func_name == "raw":
                result = func(recording_id)
            elif func_name == "split_channel":
                result = func(*inputs, **params)
            else:
                result = func(*inputs, **params)
            
            # 缓存结果
            value_cache[cache_key] = result
            logger.info(f"[EXECUTED] Node {node_id} ({func_name})")
        
        node_outputs[node_id] = result
    
    return node_outputs, value_cache

def split_channel(result_dict, chan):
    if isinstance(result_dict, dict) and chan in result_dict:
        return result_dict[chan]
    return []

def run_pipeline(pipeid, recording_id, until_node=None, dag_loader=None):
    if dag_loader is None:
        raise ValueError("dag_loader function must be provided.")

    node_map = dag_loader(pipeid)
    execution_order = toposort(node_map)

    if not execution_order:
        raise ValueError(f"No nodes found for pipeline {pipeid}")
    if until_node is None:
        until_node = execution_order[-1]

    cache = {}
    for nid in execution_order:
        func_name = node_map[nid]["func"]
        input_ids = node_map[nid]["inputnodes"]
        params = node_map[nid]["params"]

        inputs = [cache[inid] for inid in input_ids]
        func = resolve_function(func_name)

        if func_name == "raw":
            result = func(recording_id)
        elif func_name == "split_channel":
            result = func(*inputs, **params)
        else:
            result = func(*inputs, **params)

        cache[nid] = result
        if nid == until_node:
            return result

    raise ValueError(f"Target node '{until_node}' not found.")

def toposort(graph):
    from collections import defaultdict, deque
    indegree = defaultdict(int)
    for node in graph:
        for dep in graph[node]["inputnodes"]:
            indegree[node] += 1

    queue = deque([n for n in graph if indegree[n] == 0])
    sorted_nodes = []

    while queue:
        node = queue.popleft()
        sorted_nodes.append(node)
        for target in graph:
            if node in graph[target]["inputnodes"]:
                indegree[target] -= 1
                if indegree[target] == 0:
                    queue.append(target)

    return sorted_nodes
