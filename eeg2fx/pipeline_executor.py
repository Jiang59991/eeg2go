
from eeg2fx.function_registry import PREPROCESSING_FUNCS, FEATURE_FUNCS, UTILITY_FUNCS

def resolve_function(func_name):
    if func_name in PREPROCESSING_FUNCS:
        return PREPROCESSING_FUNCS[func_name]
    if func_name in FEATURE_FUNCS:
        return FEATURE_FUNCS[func_name]
    if func_name in UTILITY_FUNCS:
        return UTILITY_FUNCS[func_name]
    raise ValueError(f"Function '{func_name}' is not registered in function_registry.")

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
