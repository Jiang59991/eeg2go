import json
from eeg2fx.feature.common import auto_gc
from eeg2fx.function_registry import PREPROCESSING_FUNCS, FEATURE_FUNCS, UTILITY_FUNCS, EPOCH_BY_EVENT_FUNCS
from eeg2fx.featureset_grouping import load_pipeline_structure


def resolve_function(func_name, context=None):
    if func_name in PREPROCESSING_FUNCS:
        return PREPROCESSING_FUNCS[func_name]
    if func_name in EPOCH_BY_EVENT_FUNCS:
        return EPOCH_BY_EVENT_FUNCS[func_name](context)
    if func_name in FEATURE_FUNCS:
        return FEATURE_FUNCS[func_name]
    if func_name in UTILITY_FUNCS:
        return UTILITY_FUNCS[func_name]
    raise ValueError(f"Function '{func_name}' is not registered in function_registry.")

def split_channel(result_dict, chan):
    if isinstance(result_dict, dict) and chan in result_dict:
        return result_dict[chan]
    return []

@auto_gc
def run_pipeline(pipeid, recording_id, value_cache, node_output):
    """
    Execute one pipeline, use shared value_cache to avoid redoing nodes.
    Return all intermediate results for this pipeline.
    """
    dag = load_pipeline_structure(pipeid)
    execution_order = toposort(dag)
    context = {"recording_id": recording_id}

    for nid in execution_order:
        node = dag[nid]
        func_name = node["func"]
        params = node["params"]
        input_ids = node["inputnodes"]
        inputs = [node_output[i] for i in input_ids]

        cache_key = (
            func_name,
            json.dumps(params, sort_keys=True),
            tuple(input_ids)
        )

        if cache_key in value_cache:
            output = value_cache[cache_key]
            # print(f"[CACHE HIT] func={func_name} | key={cache_key}")
        else:
            func = resolve_function(func_name, context=context)
            # print(f"[EXECUTE] func={func_name} | input_nodes={input_ids} | params={params}")

            if func_name == "raw":
                output = func(recording_id, **params)
            else:
                if "chans" in node:
                    output = func(*inputs, chans=node["chans"], **params)
                else:
                    output = func(*inputs, **params)
            value_cache[cache_key] = output
            # print(f"[RESULT] node={nid} → output_type={type(output)} | shape={getattr(output, 'shape', 'N/A') or getattr(output, 'get_data', lambda: 'no get_data')()}")

        node_output[nid] = output

    return node_output


# def run_pipeline(pipeid, recording_id, until_node=None, dag_loader=None):
#     if dag_loader is None:
#         raise ValueError("dag_loader function must be provided.")

#     node_map = dag_loader(pipeid)
#     execution_order = toposort(node_map)

#     if not execution_order:
#         raise ValueError(f"No nodes found for pipeline {pipeid}")
#     if until_node is None:
#         until_node = execution_order[-1]

#     cache = {}
#     for nid in execution_order:
#         func_name = node_map[nid]["func"]
#         input_ids = node_map[nid]["inputnodes"]
#         params = node_map[nid]["params"]

#         inputs = [cache[inid] for inid in input_ids]
#         func = resolve_function(func_name)

#         if func_name == "raw":
#             result = func(recording_id)
#         elif func_name == "split_channel":
#             result = func(*inputs, **params)
#         else:
#             result = func(*inputs, **params)

#         cache[nid] = result
#         if nid == until_node:
#             return result

#     raise ValueError(f"Target node '{until_node}' not found.")

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
