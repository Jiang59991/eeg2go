import json
from typing import Any, Dict, List, Callable
from .feature.common import auto_gc
from .function_registry import PREPROCESSING_FUNCS, FEATURE_FUNCS, UTILITY_FUNCS, EPOCH_BY_EVENT_FUNCS
from .featureset_grouping import load_pipeline_structure

def resolve_function(func_name: str, context: Any = None) -> Callable:
    """
    Resolve a function name to its corresponding callable object from the function registry.

    Args:
        func_name (str): The name of the function to resolve.
        context (Any, optional): Optional context for event-based functions.

    Returns:
        Callable: The resolved function.

    Raises:
        ValueError: If the function name is not registered.
    """
    if func_name in PREPROCESSING_FUNCS:
        return PREPROCESSING_FUNCS[func_name]
    if func_name in EPOCH_BY_EVENT_FUNCS:
        return EPOCH_BY_EVENT_FUNCS[func_name](context)
    if func_name in FEATURE_FUNCS:
        return FEATURE_FUNCS[func_name]
    if func_name in UTILITY_FUNCS:
        return UTILITY_FUNCS[func_name]
    raise ValueError(f"Function '{func_name}' is not registered in function_registry.")

def split_channel(result_dict: dict, chan: Any) -> list:
    """
    Extract the data for a specific channel from a result dictionary.

    Args:
        result_dict (dict): The dictionary containing channel data.
        chan (Any): The channel key to extract.

    Returns:
        list: The data for the specified channel, or an empty list if not found.
    """
    if isinstance(result_dict, dict) and chan in result_dict:
        return result_dict[chan]
    return []

@auto_gc
def run_pipeline(
    pipeid: int,
    recording_id: int,
    value_cache: Dict[Any, Any],
    node_output: Dict[Any, Any]
) -> Dict[Any, Any]:
    """
    Execute a pipeline and return all intermediate results.

    Args:
        pipeid (int): Pipeline definition ID.
        recording_id (int): Recording ID.
        value_cache (dict): Shared cache for node outputs to avoid redundant computation.
        node_output (dict): Output dictionary for node results.

    Returns:
        Dict[Any, Any]: All node outputs for this pipeline.
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
        else:
            func = resolve_function(func_name, context=context)
            if func_name == "raw":
                output = func(recording_id, **params)
            else:
                if "chans" in node:
                    output = func(*inputs, chans=node["chans"], **params)
                else:
                    output = func(*inputs, **params)
            value_cache[cache_key] = output

        node_output[nid] = output

    return node_output

def toposort(graph: Dict[Any, Dict[str, Any]]) -> List[Any]:
    """
    Perform topological sort on a directed acyclic graph (DAG).

    Args:
        graph (Dict[Any, Dict[str, Any]]): The DAG to sort.

    Returns:
        List[Any]: List of node IDs in topological order.
    """
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
