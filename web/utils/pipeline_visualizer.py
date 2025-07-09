"""
Pipeline可视化模块
为前端提供pipeline结构数据，使用Cytoscape.js进行可视化
"""

import json
import hashlib
from collections import defaultdict
import sqlite3
from ..config import DATABASE_PATH

def _hash_id(prefix, fields):
    """生成短而稳定的基于哈希的节点ID"""
    h = hashlib.sha1(json.dumps(fields, sort_keys=True).encode()).hexdigest()
    return f"{prefix}_{h[:8]}"

def load_pipeline_structure(pipeid):
    """加载单个pipeline的pipe_nodes + pipe_edges到DAG格式"""
    conn = sqlite3.connect(DATABASE_PATH)
    c = conn.cursor()

    # 首先获取pipeline的输出节点
    c.execute("SELECT output_node FROM pipedef WHERE id = ?", (pipeid,))
    result = c.fetchone()
    if not result:
        print(f"Pipeline {pipeid} not found in pipedef table")
        conn.close()
        return {}
    
    output_node = result[0]
    print(f"Pipeline {pipeid} output node: {output_node}")
    
    # 获取所有相关的节点（通过pipe_edges表）
    c.execute("""
        SELECT DISTINCT pn.nodeid, pn.func, pn.params
        FROM pipe_nodes pn
        JOIN pipe_edges pe ON pn.nodeid IN (pe.from_node, pe.to_node)
        WHERE pe.pipedef_id = ?
    """, (pipeid,))
    
    raw_nodes = {}
    for row in c.fetchall():
        nodeid, func, params = row
        try:
            params_dict = json.loads(params) if params else {}
        except:
            params_dict = {}
        raw_nodes[nodeid] = {"func": func, "params": params_dict}
        print(f"Found node: {nodeid} -> {func}")

    # 获取边的信息
    input_map = defaultdict(list)
    c.execute("SELECT from_node, to_node FROM pipe_edges WHERE pipedef_id = ?", (pipeid,))
    edges = c.fetchall()
    print(f"Found {len(edges)} edges for pipeline {pipeid}")
    
    for from_n, to_n in edges:
        input_map[to_n].append(from_n)
        print(f"Edge: {from_n} -> {to_n}")

    conn.close()
    
    # 为每个节点添加输入节点信息
    for nid in raw_nodes:
        raw_nodes[nid]["inputnodes"] = input_map[nid]

    print(f"Total nodes in pipeline {pipeid}: {len(raw_nodes)}")
    return raw_nodes

def get_pipeline_cytoscape_data(pipeline_id):
    """
    获取pipeline的Cytoscape.js格式数据
    
    Args:
        pipeline_id: pipeline的ID
    
    Returns:
        dict: 包含nodes和edges的Cytoscape.js格式数据
    """
    try:
        pipeline_nodes = load_pipeline_structure(pipeline_id)
        
        if not pipeline_nodes:
            return None
        
        # 构建Cytoscape.js格式的数据
        nodes = []
        edges = []
        
        # 添加节点
        for node_id, node_data in pipeline_nodes.items():
            func_name = node_data["func"]
            params = node_data["params"]
            
            # 确定节点类型和颜色
            if "input" in func_name.lower():
                node_type = "input"
                color = "#90EE90"  # lightgreen
            elif "output" in func_name.lower():
                node_type = "output"
                color = "#F0A0A0"  # lightcoral
            elif "filter" in func_name.lower():
                node_type = "filter"
                color = "#FFFFE0"  # lightyellow
            else:
                node_type = "process"
                color = "#ADD8E6"  # lightblue
            
            # 创建节点标签
            if params:
                params_str = json.dumps(params, sort_keys=True, separators=(",", ":"))
                if len(params_str) > 50:
                    params_str = params_str[:47] + "..."
                label = f"{func_name}\n{params_str}"
            else:
                label = func_name
            
            nodes.append({
                "data": {
                    "id": node_id,
                    "label": label,
                    "func": func_name,
                    "params": params,
                    "type": node_type
                },
                "classes": node_type
            })
        
        # 添加边
        edge_set = set()
        for node_id, node_data in pipeline_nodes.items():
            for parent_id in node_data["inputnodes"]:
                edge_key = (parent_id, node_id)
                if edge_key not in edge_set:
                    edges.append({
                        "data": {
                            "id": f"e{parent_id}_{node_id}",
                            "source": parent_id,
                            "target": node_id
                        }
                    })
                    edge_set.add(edge_key)
        
        return {
            "nodes": nodes,
            "edges": edges,
            "node_count": len(nodes),
            "edge_count": len(edges)
        }
        
    except Exception as e:
        print(f"获取pipeline Cytoscape数据时出错: {e}")
        return None

def get_pipeline_visualization_data(pipeline_id):
    """
    获取pipeline的可视化数据（兼容旧API）
    
    Args:
        pipeline_id: pipeline的ID
    
    Returns:
        dict: 包含Cytoscape数据和节点信息的字典
    """
    try:
        conn = sqlite3.connect(DATABASE_PATH)
        c = conn.cursor()
        
        # 获取pipeline基本信息
        c.execute("SELECT shortname, description FROM pipedef WHERE id = ?", (pipeline_id,))
        pipeline_info = c.fetchone()
        
        if not pipeline_info:
            return None
        
        pipeline_name = pipeline_info[0] or f"Pipeline {pipeline_id}"
        pipeline_desc = pipeline_info[1] or ""
        
        # 获取pipeline节点
        c.execute("""
            SELECT DISTINCT pn.nodeid, pn.func, pn.params
            FROM pipe_nodes pn
            JOIN pipe_edges pe ON pn.nodeid IN (pe.from_node, pe.to_node)
            WHERE pe.pipedef_id = ?
        """, (pipeline_id,))
        nodes = []
        for row in c.fetchall():
            try:
                params_dict = json.loads(row[2]) if row[2] else {}
            except:
                params_dict = {}
            nodes.append({
                "id": row[0],
                "func": row[1],
                "params": params_dict
            })
        
        # 获取pipeline边
        c.execute("SELECT from_node, to_node FROM pipe_edges WHERE pipedef_id = ?", (pipeline_id,))
        edges = []
        for row in c.fetchall():
            edges.append({
                "from": row[0],
                "to": row[1]
            })
        
        # 获取使用此pipeline的特征定义
        c.execute("""
            SELECT id, shortname, func, chans, params
            FROM fxdef 
            WHERE pipedef_id = ?
        """, (pipeline_id,))
        fxdefs = []
        for row in c.fetchall():
            try:
                params_dict = json.loads(row[4]) if row[4] else {}
            except:
                params_dict = {}
            fxdefs.append({
                "id": row[0],
                "shortname": row[1],
                "func": row[2],
                "chans": row[3],
                "params": params_dict
            })
        
        conn.close()
        
        # 获取Cytoscape格式的数据
        cytoscape_data = get_pipeline_cytoscape_data(pipeline_id)
        
        return {
            "pipeline_id": pipeline_id,
            "pipeline_name": pipeline_name,
            "pipeline_description": pipeline_desc,
            "cytoscape_data": cytoscape_data,
            "nodes": nodes,
            "edges": edges,
            "fxdefs": fxdefs,
            "node_count": len(nodes),
            "edge_count": len(edges),
            "fxdef_count": len(fxdefs)
        }
        
    except Exception as e:
        print(f"获取pipeline可视化数据时出错: {e}")
        return None

if __name__ == "__main__":
    # 测试代码
    pipeline_id = 1
    result = get_pipeline_visualization_data(pipeline_id)
    if result:
        print(f"Pipeline: {result['pipeline_name']}")
        print(f"节点数: {result['node_count']}")
        print(f"边数: {result['edge_count']}")
        print(f"特征定义数: {result['fxdef_count']}")
        if result['cytoscape_data']:
            print("Cytoscape数据已生成")
    else:
        print("无法获取pipeline数据") 