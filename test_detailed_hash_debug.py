#!/usr/bin/env python3
"""
详细的哈希调试脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eeg2fx.featureset_grouping import load_pipeline_structure, build_feature_dag, load_fxdefs_for_set, get_node_hash
import hashlib

def debug_hash_calculation():
    """调试哈希计算过程"""
    print("=== 详细哈希调试 ===")
    
    # 加载feature set 1的fxdefs
    fxdefs = load_fxdefs_for_set(1)
    print(f"加载到 {len(fxdefs)} 个fxdefs")
    
    # 检查pipeline 1的详细哈希计算
    pipeid = 1
    print(f"\n--- Pipeline {pipeid} 详细哈希计算 ---")
    pipeline = load_pipeline_structure(pipeid)
    
    # 手动验证raw节点的哈希计算
    if 'raw' in pipeline:
        raw_node = pipeline['raw']
        print(f"Raw节点原始信息:")
        print(f"  func: {raw_node['func']}")
        print(f"  params: {raw_node['params']}")
        print(f"  inputnodes: {raw_node.get('inputnodes', 'None')}")
        
        # 手动计算node_hash
        manual_node_hash = get_node_hash('raw', raw_node)
        print(f"  手动计算的node_hash: {manual_node_hash}")
        print(f"  实际的node_hash: {raw_node.get('node_hash', 'N/A')}")
        print(f"  是否匹配: {manual_node_hash == raw_node.get('node_hash', 'N/A')}")
        
        # 手动计算upstream_hash
        parent_hash = raw_node.get('parent_hash', '')
        node_hash = raw_node.get('node_hash', '')
        manual_upstream_hash = hashlib.md5((parent_hash + node_hash).encode()).hexdigest()
        print(f"  手动计算的upstream_hash: {manual_upstream_hash}")
        print(f"  实际的upstream_hash: {raw_node.get('upstream_hash', 'N/A')}")
        print(f"  是否匹配: {manual_upstream_hash == raw_node.get('upstream_hash', 'N/A')}")
    
    # 检查所有节点的哈希值
    print(f"\n--- 所有节点的哈希值 ---")
    for nid, node in pipeline.items():
        print(f"\n节点: {nid}")
        print(f"  func: {node['func']}")
        print(f"  inputnodes: {node.get('inputnodes', 'None')}")
        print(f"  node_hash: {node.get('node_hash', 'N/A')}")
        print(f"  parent_hash: {node.get('parent_hash', 'N/A')}")
        print(f"  upstream_hash: {node.get('upstream_hash', 'N/A')}")
    
    # 构建DAG并检查raw节点的upstream_paths
    print(f"\n=== 构建DAG并检查upstream_paths ===")
    dag = build_feature_dag(fxdefs)
    
    if 'raw' in dag:
        raw_dag_node = dag['raw']
        upstream_paths = raw_dag_node['upstream_paths']
        print(f"Raw节点在DAG中的upstream_paths数量: {len(upstream_paths)}")
        
        for i, path in enumerate(upstream_paths):
            print(f"  路径 {i+1}: {path}")
            if len(path) == 4:
                node_hash, parent_hash, inputnodes, upstream_hash = path
                print(f"    节点哈希: {node_hash}")
                print(f"    父节点哈希: {parent_hash}")
                print(f"    输入节点: {inputnodes}")
                print(f"    上游哈希: {upstream_hash}")
    
    # 检查是否有其他节点引用了raw的upstream_hash
    print(f"\n=== 检查其他节点对raw upstream_hash的引用 ===")
    raw_upstream_hash = None
    if 'raw' in pipeline:
        raw_upstream_hash = pipeline['raw'].get('upstream_hash')
        print(f"Raw节点的upstream_hash: {raw_upstream_hash}")
    
    if raw_upstream_hash:
        for nid, node in pipeline.items():
            if nid != 'raw' and node.get('parent_hash') == raw_upstream_hash:
                print(f"节点 {nid} 的parent_hash引用了raw的upstream_hash")

if __name__ == "__main__":
    debug_hash_calculation() 