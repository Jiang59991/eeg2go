#!/usr/bin/env python3
"""
调试raw节点的问题
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eeg2fx.featureset_grouping import load_pipeline_structure, build_feature_dag, load_fxdefs_for_set

def debug_raw_node():
    """调试raw节点的问题"""
    print("=== 调试raw节点问题 ===")
    
    # 加载feature set 1的fxdefs
    fxdefs = load_fxdefs_for_set(1)
    print(f"加载到 {len(fxdefs)} 个fxdefs")
    
    # 检查每个pipeline的raw节点
    pipeline_ids = set(fx['pipeid'] for fx in fxdefs)
    print(f"涉及的pipeline IDs: {pipeline_ids}")
    
    for pipeid in pipeline_ids:
        print(f"\n--- Pipeline {pipeid} ---")
        pipeline = load_pipeline_structure(pipeid)
        
        if 'raw' in pipeline:
            raw_node = pipeline['raw']
            print(f"Raw节点信息:")
            print(f"  func: {raw_node['func']}")
            print(f"  params: {raw_node['params']}")
            print(f"  inputnodes: {raw_node.get('inputnodes', 'None')}")
            print(f"  node_hash: {raw_node.get('node_hash', 'N/A')}")
            print(f"  parent_hash: {raw_node.get('parent_hash', 'N/A')}")
            print(f"  upstream_hash: {raw_node.get('upstream_hash', 'N/A')}")
        else:
            print("Raw节点不在这个pipeline中")
    
    # 构建DAG
    print(f"\n=== 构建DAG ===")
    dag = build_feature_dag(fxdefs)
    
    if 'raw' in dag:
        raw_dag_node = dag['raw']
        print(f"Raw节点在DAG中的信息:")
        print(f"  func: {raw_dag_node['func']}")
        print(f"  inputnodes: {raw_dag_node['inputnodes']}")
        print(f"  params: {raw_dag_node['params']}")
        print(f"  pipeline_paths: {raw_dag_node['pipeline_paths']}")
        print(f"  fxdef_ids: {raw_dag_node['fxdef_ids']}")
        print(f"  upstream_paths: {raw_dag_node['upstream_paths']}")
        
        # 检查upstream_paths的数量
        upstream_paths = raw_dag_node['upstream_paths']
        print(f"  upstream_paths数量: {len(upstream_paths)}")
        
        for i, path in enumerate(upstream_paths):
            print(f"    路径 {i+1}: {path}")
            if len(path) == 4:
                node_hash, parent_hash, inputnodes, upstream_hash = path
                print(f"      节点哈希: {node_hash}")
                print(f"      父节点哈希: {parent_hash}")
                print(f"      输入节点: {inputnodes}")
                print(f"      上游哈希: {upstream_hash}")
    else:
        print("Raw节点不在DAG中")

if __name__ == "__main__":
    debug_raw_node() 