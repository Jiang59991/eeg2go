#!/usr/bin/env python3
"""
测试新的缓存逻辑
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eeg2fx.node_executor import NodeExecutor
from eeg2fx.featureset_grouping import build_feature_dag, load_fxdefs_for_set

def test_upstream_paths():
    """测试upstream路径"""
    print("=== 测试upstream路径 ===")
    
    # 模拟一个节点的upstream路径
    upstream_paths = [
        ("filter_hash_1", "raw_hash", "raw", "upstream_hash_1"),
        ("filter_hash_2", "raw_hash", "raw", "upstream_hash_2"),
    ]
    
    print(f"Upstream路径数: {len(upstream_paths)}")
    
    # 为每个upstream路径生成缓存键
    cache_keys = []
    for i, upstream_path in enumerate(upstream_paths):
        cache_key = str(upstream_path)
        cache_keys.append(cache_key)
        print(f"  路径 {i+1}: {upstream_path}")
        print(f"  缓存键: {cache_key}")
    
    # 检查缓存键是否唯一
    unique_keys = set(cache_keys)
    print(f"\n唯一缓存键数: {len(unique_keys)}")
    print(f"总缓存键数: {len(cache_keys)}")
    print(f"缓存键是否唯一: {len(unique_keys) == len(cache_keys)}")

def test_dag_execution_logic():
    """测试DAG执行逻辑"""
    print("\n=== 测试DAG执行逻辑 ===")
    
    # 加载feature set 1
    feature_set_id = 1
    fxdefs = load_fxdefs_for_set(feature_set_id)
    
    print(f"加载到 {len(fxdefs)} 个fxdefs")
    
    # 构建DAG
    dag = build_feature_dag(fxdefs)
    print(f"DAG包含 {len(dag)} 个节点")
    
    # 检查每个节点的upstream路径
    for node_id, node in dag.items():
        func_name = node["func"]
        params = node["params"]
        input_ids = node["inputnodes"]
        upstream_paths = node.get("upstream_paths", set())
        
        print(f"\n节点: {node_id}")
        print(f"  函数: {func_name}")
        print(f"  输入节点: {input_ids}")
        print(f"  Upstream路径数: {len(upstream_paths)}")
        
        # 显示upstream路径
        for i, upstream_path in enumerate(upstream_paths):
            print(f"    路径 {i+1}: {upstream_path}")
            if len(upstream_path) >= 4:
                node_hash, parent_hash, inputnodes, upstream_hash = upstream_path
                print(f"      节点哈希: {node_hash}")
                print(f"      父节点哈希: {parent_hash}")
                print(f"      输入节点: {inputnodes}")
                print(f"      上游哈希: {upstream_hash}")

if __name__ == "__main__":
    test_upstream_paths()
    test_dag_execution_logic() 