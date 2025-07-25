#!/usr/bin/env python3
"""
测试修改后的featureset_fetcher
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eeg2fx.featureset_fetcher import run_feature_set, build_feature_dag, load_fxdefs_for_set

def test_featureset_fetcher():
    """测试featureset_fetcher的功能"""
    print("=== 测试featureset_fetcher ===")
    
    # 测试feature set 1
    feature_set_id = 1
    recording_id = 1
    
    print(f"运行feature set {feature_set_id} on recording {recording_id}")
    
    try:
        # 运行特征提取
        results = run_feature_set(feature_set_id, recording_id)
        
        print(f"\n提取到 {len(results)} 个特征:")
        for fxid, res in results.items():
            dim = res["dim"]
            shape = res["shape"]
            preview = res["value"][:5] if isinstance(res["value"], list) else res["value"]
            shape_str = "×".join(str(s) for s in shape) if shape else "-"
            print(f"fxdef_id={fxid:>3} | dim={dim:<6} | shape={shape_str:<8} | value={preview}")
            
    except Exception as e:
        print(f"测试失败: {e}")
        import traceback
        traceback.print_exc()

def test_dag_building():
    """测试DAG构建"""
    print("\n=== 测试DAG构建 ===")
    
    feature_set_id = 1
    fxdefs = load_fxdefs_for_set(feature_set_id)
    
    print(f"加载到 {len(fxdefs)} 个fxdefs")
    
    # 构建DAG
    dag = build_feature_dag(fxdefs)
    
    print(f"DAG包含 {len(dag)} 个节点")
    
    # 检查节点类型
    node_types = {}
    for node_id, node in dag.items():
        func = node["func"]
        if func not in node_types:
            node_types[func] = []
        node_types[func].append(node_id)
    
    print(f"\n节点类型分布:")
    for func, nodes in node_types.items():
        print(f"  {func}: {len(nodes)} 个节点")
    
    # 检查raw节点
    if 'raw' in dag:
        raw_node = dag['raw']
        print(f"\nRaw节点信息:")
        print(f"  func: {raw_node['func']}")
        print(f"  inputnodes: {raw_node['inputnodes']}")
        print(f"  pipeline_paths: {len(raw_node['pipeline_paths'])} 个路径")
        print(f"  upstream_paths: {len(raw_node['upstream_paths'])} 个路径")
        print(f"  fxdef_ids: {len(raw_node['fxdef_ids'])} 个fxdef")

if __name__ == "__main__":
    test_dag_building()
    test_featureset_fetcher() 