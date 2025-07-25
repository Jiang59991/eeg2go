#!/usr/bin/env python3
"""
调试DAG结构和节点输出
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eeg2fx.featureset_grouping import build_feature_dag, load_fxdefs_for_set
from eeg2fx.node_executor import NodeExecutor

def debug_dag_structure():
    """调试DAG结构"""
    print("=== 调试DAG结构 ===")
    
    # 加载数据
    feature_set_id = 4
    fxdefs = load_fxdefs_for_set(feature_set_id)
    print(f"加载到 {len(fxdefs)} 个fxdefs")
    
    # 构建DAG
    dag = build_feature_dag(fxdefs)
    print(f"DAG包含 {len(dag)} 个节点")
    
    # 显示DAG结构
    print("\n=== DAG节点结构 ===")
    for node_id, node in dag.items():
        func_name = node["func"]
        params = node["params"]
        input_ids = node["inputnodes"]
        upstream_paths = node.get("upstream_paths", set())
        fxdef_ids = node.get("fxdef_ids", [])
        
        print(f"\n节点: {node_id}")
        print(f"  函数: {func_name}")
        print(f"  参数: {params}")
        print(f"  输入节点: {input_ids}")
        print(f"  Upstream路径数: {len(upstream_paths)}")
        print(f"  关联的fxdef_ids: {fxdef_ids}")
        
        # 显示upstream路径
        for i, upstream_path in enumerate(upstream_paths):
            print(f"    路径 {i+1}: {upstream_path}")
    
    # 执行DAG
    print("\n=== 执行DAG ===")
    recording_id = 22
    executor = NodeExecutor(recording_id)
    node_outputs = executor.execute_dag(dag)
    
    print(f"\n=== 节点输出 ===")
    print(f"输出节点数: {len(node_outputs)}")
    for upstream_hash, output in node_outputs.items():
        print(f"  {upstream_hash}: {type(output)} - {str(output)[:100]}...")
    
    # 分析特征提取问题
    print("\n=== 分析特征提取问题 ===")
    for fx in fxdefs:
        fxid = fx["id"]
        chan = fx["chans"]
        func = fx["func"]
        
        print(f"\n特征 {fxid}:")
        print(f"  通道: {chan}")
        print(f"  函数: {func}")
        
        # 查找对应的分割节点
        split_found = False
        for node_id, node in dag.items():
            if (node["func"] == "split_channel" and 
                node["params"].get("chan") == chan and
                fxid in node.get("fxdef_ids", [])):
                print(f"  找到分割节点: {node_id}")
                split_found = True
                break
        
        if not split_found:
            print(f"  ❌ 未找到分割节点")
            
            # 查找特征节点
            feature_found = False
            for node_id, node in dag.items():
                if (node["func"] == func and fxid in node.get("fxdef_ids", [])):
                    print(f"  找到特征节点: {node_id}")
                    feature_found = True
                    
                    # 检查特征节点的输出
                    upstream_paths = node.get("upstream_paths", set())
                    for upstream_path in upstream_paths:
                        if len(upstream_path) >= 4:
                            upstream_hash = upstream_path[3]
                            if upstream_hash in node_outputs:
                                print(f"    特征节点输出: {upstream_hash}")
                            else:
                                print(f"    ❌ 特征节点输出未找到: {upstream_hash}")
                    break
            
            if not feature_found:
                print(f"  ❌ 未找到特征节点")

if __name__ == "__main__":
    debug_dag_structure() 