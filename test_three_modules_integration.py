#!/usr/bin/env python3
"""
测试三个模块的集成：featureset_grouping, featureset_fetcher, node_executor
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eeg2fx.featureset_grouping import build_feature_dag, load_fxdefs_for_set
from eeg2fx.featureset_fetcher import run_feature_set
from eeg2fx.node_executor import NodeExecutor

def test_dag_building():
    """测试DAG构建"""
    print("=== 测试DAG构建 ===")
    
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
    
    return dag

def test_node_executor():
    """测试NodeExecutor"""
    print("\n=== 测试NodeExecutor ===")
    
    feature_set_id = 1
    fxdefs = load_fxdefs_for_set(feature_set_id)
    dag = build_feature_dag(fxdefs)
    
    recording_id = 1
    executor = NodeExecutor(recording_id)
    
    print(f"开始执行DAG，包含 {len(dag)} 个节点")
    
    try:
        # 执行DAG
        node_outputs = executor.execute_dag(dag)
        
        print(f"执行完成，输出节点数: {len(node_outputs)}")
        
        # 获取执行报告
        report = executor.generate_execution_report()
        print(f"执行报告: {report}")
        
        # 获取状态摘要
        summary = executor.get_dag_status_summary()
        print(f"状态摘要: {summary}")
        
        return node_outputs
        
    except Exception as e:
        print(f"执行失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_featureset_fetcher():
    """测试featureset_fetcher"""
    print("\n=== 测试featureset_fetcher ===")
    
    feature_set_id = 1
    recording_id = 1
    
    try:
        # 运行特征提取
        results = run_feature_set(feature_set_id, recording_id)
        
        print(f"提取到 {len(results)} 个特征:")
        for fxid, res in results.items():
            dim = res["dim"]
            shape = res["shape"]
            preview = res["value"][:5] if isinstance(res["value"], list) else res["value"]
            shape_str = "×".join(str(s) for s in shape) if shape else "-"
            print(f"fxdef_id={fxid:>3} | dim={dim:<6} | shape={shape_str:<8} | value={preview}")
        
        return results
        
    except Exception as e:
        print(f"特征提取失败: {e}")
        import traceback
        traceback.print_exc()
        return None

def test_integration():
    """测试三个模块的集成"""
    print("=== 测试三个模块的集成 ===")
    
    # 1. 测试DAG构建
    dag = test_dag_building()
    
    # 2. 测试NodeExecutor
    node_outputs = test_node_executor()
    
    # 3. 测试featureset_fetcher
    results = test_featureset_fetcher()
    
    print("\n=== 集成测试完成 ===")
    print(f"DAG构建: {'成功' if dag else '失败'}")
    print(f"NodeExecutor: {'成功' if node_outputs else '失败'}")
    print(f"featureset_fetcher: {'成功' if results else '失败'}")

if __name__ == "__main__":
    test_integration() 