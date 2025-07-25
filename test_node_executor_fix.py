#!/usr/bin/env python3
"""
测试node_executor的修复
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eeg2fx.node_executor import NodeExecutor
from eeg2fx.featureset_grouping import build_feature_dag, load_fxdefs_for_set

def test_node_executor_basic():
    """测试NodeExecutor的基本功能"""
    print("=== 测试NodeExecutor基本功能 ===")
    
    try:
        # 创建executor
        recording_id = 1
        executor = NodeExecutor(recording_id)
        print(f"NodeExecutor创建成功，recording_id: {recording_id}")
        
        # 测试拓扑排序
        test_dag = {
            "raw": {"func": "raw", "params": {}, "inputnodes": []},
            "filter": {"func": "filter", "params": {"hp": 1.0}, "inputnodes": ["raw"]}
        }
        
        execution_order = executor.toposort(test_dag)
        print(f"拓扑排序结果: {execution_order}")
        
        print("基本功能测试通过")
        return True
        
    except Exception as e:
        print(f"基本功能测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_node_executor_with_real_dag():
    """测试NodeExecutor与真实DAG"""
    print("\n=== 测试NodeExecutor与真实DAG ===")
    
    try:
        # 加载feature set 1
        feature_set_id = 1
        fxdefs = load_fxdefs_for_set(feature_set_id)
        print(f"加载到 {len(fxdefs)} 个fxdefs")
        
        # 构建DAG
        dag = build_feature_dag(fxdefs)
        print(f"DAG包含 {len(dag)} 个节点")
        
        # 创建executor
        recording_id = 1
        executor = NodeExecutor(recording_id)
        
        # 测试拓扑排序
        execution_order = executor.toposort(dag)
        print(f"拓扑排序结果: {execution_order[:5]}...")  # 只显示前5个
        
        print("真实DAG测试通过")
        return True
        
    except Exception as e:
        print(f"真实DAG测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test1 = test_node_executor_basic()
    test2 = test_node_executor_with_real_dag()
    
    print(f"\n=== 测试结果 ===")
    print(f"基本功能测试: {'通过' if test1 else '失败'}")
    print(f"真实DAG测试: {'通过' if test2 else '失败'}") 