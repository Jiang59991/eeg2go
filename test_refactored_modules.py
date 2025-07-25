#!/usr/bin/env python3
"""
测试重构后的模块协作
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eeg2fx.featureset_grouping import build_feature_dag, load_fxdefs_for_set, toposort
from eeg2fx.featureset_fetcher import run_feature_set
from eeg2fx.node_executor import NodeExecutor

def test_module_cooperation():
    """测试模块协作"""
    print("=== 测试模块协作 ===")
    
    try:
        # 1. 测试DAG构建
        feature_set_id = 1
        fxdefs = load_fxdefs_for_set(feature_set_id)
        print(f"✓ 加载到 {len(fxdefs)} 个fxdefs")
        
        dag = build_feature_dag(fxdefs)
        print(f"✓ DAG构建完成，包含 {len(dag)} 个节点")
        
        # 2. 测试拓扑排序
        execution_order = toposort(dag)
        print(f"✓ 拓扑排序完成，执行顺序: {execution_order[:5]}...")
        
        # 3. 测试NodeExecutor
        recording_id = 1
        executor = NodeExecutor(recording_id)
        executor_order = executor.toposort(dag)
        print(f"✓ NodeExecutor拓扑排序: {executor_order[:5]}...")
        
        # 验证两个拓扑排序结果一致
        assert execution_order == executor_order, "拓扑排序结果不一致"
        print("✓ 拓扑排序结果一致")
        
        # 4. 测试NodeExecutor
        print("开始测试DAG执行...")
        executor = NodeExecutor(recording_id)
        node_outputs = executor.execute_dag(dag)
        print(f"✓ DAG执行完成，输出节点数: {len(node_outputs)}")
        
        print("模块协作测试通过")
        return True
        
    except Exception as e:
        print(f"模块协作测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_feature_extraction():
    """测试特征提取"""
    print("\n=== 测试特征提取 ===")
    
    try:
        feature_set_id = 1
        recording_id = 1
        
        print(f"开始提取特征集 {feature_set_id} 在录音 {recording_id} 上的特征...")
        results = run_feature_set(feature_set_id, recording_id)
        
        print(f"✓ 特征提取完成，获得 {len(results)} 个特征")
        
        # 显示前几个特征的结果
        for i, (fxid, result) in enumerate(list(results.items())[:3]):
            print(f"  特征 {fxid}: dim={result['dim']}, shape={result['shape']}")
        
        print("特征提取测试通过")
        return True
        
    except Exception as e:
        print(f"特征提取测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test1 = test_module_cooperation()
    test2 = test_feature_extraction()
    
    print(f"\n=== 测试结果 ===")
    print(f"模块协作测试: {'通过' if test1 else '失败'}")
    print(f"特征提取测试: {'通过' if test2 else '失败'}")
    
    if test1 and test2:
        print("\n🎉 所有测试通过！重构成功！")
    else:
        print("\n❌ 部分测试失败，需要进一步检查") 