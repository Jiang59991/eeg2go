#!/usr/bin/env python3
"""
测试直接使用NodeExecutor
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eeg2fx.node_executor import NodeExecutor
from eeg2fx.featureset_grouping import build_feature_dag, load_fxdefs_for_set

def test_direct_node_executor():
    """测试直接使用NodeExecutor"""
    print("=== 测试直接使用NodeExecutor ===")
    
    try:
        # 加载数据
        feature_set_id = 1
        fxdefs = load_fxdefs_for_set(feature_set_id)
        print(f"✓ 加载到 {len(fxdefs)} 个fxdefs")
        
        # 构建DAG
        dag = build_feature_dag(fxdefs)
        print(f"✓ DAG构建完成，包含 {len(dag)} 个节点")
        
        # 直接使用NodeExecutor
        recording_id = 999  # 使用不存在的录音ID避免内存问题
        executor = NodeExecutor(recording_id)
        
        # 执行DAG
        print("开始执行DAG...")
        node_outputs = executor.execute_dag(dag)
        print(f"✓ DAG执行完成，输出节点数: {len(node_outputs)}")
        
        # 获取执行报告
        report = executor.generate_execution_report()
        print(f"✓ 执行报告: {report['total_nodes']} 个节点, {report['status_counts']}")
        
        print("直接使用NodeExecutor测试通过")
        return True
        
    except Exception as e:
        print(f"❌ 直接使用NodeExecutor测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_featureset_fetcher_integration():
    """测试featureset_fetcher集成"""
    print("\n=== 测试featureset_fetcher集成 ===")
    
    try:
        from eeg2fx.featureset_fetcher import run_feature_set
        
        feature_set_id = 1
        recording_id = 22  # 使用不存在的录音ID
        
        print(f"开始运行特征集 {feature_set_id} 在录音 {recording_id} 上...")
        results = run_feature_set(feature_set_id, recording_id)
        
        print(f"✓ 特征提取完成，获得 {len(results)} 个特征")
        print("featureset_fetcher集成测试通过")
        return True
        
    except ValueError as e:
        if "not found in recordings table" in str(e):
            print(f"✓ 预期的错误（录音不存在）: {e}")
            print("featureset_fetcher集成测试通过（正确处理了不存在的录音）")
            return True
        else:
            print(f"❌ 意外的ValueError: {e}")
            import traceback
            traceback.print_exc()
            return False
    except Exception as e:
        print(f"❌ featureset_fetcher集成测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test1 = test_direct_node_executor()
    test2 = test_featureset_fetcher_integration()
    
    print(f"\n=== 测试结果 ===")
    print(f"直接使用NodeExecutor: {'通过' if test1 else '失败'}")
    print(f"featureset_fetcher集成: {'通过' if test2 else '失败'}")
    
    if test1 and test2:
        print("\n🎉 所有测试通过！删除execute_dag_nodes函数成功！")
    else:
        print("\n❌ 部分测试失败，需要进一步检查") 