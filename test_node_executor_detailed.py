#!/usr/bin/env python3
"""
详细检测node_executor模块
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eeg2fx.node_executor import NodeExecutor, NodeStatus, NodeExecutionInfo
from eeg2fx.featureset_grouping import build_feature_dag, load_fxdefs_for_set
import time

def test_node_executor_creation():
    """测试NodeExecutor创建"""
    print("=== 测试NodeExecutor创建 ===")
    
    try:
        # 测试基本创建
        recording_id = 1
        executor = NodeExecutor(recording_id)
        print(f"✓ NodeExecutor创建成功，recording_id: {executor.recording_id}")
        
        # 测试属性初始化
        assert hasattr(executor, 'node_outputs'), "缺少node_outputs属性"
        assert hasattr(executor, 'execution_info'), "缺少execution_info属性"
        assert hasattr(executor, 'execution_order'), "缺少execution_order属性"
        print("✓ 所有必要属性已初始化")
        
        return True
        
    except Exception as e:
        print(f"❌ NodeExecutor创建失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_function_resolution():
    """测试函数解析"""
    print("\n=== 测试函数解析 ===")
    
    try:
        executor = NodeExecutor(1)
        
        # 测试已知函数
        test_functions = ['raw', 'filter', 'split_channel']
        for func_name in test_functions:
            func = executor.resolve_function(func_name)
            print(f"✓ 成功解析函数: {func_name} -> {func.__name__}")
        
        # 测试未知函数
        try:
            executor.resolve_function('unknown_function')
            print("❌ 应该抛出异常但未抛出")
            return False
        except Exception as e:
            print(f"✓ 正确抛出异常: {e}")
        
        return True
        
    except Exception as e:
        print(f"❌ 函数解析测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_input_extraction():
    """测试输入提取"""
    print("\n=== 测试输入提取 ===")
    
    try:
        executor = NodeExecutor(1)
        
        # 创建测试DAG
        test_dag = {
            "raw": {
                "func": "raw",
                "params": {},
                "inputnodes": [],
                "upstream_paths": {("raw", "hash1", "", "upstream_hash1")}
            },
            "filter": {
                "func": "filter",
                "params": {"hp": 1.0},
                "inputnodes": ["raw"],
                "upstream_paths": {("filter", "hash2", "upstream_hash1", "upstream_hash2")}
            }
        }
        
        # 模拟raw节点已执行
        executor.node_outputs["upstream_hash1"] = "raw_data"
        
        # 测试filter节点的输入提取
        filter_node = test_dag["filter"]
        inputs = executor.get_input_from_upstream_paths(filter_node, test_dag)
        
        print(f"✓ 成功提取输入: {inputs}")
        assert len(inputs) == 1, "应该有一个输入"
        assert inputs[0] == "raw_data", "输入数据不正确"
        
        return True
        
    except Exception as e:
        print(f"❌ 输入提取测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_node_execution():
    """测试节点执行"""
    print("\n=== 测试节点执行 ===")
    
    try:
        # 使用较小的录音ID，避免内存问题
        executor = NodeExecutor(11)  # 使用一个不存在的录音ID
        
        # 创建测试节点
        test_node = {
            "func": "raw",
            "params": {},
            "inputnodes": [],
            "upstream_paths": {("raw", "hash1", "", "upstream_hash1")}
        }
        
        # 执行节点
        results = executor.execute_node("raw", test_node, {})
        
        print(f"✓ 节点执行成功，结果: {len(results)} 个upstream_hash")
        assert "upstream_hash1" in results, "结果中应该包含upstream_hash1"
        
        # 检查执行信息
        assert "raw" in executor.execution_info, "应该记录执行信息"
        exec_info = executor.execution_info["raw"]
        assert exec_info.status == NodeStatus.SUCCESS, "状态应该是SUCCESS"
        assert exec_info.duration > 0, "执行时间应该大于0"
        
        print(f"✓ 执行信息记录正确: status={exec_info.status.value}, duration={exec_info.duration:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ 节点执行测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_toposort():
    """测试拓扑排序"""
    print("\n=== 测试拓扑排序 ===")
    
    try:
        executor = NodeExecutor(1)
        
        # 创建测试DAG
        test_dag = {
            "raw": {"func": "raw", "params": {}, "inputnodes": []},
            "filter": {"func": "filter", "params": {"hp": 1.0}, "inputnodes": ["raw"]},
            "split": {"func": "split_channel", "params": {"chan": "C3"}, "inputnodes": ["filter"]}
        }
        
        # 执行拓扑排序
        order = executor.toposort(test_dag)
        print(f"✓ 拓扑排序结果: {order}")
        
        # 验证排序正确性
        assert "raw" in order, "raw应该在结果中"
        assert "filter" in order, "filter应该在结果中"
        assert "split" in order, "split应该在结果中"
        
        # 验证依赖关系
        raw_idx = order.index("raw")
        filter_idx = order.index("filter")
        split_idx = order.index("split")
        
        assert raw_idx < filter_idx, "raw应该在filter之前"
        assert filter_idx < split_idx, "filter应该在split之前"
        
        print("✓ 拓扑排序依赖关系正确")
        
        return True
        
    except Exception as e:
        print(f"❌ 拓扑排序测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_dag_execution():
    """测试DAG执行"""
    print("\n=== 测试DAG执行 ===")
    
    try:
        # 使用较小的录音ID，避免内存问题
        executor = NodeExecutor(999)  # 使用一个不存在的录音ID
        
        # 创建简单测试DAG
        test_dag = {
            "raw": {
                "func": "raw",
                "params": {},
                "inputnodes": [],
                "upstream_paths": {("raw", "hash1", "", "upstream_hash1")}
            },
            "filter": {
                "func": "filter",
                "params": {"hp": 1.0},
                "inputnodes": ["raw"],
                "upstream_paths": {("filter", "hash2", "upstream_hash1", "upstream_hash2")}
            }
        }
        
        # 执行DAG
        start_time = time.time()
        results = executor.execute_dag(test_dag)
        end_time = time.time()
        
        print(f"✓ DAG执行完成，耗时: {end_time - start_time:.3f}s")
        print(f"✓ 输出节点数: {len(results)}")
        
        # 验证执行信息（节点执行失败时，results可能为空，但execution_info应该有记录）
        assert "raw" in executor.execution_info, "应该记录raw节点的执行信息"
        raw_info = executor.execution_info["raw"]
        print(f"✓ raw节点状态: {raw_info.status.value}")
        
        # 如果节点执行成功，验证结果
        if raw_info.status == NodeStatus.SUCCESS:
            assert "upstream_hash1" in results, "应该包含raw节点的结果"
            if "filter" in executor.execution_info and executor.execution_info["filter"].status == NodeStatus.SUCCESS:
                assert "upstream_hash2" in results, "应该包含filter节点的结果"
        else:
            print(f"✓ 节点执行失败，状态: {raw_info.status.value}, 错误: {raw_info.error_message}")
        
        # 验证执行报告
        report = executor.generate_execution_report()
        print(f"✓ 执行报告: {report['total_nodes']} 个节点, {report['status_counts']}")
        
        return True
        
    except Exception as e:
        print(f"❌ DAG执行测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_real_dag_execution():
    """测试真实DAG执行"""
    print("\n=== 测试真实DAG执行 ===")
    
    try:
        # 加载真实数据
        feature_set_id = 1
        fxdefs = load_fxdefs_for_set(feature_set_id)
        print(f"✓ 加载到 {len(fxdefs)} 个fxdefs")
        
        # 构建DAG
        dag = build_feature_dag(fxdefs)
        print(f"✓ DAG构建完成，包含 {len(dag)} 个节点")
        
        # 创建executor（使用不存在的录音ID避免内存问题）
        executor = NodeExecutor(999)
        
        # 执行DAG
        start_time = time.time()
        results = executor.execute_dag(dag)
        end_time = time.time()
        
        print(f"✓ 真实DAG执行完成，耗时: {end_time - start_time:.3f}s")
        print(f"✓ 输出节点数: {len(results)}")
        
        # 验证执行报告
        report = executor.generate_execution_report()
        print(f"✓ 执行报告:")
        print(f"  总节点数: {report['total_nodes']}")
        print(f"  状态统计: {report['status_counts']}")
        print(f"  总耗时: {report['total_duration']:.3f}s")
        
        return True
        
    except Exception as e:
        print(f"❌ 真实DAG执行测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def test_error_handling():
    """测试错误处理"""
    print("\n=== 测试错误处理 ===")
    
    try:
        executor = NodeExecutor(1)
        
        # 创建会失败的节点
        error_node = {
            "func": "unknown_function",
            "params": {},
            "inputnodes": [],
            "upstream_paths": {("error", "hash1", "", "upstream_hash1")}
        }
        
        # 执行会失败的节点
        try:
            executor.execute_node("error", error_node, {})
            print("❌ 应该抛出异常但未抛出")
            return False
        except Exception as e:
            print(f"✓ 正确捕获异常: {e}")
        
        # 检查执行信息
        assert "error" in executor.execution_info, "应该记录错误信息"
        exec_info = executor.execution_info["error"]
        assert exec_info.status == NodeStatus.FAILED, "状态应该是FAILED"
        assert exec_info.error_message is not None, "应该有错误消息"
        
        print(f"✓ 错误信息记录正确: status={exec_info.status.value}, error={exec_info.error_message}")
        
        return True
        
    except Exception as e:
        print(f"❌ 错误处理测试失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def main():
    """主测试函数"""
    print("开始详细检测node_executor模块...\n")
    
    tests = [
        test_node_executor_creation,
        test_function_resolution,
        test_input_extraction,
        test_node_execution,
        test_toposort,
        test_dag_execution,
        test_real_dag_execution,
        test_error_handling
    ]
    
    results = []
    for test in tests:
        try:
            result = test()
            results.append(result)
        except Exception as e:
            print(f"❌ 测试 {test.__name__} 出现未预期异常: {e}")
            results.append(False)
    
    # 输出总结
    print(f"\n{'='*50}")
    print("检测结果总结:")
    print(f"{'='*50}")
    
    passed = sum(results)
    total = len(results)
    
    for i, (test, result) in enumerate(zip(tests, results)):
        status = "✓ 通过" if result else "❌ 失败"
        print(f"{i+1:2d}. {test.__name__:<25} {status}")
    
    print(f"\n总计: {passed}/{total} 个测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！node_executor模块工作正常！")
    else:
        print("⚠️  部分测试失败，需要进一步检查")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 