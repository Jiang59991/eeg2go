#!/usr/bin/env python3
"""
实验引擎测试脚本

该脚本测试实验引擎的基本功能，包括模块发现、参数验证等。
"""

import os
import sys
from feature_mill.experiment_engine import list_experiments, get_experiment_info

def test_experiment_discovery():
    """测试实验模块发现功能"""
    print("=" * 50)
    print("测试实验模块发现功能")
    print("=" * 50)
    
    experiments = list_experiments()
    print(f"发现的实验模块: {experiments}")
    
    if not experiments:
        print("警告: 未发现任何实验模块")
        return False
    
    return True

def test_experiment_info():
    """测试实验信息获取功能"""
    print("\n" + "=" * 50)
    print("测试实验信息获取功能")
    print("=" * 50)
    
    experiments = list_experiments()
    
    for exp_name in experiments:
        print(f"\n获取实验 '{exp_name}' 的信息:")
        info = get_experiment_info(exp_name)
        
        if 'error' in info:
            print(f"  错误: {info['error']}")
            return False
        else:
            print(f"  名称: {info['name']}")
            print(f"  模块: {info['module']}")
            print(f"  有run函数: {info['has_run_function']}")
            print(f"  文档: {info['docstring'][:100]}...")
    
    return True

def test_imports():
    """测试必要的导入"""
    print("\n" + "=" * 50)
    print("测试必要的导入")
    print("=" * 50)
    
    try:
        from feature_mill.experiment_engine import run_experiment
        print("✓ run_experiment 导入成功")
    except ImportError as e:
        print(f"✗ run_experiment 导入失败: {e}")
        return False
    
    try:
        from eeg2fx.featureset_fetcher import run_feature_set
        print("✓ eeg2fx.featureset_fetcher 导入成功")
    except ImportError as e:
        print(f"✗ eeg2fx.featureset_fetcher 导入失败: {e}")
        return False
    
    try:
        from eeg2fx.featureset_grouping import load_fxdefs_for_set
        print("✓ eeg2fx.featureset_grouping 导入成功")
    except ImportError as e:
        print(f"✗ eeg2fx.featureset_grouping 导入失败: {e}")
        return False
    
    try:
        import pandas as pd
        print("✓ pandas 导入成功")
    except ImportError as e:
        print(f"✗ pandas 导入失败: {e}")
        return False
    
    try:
        import matplotlib.pyplot as plt
        print("✓ matplotlib 导入成功")
    except ImportError as e:
        print(f"✗ matplotlib 导入失败: {e}")
        return False
    
    try:
        from scipy import stats
        print("✓ scipy 导入成功")
    except ImportError as e:
        print(f"✗ scipy 导入失败: {e}")
        return False
    
    return True

def test_database_connection():
    """测试数据库连接"""
    print("\n" + "=" * 50)
    print("测试数据库连接")
    print("=" * 50)
    
    db_path = "database/eeg2go.db"
    
    if not os.path.exists(db_path):
        print(f"✗ 数据库文件不存在: {db_path}")
        return False
    
    try:
        import sqlite3
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 检查必要的表是否存在
        tables = ['datasets', 'subjects', 'recordings', 'feature_sets', 'fxdef']
        for table in tables:
            cursor.execute(f"SELECT name FROM sqlite_master WHERE type='table' AND name='{table}'")
            if cursor.fetchone():
                print(f"✓ 表 '{table}' 存在")
            else:
                print(f"✗ 表 '{table}' 不存在")
        
        conn.close()
        print("✓ 数据库连接成功")
        return True
        
    except Exception as e:
        print(f"✗ 数据库连接失败: {e}")
        return False

def test_experiment_module_structure():
    """测试实验模块结构"""
    print("\n" + "=" * 50)
    print("测试实验模块结构")
    print("=" * 50)
    
    experiments = list_experiments()
    all_passed = True
    
    for exp_name in experiments:
        print(f"\n检查实验模块 '{exp_name}':")
        
        try:
            # 尝试导入模块
            module_name = f"feature_mill.experiments.{exp_name}"
            module = __import__(module_name, fromlist=['run'])
            
            # 检查是否有run函数
            if hasattr(module, 'run'):
                print(f"  ✓ 包含 run 函数")
                
                # 检查函数签名
                import inspect
                sig = inspect.signature(module.run)
                params = list(sig.parameters.keys())
                
                expected_params = ['df_feat', 'df_meta', 'output_dir']
                missing_params = [p for p in expected_params if p not in params]
                
                if not missing_params:
                    print(f"  ✓ 函数签名正确")
                else:
                    print(f"  ✗ 缺少参数: {missing_params}")
                    all_passed = False
            else:
                print(f"  ✗ 缺少 run 函数")
                all_passed = False
                
        except Exception as e:
            print(f"  ✗ 模块检查失败: {e}")
            all_passed = False
    
    return all_passed

def test_feature_extraction_functions():
    """测试特征提取相关函数"""
    print("\n" + "=" * 50)
    print("测试特征提取相关函数")
    print("=" * 50)
    
    try:
        from feature_mill.experiment_engine import (
            get_recording_ids_for_dataset,
            get_fxdef_meta,
            extract_feature_matrix_direct,
            get_relevant_metadata
        )
        print("✓ 特征提取函数导入成功")
        
        # 测试数据库连接
        db_path = "database/eeg2go.db"
        if os.path.exists(db_path):
            try:
                # 测试获取记录ID
                recording_ids = get_recording_ids_for_dataset(1, db_path)
                print(f"✓ 获取记录ID成功，数据集1有 {len(recording_ids)} 条记录")
            except Exception as e:
                print(f"✗ 获取记录ID失败: {e}")
        else:
            print("⚠ 数据库文件不存在，跳过特征提取测试")
        
        return True
        
    except ImportError as e:
        print(f"✗ 特征提取函数导入失败: {e}")
        return False

def main():
    """主测试函数"""
    print("EEG2Go 实验引擎测试")
    print("=" * 60)
    
    # 运行所有测试
    tests = [
        ("导入测试", test_imports),
        ("数据库连接测试", test_database_connection),
        ("实验发现测试", test_experiment_discovery),
        ("实验信息测试", test_experiment_info),
        ("模块结构测试", test_experiment_module_structure),
        ("特征提取函数测试", test_feature_extraction_functions),
    ]
    
    results = []
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n测试 '{test_name}' 出现异常: {e}")
            results.append((test_name, False))
    
    # 输出测试结果摘要
    print("\n" + "=" * 60)
    print("测试结果摘要")
    print("=" * 60)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ 通过" if result else "✗ 失败"
        print(f"{test_name}: {status}")
        if result:
            passed += 1
    
    print(f"\n总体结果: {passed}/{total} 测试通过")
    
    if passed == total:
        print("🎉 所有测试通过！实验引擎可以正常使用。")
    else:
        print("⚠️  部分测试失败，请检查相关配置。")

if __name__ == "__main__":
    main() 