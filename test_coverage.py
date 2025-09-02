#!/usr/bin/env python3
"""
测试覆盖率处理功能的脚本
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from feature_mill.experiment_engine import check_features_exist_in_db, extract_features_with_coverage
from logging_config import logger

def test_coverage_checking():
    """测试覆盖率检查功能"""
    print("=== 测试覆盖率检查功能 ===")
    
    # 测试参数
    dataset_id = 1  # 根据实际情况调整
    feature_set_id = 1  # 根据实际情况调整
    db_path = "database/eeg2go.db"
    
    try:
        # 测试不同的覆盖率阈值
        thresholds = [0.8, 0.9, 0.95, 0.99, 1.0]
        
        for threshold in thresholds:
            print(f"\n--- 测试覆盖率阈值: {threshold:.1%} ---")
            
            result = check_features_exist_in_db(dataset_id, feature_set_id, db_path, threshold)
            
            print(f"覆盖率: {result['coverage_ratio']:.2%}")
            print(f"缺失数量: {result['missing_count']}")
            print(f"总期望数量: {result['total_expected']}")
            print(f"100%完整: {result['exists']}")
            print(f"可使用现有数据: {result['can_use_existing']}")
            
            if result['can_use_existing']:
                print("✅ 可以使用现有数据")
            else:
                print("❌ 需要计算缺失特征")
                
    except Exception as e:
        print(f"测试失败: {e}")
        logger.error(f"测试失败: {e}")

def test_feature_extraction():
    """测试特征提取功能"""
    print("\n=== 测试特征提取功能 ===")
    
    # 测试参数
    dataset_id = 1  # 根据实际情况调整
    feature_set_id = 1  # 根据实际情况调整
    db_path = "database/eeg2go.db"
    min_coverage = 0.95
    
    try:
        print(f"尝试提取特征，最小覆盖率要求: {min_coverage:.1%}")
        
        # 检查覆盖率
        coverage_info = check_features_exist_in_db(dataset_id, feature_set_id, db_path, min_coverage)
        
        if coverage_info['can_use_existing']:
            print(f"✅ 覆盖率足够 ({coverage_info['coverage_ratio']:.2%})，开始提取...")
            
            # 提取特征
            df = extract_features_with_coverage(dataset_id, feature_set_id, db_path, min_coverage)
            
            print(f"特征矩阵提取成功: {df.shape}")
            print(f"特征列数: {df.shape[1]}")
            print(f"样本数: {df.shape[0]}")
            
            # 显示前几列
            print("\n前5列特征:")
            for i, col in enumerate(df.columns[:5]):
                print(f"  {i+1}. {col}")
                
        else:
            print(f"❌ 覆盖率不足 ({coverage_info['coverage_ratio']:.2%})，无法使用现有数据")
            
    except Exception as e:
        print(f"测试失败: {e}")
        logger.error(f"测试失败: {e}")

if __name__ == "__main__":
    print("开始测试覆盖率处理功能...")
    
    # 测试覆盖率检查
    test_coverage_checking()
    
    # 测试特征提取
    test_feature_extraction()
    
    print("\n测试完成！")
