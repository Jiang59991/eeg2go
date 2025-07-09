#!/usr/bin/env python3
"""
性能分析测试脚本

这个脚本用于测试修复后的特征选择性能分析功能
"""

import os
import sys
import pandas as pd

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_mill.test_feature_selection_validation import analyze_selection_methods_performance

def test_performance_analysis():
    """测试性能分析功能"""
    print("🔬 开始性能分析测试")
    print("=" * 40)
    
    # 测试目录
    results_dir = "data/experiments/feature_selection_validation"
    
    try:
        # 运行性能分析
        print("📊 步骤1: 运行性能分析...")
        performance_analysis = analyze_selection_methods_performance(results_dir)
        
        print("性能分析结果:")
        for key, value in performance_analysis.items():
            if key != 'findings':
                print(f"  {key}: {value}")
        
        if performance_analysis['findings']:
            print("\n📖 发现:")
            for finding in performance_analysis['findings']:
                print(f"  - {finding}")
        
        # 检查是否有错误
        has_error = any('错误' in finding for finding in performance_analysis['findings'])
        
        if has_error:
            print("\n❌ 性能分析测试失败")
            return False
        else:
            print("\n✅ 性能分析测试成功")
            return True
            
    except Exception as e:
        print(f"❌ 性能分析测试失败: {e}")
        return False


if __name__ == "__main__":
    test_performance_analysis() 