#!/usr/bin/env python3
"""
简单的特征统计实验测试脚本
"""

import os
import sys
from feature_mill.experiment_engine import run_experiment

def test_feature_statistics():
    """测试特征统计实验"""
    print("🔬 开始特征统计实验测试")
    
    # 实验参数
    dataset_id = 3  # minimal_harvard数据集
    feature_set_id = 1
    output_dir = "data/experiments/feature_statistics_test"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        print("📊 运行特征统计实验...")
        
        result = run_experiment(
            experiment_type='feature_statistics',
            dataset_id=dataset_id,
            feature_set_id=feature_set_id,
            output_dir=output_dir,
            extra_args={
                'top_n_features': 10,
                'plot_distributions': True,
                'plot_correlation_heatmap': True,
                'plot_outliers': True
            }
        )
        
        print("✅ 特征统计实验完成")
        print(f"📁 结果保存在: {output_dir}")
        print(f"🔍 实验ID: {result.get('experiment_result_id', 'N/A')}")
        print(f"⏱️ 耗时: {result.get('duration', 'N/A')}秒")
        
        # 检查生成的文件
        print("\n📋 生成的文件:")
        if os.path.exists(output_dir):
            files = os.listdir(output_dir)
            for file in files:
                print(f"  - {file}")
        
        return result
        
    except Exception as e:
        print(f"❌ 特征统计实验测试失败: {e}")
        return None

if __name__ == "__main__":
    test_feature_statistics() 