#!/usr/bin/env python3
"""
相关性分析实验示例脚本

该脚本演示如何使用 experiment_engine 运行相关性分析实验。
专门设计用于测试年龄相关的EEG特征，现在支持recording级别的统计分析。
"""

import os
import logging
import sys
from feature_mill.experiment_engine import run_experiment, list_experiments, get_experiment_info, setup_logging

def main():
    """主函数"""
    print("=" * 60)
    print("EEG2Go 年龄相关性分析实验示例 (Recording-Level)")
    print("=" * 60)

    # 初始化日志系统（写入 experiment.log）
    from feature_mill.experiment_engine import setup_logging
    log_path = os.path.join('data', 'processed', 'age_correlation_analysis', 'experiment.log')
    setup_logging(log_file=log_path, log_level=logging.INFO)
    
    # 检查可用的实验
    available_experiments = list_experiments()
    print(f"可用的实验模块: {available_experiments}")
    
    if 'correlation' not in available_experiments:
        print("错误: 未找到 correlation 实验模块")
        return
    
    # 获取实验信息
    experiment_info = get_experiment_info('correlation')
    print(f"\n实验信息:")
    print(f"  名称: {experiment_info['name']}")
    print(f"  模块: {experiment_info['module']}")
    print(f"  有run函数: {experiment_info['has_run_function']}")
    
    # 实验参数 - 专门针对年龄相关性分析（recording级别）
    experiment_params = {
        'experiment_type': 'correlation',
        'dataset_id': 1,  # 请根据实际数据集ID调整
        'feature_set_id': 3,  # 使用年龄相关特征集
        'output_dir': 'data/experiments/age_correlation_analysis',
        'extra_args': {
            'target_vars': ['age'],  # 分析年龄相关变量
            'method': 'pearson',  # 使用皮尔逊相关系数
            'min_corr': 0.1,  # 降低阈值以显示更多相关性
            'top_n': 20,  # 显示前20个最相关特征
            'plot_corr_matrix': True,
            'plot_scatter': True,
            'save_detailed_results': True  # 保存详细结果
        },
        'db_path': 'database/eeg2go.db'
    }
    
    print(f"\n实验参数:")
    print(f"  数据集ID: {experiment_params['dataset_id']}")
    print(f"  特征集ID: {experiment_params['feature_set_id']}")
    print(f"  输出目录: {experiment_params['output_dir']}")
    print(f"  目标变量: {experiment_params['extra_args']['target_vars']}")
    print(f"  分析方法: {experiment_params['extra_args']['method']}")
    print(f"  最小相关系数: {experiment_params['extra_args']['min_corr']}")
    print(f"  显示前N个特征: {experiment_params['extra_args']['top_n']}")
    
    print(f"\n预期结果 (Recording-Level):")
    print(f"  1. Alpha峰值频率_mean - 应与年龄呈负相关")
    print(f"  2. Alpha功率_mean - 应与年龄呈负相关")
    print(f"  3. Theta/Alpha比值_mean - 应与年龄呈正相关")
    print(f"  4. Beta功率_mean - 应与年龄呈负相关")
    print(f"  5. 频谱边缘频率_mean - 应与年龄呈负相关")
    print(f"  6. Alpha不对称性_mean - 可能与年龄相关")
    print(f"  7. 各种统计量 (std, min, max, median, count) 也可能与年龄相关")
    
    print(f"\nRecording级别统计说明:")
    print(f"  - _mean: 所有epoch的平均值")
    print(f"  - _std: 所有epoch的标准差")
    print(f"  - _min: 所有epoch的最小值")
    print(f"  - _max: 所有epoch的最大值")
    print(f"  - _median: 所有epoch的中位数")
    print(f"  - _count: 有效epoch的数量")
    
    try:
        # 运行实验
        print("\n开始运行实验...")
        result = run_experiment(**experiment_params)
        
        print("\n" + "=" * 60)
        print("实验完成!")
        print("=" * 60)
        print(f"状态: {result['status']}")
        print(f"输出目录: {result['output_dir']}")
        print(f"运行时间: {result['duration']:.2f} 秒")
        print(f"\n实验摘要:")
        print(result['summary'])
        
        # 显示结果文件
        output_dir = experiment_params['output_dir']
        if os.path.exists(output_dir):
            print(f"\n生成的结果文件:")
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    print(f"  {file} ({size} bytes)")
        
    except Exception as e:
        print(f"\n实验运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 