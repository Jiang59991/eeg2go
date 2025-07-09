#!/usr/bin/env python3
"""
相关性分析实验示例脚本

该脚本演示如何使用 experiment_engine 运行相关性分析实验。
专门设计用于测试年龄相关的EEG特征，现在支持recording级别的统计分析。
"""

import os
import logging
import sys
import sqlite3
from logging_config import logger
from feature_mill.experiment_engine import run_experiment, list_experiments, get_experiment_info

def get_age_correlation_featureset_id(db_path='database/eeg2go.db'):
    """获取 age_correlation_features 特征集ID"""
    conn = sqlite3.connect(db_path)
    c = conn.cursor()
    
    # 查找 age_correlation_features 特征集
    c.execute("SELECT id, name, description FROM feature_sets WHERE name = 'age_correlation_features'")
    row = c.fetchone()
    
    if row:
        set_id, name, description = row
        # print(f"找到特征集 '{name}' (ID: {set_id})")
        # print(f"描述: {description}")
        
        # 获取特征集中的特征数量
        c.execute("SELECT COUNT(*) FROM feature_set_items WHERE feature_set_id = ?", (set_id,))
        feature_count = c.fetchone()[0]
        # print(f"特征数量: {feature_count}")
        
    else:
        # print("✗ 未找到 'age_correlation_features' 特征集")
        # print("请先运行以下命令创建特征集:")
        # print("  python setup_age_correlation.py")
        # print("  或者")
        # print("  python database/age_correlation_features.py")
        set_id = None
    
    conn.close()
    return set_id

def main():
    """主函数"""
    # log_path = os.path.join('data', 'processed', 'age_correlation_analysis', 'experiment.log')
    # setup_logging(log_file=log_path, log_level=logging.INFO)

    logger.info("=" * 60)
    logger.info("EEG2Go 年龄相关性分析实验示例 (Recording-Level)")
    logger.info("=" * 60)

    # 检查可用的实验
    available_experiments = list_experiments()
    # print(f"可用的实验模块: {available_experiments}")
    
    if 'correlation' not in available_experiments:
        logger.error("错误: 未找到 correlation 实验模块")
        return
    
    # 获取实验信息
    experiment_info = get_experiment_info('correlation')
    logger.info(f"\n实验信息:")
    
    # 检查是否有错误
    if 'error' in experiment_info:
        logger.error(f"  错误: {experiment_info['error']}")
        return
    
    # 安全地获取信息
    logger.info(f"  名称: {experiment_info.get('name', 'Unknown')}")
    logger.info(f"  模块: {experiment_info.get('module', 'Unknown')}")
    logger.info(f"  有run函数: {experiment_info.get('has_run_function', False)}")
    
    # # 动态获取年龄相关特征集ID
    # print(f"\n获取年龄相关特征集...")
    # feature_set_id = get_age_correlation_featureset_id()
    # if not feature_set_id:
    #     print("错误: 无法获取年龄相关特征集ID")
    #     return
    
    # 实验参数 - 专门针对年龄相关性分析（recording级别）
    experiment_params = {
        'experiment_type': 'correlation',
        'dataset_id': 1,  # 请根据实际数据集ID调整
        'feature_set_id': 1,  # 动态获取特征集ID
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
    
    logger.info(f"\n实验参数:")
    logger.info(f"  数据集ID: {experiment_params['dataset_id']}")
    logger.info(f"  特征集ID: {experiment_params['feature_set_id']}")
    logger.info(f"  输出目录: {experiment_params['output_dir']}")
    logger.info(f"  目标变量: {experiment_params['extra_args']['target_vars']}")
    logger.info(f"  分析方法: {experiment_params['extra_args']['method']}")
    logger.info(f"  最小相关系数: {experiment_params['extra_args']['min_corr']}")
    logger.info(f"  显示前N个特征: {experiment_params['extra_args']['top_n']}")
    
    logger.info(f"\n预期结果 (Recording-Level):")
    logger.info(f"  1. Alpha峰值频率_mean - 应与年龄呈负相关")
    logger.info(f"  2. Alpha功率_mean - 应与年龄呈负相关")
    logger.info(f"  3. Theta/Alpha比值_mean - 应与年龄呈正相关")
    logger.info(f"  4. Beta功率_mean - 应与年龄呈负相关")
    logger.info(f"  5. 频谱边缘频率_mean - 应与年龄呈负相关")
    logger.info(f"  6. Alpha不对称性_mean - 可能与年龄相关")
    logger.info(f"  7. 各种统计量 (std, min, max, median, count) 也可能与年龄相关")
    
    logger.info(f"\nRecording级别统计说明:")
    logger.info(f"  - _mean: 所有epoch的平均值")
    logger.info(f"  - _std: 所有epoch的标准差")
    logger.info(f"  - _min: 所有epoch的最小值")
    logger.info(f"  - _max: 所有epoch的最大值")
    logger.info(f"  - _median: 所有epoch的中位数")
    logger.info(f"  - _count: 有效epoch的数量")
    
    try:
        # 运行实验
        logger.info("\n开始运行实验...")
        result = run_experiment(**experiment_params)
        
        logger.info("\n" + "=" * 60)
        logger.info("实验完成!")
        logger.info("=" * 60)
        logger.info(f"状态: {result['status']}")
        logger.info(f"输出目录: {result['output_dir']}")
        logger.info(f"运行时间: {result['duration']:.2f} 秒")
        logger.info(f"\n实验摘要:")
        logger.info(result['summary'])
        
        # 显示结果文件
        output_dir = experiment_params['output_dir']
        if os.path.exists(output_dir):
            logger.info(f"\n生成的结果文件:")
            for file in os.listdir(output_dir):
                file_path = os.path.join(output_dir, file)
                if os.path.isfile(file_path):
                    size = os.path.getsize(file_path)
                    logger.info(f"  {file} ({size} bytes)")
        
    except Exception as e:
        logger.error(f"\n实验运行失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 