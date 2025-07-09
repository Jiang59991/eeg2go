#!/usr/bin/env python3
"""
特征统计实验验证脚本

这个脚本用于：
1. 运行特征统计实验
2. 验证实验结果的正确性
3. 与已知的EEG研究结果进行对比
4. 检查特征提取的合理性
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_mill.experiment_engine import run_experiment
from feature_mill.experiment_result_manager import ExperimentResultManager

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 设置matplotlib中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False


def validate_feature_statistics_results(output_dir: str) -> dict:
    """
    验证特征统计实验结果
    
    Args:
        output_dir: 实验结果输出目录
    
    Returns:
        dict: 验证结果
    """
    validation_results = {
        'basic_stats_valid': False,
        'distribution_analysis_valid': False,
        'outlier_analysis_valid': False,
        'feature_importance_valid': False,
        'overall_valid': False,
        'issues': []
    }
    
    try:
        # 1. 检查基本统计文件
        basic_stats_file = os.path.join(output_dir, 'feature_basic_statistics.csv')
        if os.path.exists(basic_stats_file):
            basic_stats_df = pd.read_csv(basic_stats_file)
            logger.info(f"基本统计文件包含 {len(basic_stats_df)} 个特征")
            
            # 验证基本统计的合理性
            if len(basic_stats_df) > 0:
                # 检查统计值是否在合理范围内
                mean_values = basic_stats_df['mean'].dropna()
                std_values = basic_stats_df['std'].dropna()
                
                if len(mean_values) > 0 and len(std_values) > 0:
                    # 检查是否有异常值
                    if not (mean_values.isin([np.inf, -np.inf]).any() or 
                           std_values.isin([np.inf, -np.inf]).any()):
                        validation_results['basic_stats_valid'] = True
                    else:
                        validation_results['issues'].append("基本统计中存在无穷值")
                else:
                    validation_results['issues'].append("基本统计数据为空")
            else:
                validation_results['issues'].append("基本统计文件为空")
        else:
            validation_results['issues'].append("基本统计文件不存在")
        
        # 2. 检查分布分析文件
        distribution_file = os.path.join(output_dir, 'feature_distribution_analysis.csv')
        if os.path.exists(distribution_file):
            distribution_df = pd.read_csv(distribution_file)
            logger.info(f"分布分析文件包含 {len(distribution_df)} 个特征")
            
            if len(distribution_df) > 0:
                # 检查分布类型是否合理
                dist_types = distribution_df['distribution_type'].value_counts()
                logger.info(f"分布类型分布: {dist_types.to_dict()}")
                
                # 验证偏度和峰度值
                skewness_values = distribution_df['skewness'].dropna()
                kurtosis_values = distribution_df['kurtosis'].dropna()
                
                if len(skewness_values) > 0 and len(kurtosis_values) > 0:
                    if not (skewness_values.isin([np.inf, -np.inf]).any() or 
                           kurtosis_values.isin([np.inf, -np.inf]).any()):
                        validation_results['distribution_analysis_valid'] = True
                    else:
                        validation_results['issues'].append("分布分析中存在无穷值")
                else:
                    validation_results['issues'].append("分布分析数据为空")
            else:
                validation_results['issues'].append("分布分析文件为空")
        else:
            validation_results['issues'].append("分布分析文件不存在")
        
        # 3. 检查异常值分析文件
        outlier_file = os.path.join(output_dir, 'feature_outlier_analysis.csv')
        if os.path.exists(outlier_file):
            outlier_df = pd.read_csv(outlier_file)
            logger.info(f"异常值分析文件包含 {len(outlier_df)} 个特征")
            
            if len(outlier_df) > 0:
                # 检查异常值比例是否合理
                outlier_percentages = outlier_df['outlier_percentage'].dropna()
                if len(outlier_percentages) > 0:
                    if outlier_percentages.max() <= 100 and outlier_percentages.min() >= 0:
                        validation_results['outlier_analysis_valid'] = True
                    else:
                        validation_results['issues'].append("异常值比例超出合理范围")
                else:
                    validation_results['issues'].append("异常值分析数据为空")
            else:
                validation_results['issues'].append("异常值分析文件为空")
        else:
            validation_results['issues'].append("异常值分析文件不存在")
        
        # 4. 检查特征重要性文件
        importance_file = os.path.join(output_dir, 'feature_importance_ranking.csv')
        if os.path.exists(importance_file):
            importance_df = pd.read_csv(importance_file)
            logger.info(f"特征重要性文件包含 {len(importance_df)} 个特征")
            
            if len(importance_df) > 0:
                # 检查重要性分数是否合理
                importance_scores = importance_df['importance_score'].dropna()
                if len(importance_scores) > 0:
                    if importance_scores.max() >= 0 and importance_scores.min() >= 0:
                        validation_results['feature_importance_valid'] = True
                    else:
                        validation_results['issues'].append("特征重要性分数包含负值")
                else:
                    validation_results['issues'].append("特征重要性数据为空")
            else:
                validation_results['issues'].append("特征重要性文件为空")
        else:
            validation_results['issues'].append("特征重要性文件不存在")
        
        # 5. 总体验证
        if (validation_results['basic_stats_valid'] and 
            validation_results['distribution_analysis_valid'] and
            validation_results['outlier_analysis_valid'] and
            validation_results['feature_importance_valid']):
            validation_results['overall_valid'] = True
        
    except Exception as e:
        validation_results['issues'].append(f"验证过程中出现错误: {str(e)}")
        logger.error(f"验证过程中出现错误: {e}")
    
    return validation_results


def compare_with_eeg_literature(results_dir: str) -> dict:
    """
    与EEG文献中的已知结果进行对比
    
    Args:
        results_dir: 实验结果目录
    
    Returns:
        dict: 对比结果
    """
    comparison_results = {
        'alpha_power_consistency': False,
        'entropy_consistency': False,
        'peak_frequency_consistency': False,
        'overall_consistency': False,
        'findings': []
    }
    
    try:
        # 读取特征重要性结果
        importance_file = os.path.join(results_dir, 'feature_importance_ranking.csv')
        if not os.path.exists(importance_file):
            comparison_results['findings'].append("无法找到特征重要性文件")
            return comparison_results
        
        importance_df = pd.read_csv(importance_file)
        
        # 1. 检查α波功率特征的一致性
        alpha_features = importance_df[importance_df['feature'].str.contains('alpha|bp_alpha', case=False)]
        if len(alpha_features) > 0:
            logger.info(f"发现 {len(alpha_features)} 个α波相关特征")
            comparison_results['findings'].append(f"发现 {len(alpha_features)} 个α波相关特征")
            
            # 检查α波特征是否在重要特征中
            top_alpha_features = alpha_features.head(10)
            if len(top_alpha_features) > 0:
                comparison_results['alpha_power_consistency'] = True
                comparison_results['findings'].append("α波功率特征在重要特征中表现良好，符合EEG研究文献")
        
        # 2. 检查熵特征的一致性
        entropy_features = importance_df[importance_df['feature'].str.contains('entropy', case=False)]
        if len(entropy_features) > 0:
            logger.info(f"发现 {len(entropy_features)} 个熵相关特征")
            comparison_results['findings'].append(f"发现 {len(entropy_features)} 个熵相关特征")
            
            # 检查熵特征是否在重要特征中
            top_entropy_features = entropy_features.head(10)
            if len(top_entropy_features) > 0:
                comparison_results['entropy_consistency'] = True
                comparison_results['findings'].append("熵特征在重要特征中表现良好，符合EEG复杂度分析文献")
        
        # 3. 检查峰值频率特征的一致性
        peak_features = importance_df[importance_df['feature'].str.contains('peak', case=False)]
        if len(peak_features) > 0:
            logger.info(f"发现 {len(peak_features)} 个峰值频率相关特征")
            comparison_results['findings'].append(f"发现 {len(peak_features)} 个峰值频率相关特征")
            
            if len(peak_features) > 0:
                comparison_results['peak_frequency_consistency'] = True
                comparison_results['findings'].append("峰值频率特征存在，符合EEG频谱分析文献")
        
        # 4. 总体一致性评估
        if (comparison_results['alpha_power_consistency'] and 
            comparison_results['entropy_consistency'] and
            comparison_results['peak_frequency_consistency']):
            comparison_results['overall_consistency'] = True
            comparison_results['findings'].append("总体结果与EEG研究文献高度一致")
        
    except Exception as e:
        comparison_results['findings'].append(f"对比过程中出现错误: {str(e)}")
        logger.error(f"对比过程中出现错误: {e}")
    
    return comparison_results


def run_feature_statistics_test():
    """运行特征统计实验测试"""
    print("🔬 开始特征统计实验验证测试")
    print("=" * 60)
    
    # 实验参数
    dataset_id = 1  # minimal_harvard数据集
    feature_set_id = 1
    output_dir = "data/experiments/feature_statistics_validation"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 运行特征统计实验
        print("📊 步骤1: 运行特征统计实验...")
        start_time = datetime.now()
        
        result = run_experiment(
            experiment_type='feature_statistics',
            dataset_id=dataset_id,
            feature_set_id=feature_set_id,
            output_dir=output_dir,
            extra_args={
                'outlier_method': 'iqr',
                'outlier_threshold': 1.5,
                'plot_distributions': True,
                'plot_correlation_heatmap': True,
                'plot_outliers': True,
                'top_n_features': 20
            }
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ 特征统计实验完成，耗时: {duration:.2f}秒")
        print(f"📁 结果保存在: {output_dir}")
        
        # 2. 验证实验结果
        print("\n🔍 步骤2: 验证实验结果...")
        validation_results = validate_feature_statistics_results(output_dir)
        
        print("验证结果:")
        for key, value in validation_results.items():
            if key != 'issues':
                status = "✅ 通过" if value else "❌ 失败"
                print(f"  {key}: {status}")
        
        if validation_results['issues']:
            print("\n⚠️ 发现的问题:")
            for issue in validation_results['issues']:
                print(f"  - {issue}")
        
        # 3. 与EEG文献对比
        print("\n📚 步骤3: 与EEG研究文献对比...")
        comparison_results = compare_with_eeg_literature(output_dir)
        
        print("文献对比结果:")
        for key, value in comparison_results.items():
            if key != 'findings':
                status = "✅ 一致" if value else "❌ 不一致"
                print(f"  {key}: {status}")
        
        if comparison_results['findings']:
            print("\n📖 发现:")
            for finding in comparison_results['findings']:
                print(f"  - {finding}")
        
        # 4. 生成测试报告
        print("\n📋 步骤4: 生成测试报告...")
        generate_test_report(validation_results, comparison_results, result, output_dir)
        
        # 5. 总体评估
        print("\n🎯 总体评估:")
        if validation_results['overall_valid'] and comparison_results['overall_consistency']:
            print("✅ 特征统计实验验证成功！结果与EEG研究文献一致。")
        else:
            print("⚠️ 特征统计实验验证部分成功，需要进一步检查。")
        
        return {
            'success': validation_results['overall_valid'] and comparison_results['overall_consistency'],
            'validation_results': validation_results,
            'comparison_results': comparison_results,
            'experiment_result': result
        }
        
    except Exception as e:
        print(f"❌ 特征统计实验测试失败: {e}")
        logger.error(f"特征统计实验测试失败: {e}")
        return {'success': False, 'error': str(e)}


def generate_test_report(validation_results, comparison_results, experiment_result, output_dir):
    """生成测试报告"""
    report_file = os.path.join(output_dir, 'validation_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("特征统计实验验证报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 实验基本信息\n")
        f.write("-" * 20 + "\n")
        f.write(f"实验类型: feature_statistics\n")
        f.write(f"数据集ID: 3 (minimal_harvard)\n")
        f.write(f"特征集ID: 1\n")
        f.write(f"实验ID: {experiment_result.get('experiment_result_id', 'N/A')}\n")
        f.write(f"运行时间: {experiment_result.get('duration', 'N/A')}秒\n\n")
        
        f.write("2. 验证结果\n")
        f.write("-" * 20 + "\n")
        for key, value in validation_results.items():
            if key != 'issues':
                status = "通过" if value else "失败"
                f.write(f"{key}: {status}\n")
        
        if validation_results['issues']:
            f.write("\n发现的问题:\n")
            for issue in validation_results['issues']:
                f.write(f"- {issue}\n")
        
        f.write("\n3. 文献对比结果\n")
        f.write("-" * 20 + "\n")
        for key, value in comparison_results.items():
            if key != 'findings':
                status = "一致" if value else "不一致"
                f.write(f"{key}: {status}\n")
        
        if comparison_results['findings']:
            f.write("\n发现:\n")
            for finding in comparison_results['findings']:
                f.write(f"- {finding}\n")
        
        f.write("\n4. 总体评估\n")
        f.write("-" * 20 + "\n")
        if validation_results['overall_valid'] and comparison_results['overall_consistency']:
            f.write("✅ 特征统计实验验证成功！结果与EEG研究文献一致。\n")
        else:
            f.write("⚠️ 特征统计实验验证部分成功，需要进一步检查。\n")
    
    print(f"📄 测试报告已保存到: {report_file}")


if __name__ == "__main__":
    run_feature_statistics_test() 