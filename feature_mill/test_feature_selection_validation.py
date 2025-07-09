#!/usr/bin/env python3
"""
特征选择实验验证脚本

这个脚本用于：
1. 运行特征选择实验
2. 验证实验结果的正确性
3. 与已知的EEG研究结果进行对比
4. 检查特征选择方法的合理性
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime
import logging

# 添加项目路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_mill.experiment_engine import run_experiment

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_feature_selection_results(output_dir: str) -> dict:
    """
    验证特征选择实验结果
    
    Args:
        output_dir: 实验结果输出目录
    
    Returns:
        dict: 验证结果
    """
    validation_results = {
        'selection_summary_valid': False,
        'methods_comparison_valid': False,
        'individual_methods_valid': False,
        'feature_importance_valid': False,
        'overall_valid': False,
        'issues': []
    }
    
    try:
        # 1. 检查特征选择汇总文件
        summary_file = os.path.join(output_dir, 'feature_selection_summary.csv')
        if os.path.exists(summary_file):
            summary_df = pd.read_csv(summary_file)
            logger.info(f"特征选择汇总文件包含 {len(summary_df)} 个特征")
            
            if len(summary_df) > 0:
                # 检查方法使用次数是否合理
                methods_using = summary_df['methods_using'].dropna()
                n_methods = summary_df['n_methods'].dropna()
                
                if len(methods_using) > 0 and len(n_methods) > 0:
                    # 检查方法使用次数是否在合理范围内
                    if n_methods.max() <= 7 and n_methods.min() >= 0:  # 最多7种方法
                        validation_results['selection_summary_valid'] = True
                    else:
                        validation_results['issues'].append("方法使用次数超出合理范围")
                else:
                    validation_results['issues'].append("特征选择汇总数据为空")
            else:
                validation_results['issues'].append("特征选择汇总文件为空")
        else:
            validation_results['issues'].append("特征选择汇总文件不存在")
        
        # 2. 检查方法比较文件
        comparison_file = os.path.join(output_dir, 'selection_methods_comparison.csv')
        if os.path.exists(comparison_file):
            comparison_df = pd.read_csv(comparison_file)
            comparison_df.columns = [col.strip() for col in comparison_df.columns]  # 去除列名空格
            if len(comparison_df) > 0:
                if 'mean_cv_score' in comparison_df.columns:
                    cv_scores = comparison_df['mean_cv_score'].dropna()
                    if len(cv_scores) > 0:
                        if cv_scores.max() <= 1.0 and cv_scores.min() >= -10.0:
                            validation_results['methods_comparison_valid'] = True
                        else:
                            validation_results['issues'].append("交叉验证分数超出合理范围")
                    else:
                        validation_results['issues'].append("方法比较数据为空")
                else:
                    validation_results['issues'].append("方法比较文件缺少mean_cv_score列")
            else:
                validation_results['issues'].append("方法比较文件为空")
        else:
            validation_results['issues'].append("方法比较文件不存在")
        
        # 3. 检查各个方法的单独结果文件
        method_files = [
            'selection_variance.csv',
            'selection_correlation.csv',
            'selection_univariate_f.csv',
            'selection_mutual_info.csv',
            'selection_lasso.csv',
            'selection_rfe.csv',
            'selection_pca.csv'
        ]
        
        valid_method_files = 0
        for method_file in method_files:
            file_path = os.path.join(output_dir, method_file)
            if os.path.exists(file_path):
                try:
                    method_df = pd.read_csv(file_path)
                    if len(method_df) > 0:
                        valid_method_files += 1
                        logger.info(f"方法文件 {method_file} 包含 {len(method_df)} 个特征")
                    else:
                        validation_results['issues'].append(f"方法文件 {method_file} 为空")
                except Exception as e:
                    validation_results['issues'].append(f"读取方法文件 {method_file} 失败: {str(e)}")
            else:
                validation_results['issues'].append(f"方法文件 {method_file} 不存在")
        
        # 至少应该有4个方法文件有效
        if valid_method_files >= 4:
            validation_results['individual_methods_valid'] = True
        else:
            validation_results['issues'].append(f"有效的方法文件数量不足: {valid_method_files}/7")
        
        # 4. 检查特征重要性分析
        importance_file = os.path.join(output_dir, 'feature_importance_analysis.png')
        if os.path.exists(importance_file):
            validation_results['feature_importance_valid'] = True
        else:
            validation_results['issues'].append("特征重要性分析图不存在")
        
        # 5. 总体验证
        if (validation_results['selection_summary_valid'] and 
            validation_results['methods_comparison_valid'] and
            validation_results['individual_methods_valid'] and
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
        'alpha_power_selection': False,
        'entropy_selection': False,
        'peak_frequency_selection': False,
        'method_diversity': False,
        'overall_consistency': False,
        'findings': []
    }
    
    try:
        # 读取特征选择汇总结果
        summary_file = os.path.join(results_dir, 'feature_selection_summary.csv')
        if not os.path.exists(summary_file):
            comparison_results['findings'].append("无法找到特征选择汇总文件")
            return comparison_results
        
        summary_df = pd.read_csv(summary_file)
        
        # 1. 检查α波功率特征的选择情况
        alpha_features = summary_df[summary_df['feature'].str.contains('alpha|bp_alpha', case=False)]
        if len(alpha_features) > 0:
            logger.info(f"发现 {len(alpha_features)} 个α波相关特征被选择")
            comparison_results['findings'].append(f"发现 {len(alpha_features)} 个α波相关特征被选择")
            
            # 检查α波特征是否被多种方法选择
            high_selection_alpha = alpha_features[alpha_features['n_methods'] >= 3]
            if len(high_selection_alpha) > 0:
                comparison_results['alpha_power_selection'] = True
                comparison_results['findings'].append("α波功率特征被多种方法选择，符合EEG研究文献")
        
        # 2. 检查熵特征的选择情况
        entropy_features = summary_df[summary_df['feature'].str.contains('entropy', case=False)]
        if len(entropy_features) > 0:
            logger.info(f"发现 {len(entropy_features)} 个熵相关特征被选择")
            comparison_results['findings'].append(f"发现 {len(entropy_features)} 个熵相关特征被选择")
            
            # 检查熵特征是否被多种方法选择
            high_selection_entropy = entropy_features[entropy_features['n_methods'] >= 3]
            if len(high_selection_entropy) > 0:
                comparison_results['entropy_selection'] = True
                comparison_results['findings'].append("熵特征被多种方法选择，符合EEG复杂度分析文献")
        
        # 3. 检查峰值频率特征的选择情况
        peak_features = summary_df[summary_df['feature'].str.contains('peak', case=False)]
        if len(peak_features) > 0:
            logger.info(f"发现 {len(peak_features)} 个峰值频率相关特征被选择")
            comparison_results['findings'].append(f"发现 {len(peak_features)} 个峰值频率相关特征被选择")
            
            if len(peak_features) > 0:
                comparison_results['peak_frequency_selection'] = True
                comparison_results['findings'].append("峰值频率特征被选择，符合EEG频谱分析文献")
        
        # 4. 检查方法多样性
        methods_comparison_file = os.path.join(results_dir, 'selection_methods_comparison.csv')
        if os.path.exists(methods_comparison_file):
            methods_df = pd.read_csv(methods_comparison_file)
            if len(methods_df) >= 5:  # 至少应该有5种方法
                comparison_results['method_diversity'] = True
                comparison_results['findings'].append(f"使用了 {len(methods_df)} 种特征选择方法，方法多样性良好")
        
        # 5. 总体一致性评估
        if (comparison_results['alpha_power_selection'] and 
            comparison_results['entropy_selection'] and
            comparison_results['peak_frequency_selection'] and
            comparison_results['method_diversity']):
            comparison_results['overall_consistency'] = True
            comparison_results['findings'].append("总体结果与EEG研究文献高度一致")
        
    except Exception as e:
        comparison_results['findings'].append(f"对比过程中出现错误: {str(e)}")
        logger.error(f"对比过程中出现错误: {e}")
    
    return comparison_results


def analyze_selection_methods_performance(results_dir: str) -> dict:
    """
    分析特征选择方法的性能
    
    Args:
        results_dir: 实验结果目录
    
    Returns:
        dict: 性能分析结果
    """
    performance_analysis = {
        'best_method': None,
        'method_rankings': [],
        'feature_overlap': {},
        'findings': []
    }
    
    try:
        # 读取方法比较结果
        comparison_file = os.path.join(results_dir, 'selection_methods_comparison.csv')
        if os.path.exists(comparison_file):
            comparison_df = pd.read_csv(comparison_file)
            
            # 找出最佳方法
            if 'mean_cv_score' in comparison_df.columns:
                best_idx = comparison_df['mean_cv_score'].idxmax()
                best_method = best_idx  # 索引就是方法名称
                performance_analysis['best_method'] = best_method
                performance_analysis['findings'].append(f"最佳方法: {best_method}")
            
            # 方法排名
            if 'mean_cv_score' in comparison_df.columns:
                ranked_methods = comparison_df.sort_values('mean_cv_score', ascending=False)
                performance_analysis['method_rankings'] = ranked_methods.index.tolist()
                performance_analysis['findings'].append(f"方法排名: {', '.join(ranked_methods.index.tolist())}")
        
        # 分析特征重叠
        summary_file = os.path.join(results_dir, 'feature_selection_summary.csv')
        if os.path.exists(summary_file):
            summary_df = pd.read_csv(summary_file)
            
            # 找出被多种方法选择的特征
            high_selection_features = summary_df[summary_df['n_methods'] >= 3]
            if len(high_selection_features) > 0:
                performance_analysis['feature_overlap'] = {
                    'high_selection_features': high_selection_features['feature'].tolist(),
                    'count': len(high_selection_features)
                }
                performance_analysis['findings'].append(f"被3种以上方法选择的特征: {len(high_selection_features)} 个")
        
    except Exception as e:
        performance_analysis['findings'].append(f"性能分析过程中出现错误: {str(e)}")
        logger.error(f"性能分析过程中出现错误: {e}")
    
    return performance_analysis


def run_feature_selection_test():
    """运行特征选择实验测试"""
    print("🔬 开始特征选择实验验证测试")
    print("=" * 60)
    
    # 实验参数
    dataset_id = 1  # minimal_harvard数据集
    feature_set_id = 1
    output_dir = "data/experiments/feature_selection_validation"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 运行特征选择实验
        print("📊 步骤1: 运行特征选择实验...")
        start_time = datetime.now()
        
        result = run_experiment(
            experiment_type='feature_selection',
            dataset_id=dataset_id,
            feature_set_id=feature_set_id,
            output_dir=output_dir,
            extra_args={
                'target_var': 'age',
                'n_features': 20,
                'variance_threshold': 0.01,
                'correlation_threshold': 0.95,
                'plot_selection_results': True,
                'plot_feature_importance': True
            }
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ 特征选择实验完成，耗时: {duration:.2f}秒")
        print(f"📁 结果保存在: {output_dir}")
        
        # 2. 验证实验结果
        print("\n🔍 步骤2: 验证实验结果...")
        validation_results = validate_feature_selection_results(output_dir)
        
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
        
        # 4. 分析选择方法性能
        print("\n📈 步骤4: 分析特征选择方法性能...")
        performance_analysis = analyze_selection_methods_performance(output_dir)
        
        if performance_analysis['findings']:
            print("性能分析结果:")
            for finding in performance_analysis['findings']:
                print(f"  - {finding}")
        
        # 5. 生成测试报告
        print("\n📋 步骤5: 生成测试报告...")
        generate_test_report(validation_results, comparison_results, performance_analysis, result, output_dir)
        
        # 6. 总体评估
        print("\n🎯 总体评估:")
        if validation_results['overall_valid'] and comparison_results['overall_consistency']:
            print("✅ 特征选择实验验证成功！结果与EEG研究文献一致。")
        else:
            print("⚠️ 特征选择实验验证部分成功，需要进一步检查。")
        
        return {
            'success': validation_results['overall_valid'] and comparison_results['overall_consistency'],
            'validation_results': validation_results,
            'comparison_results': comparison_results,
            'performance_analysis': performance_analysis,
            'experiment_result': result
        }
        
    except Exception as e:
        print(f"❌ 特征选择实验测试失败: {e}")
        logger.error(f"特征选择实验测试失败: {e}")
        return {'success': False, 'error': str(e)}


def generate_test_report(validation_results, comparison_results, performance_analysis, experiment_result, output_dir):
    """生成测试报告"""
    report_file = os.path.join(output_dir, 'validation_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("特征选择实验验证报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 实验基本信息\n")
        f.write("-" * 20 + "\n")
        f.write(f"实验类型: feature_selection\n")
        f.write(f"数据集ID: 3 (minimal_harvard)\n")
        f.write(f"特征集ID: 1\n")
        f.write(f"目标变量: age\n")
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
        
        f.write("\n4. 性能分析结果\n")
        f.write("-" * 20 + "\n")
        if performance_analysis['best_method']:
            f.write(f"最佳方法: {performance_analysis['best_method']}\n")
        
        if performance_analysis['method_rankings']:
            f.write(f"方法排名: {', '.join(performance_analysis['method_rankings'])}\n")
        
        if performance_analysis['feature_overlap']:
            overlap_info = performance_analysis['feature_overlap']
            f.write(f"高选择特征数量: {overlap_info['count']}\n")
        
        if performance_analysis['findings']:
            f.write("\n性能分析发现:\n")
            for finding in performance_analysis['findings']:
                f.write(f"- {finding}\n")
        
        f.write("\n5. 总体评估\n")
        f.write("-" * 20 + "\n")
        if validation_results['overall_valid'] and comparison_results['overall_consistency']:
            f.write("✅ 特征选择实验验证成功！结果与EEG研究文献一致。\n")
        else:
            f.write("⚠️ 特征选择实验验证部分成功，需要进一步检查。\n")
    
    print(f"📄 测试报告已保存到: {report_file}")


if __name__ == "__main__":
    run_feature_selection_test() 