#!/usr/bin/env python3
"""
性别分类实验验证脚本

这个脚本用于验证性别分类实验的正确性和完整性：
1. 检查输出文件的完整性
2. 验证模型性能指标的合理性
3. 检查特征重要性分析
4. 验证混淆矩阵和分类报告
5. 与机器学习最佳实践对比
"""

import os
import sys
import pandas as pd
import numpy as np
import json
import logging
from datetime import datetime
from typing import Dict, List, Optional

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_mill.experiment_engine import run_experiment
from feature_mill.experiment_result_manager import ExperimentResultManager

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def validate_classification_results(output_dir: str) -> dict:
    """
    验证分类实验结果
    
    Args:
        output_dir: 实验结果输出目录
    
    Returns:
        dict: 验证结果
    """
    validation_results = {
        'model_comparison_valid': False,
        'individual_models_valid': False,
        'feature_importance_valid': False,
        'visualization_valid': False,
        'overall_valid': False,
        'issues': []
    }
    
    try:
        # 1. 检查模型比较文件
        comparison_file = os.path.join(output_dir, 'model_comparison.csv')
        if os.path.exists(comparison_file):
            comparison_df = pd.read_csv(comparison_file)
            logger.info(f"模型比较文件包含 {len(comparison_df)} 个模型")
            
            if len(comparison_df) > 0:
                # 检查性能指标是否合理
                if 'accuracy' in comparison_df.columns:
                    accuracies = comparison_df['accuracy'].dropna()
                    if len(accuracies) > 0:
                        # 检查准确率是否在合理范围内
                        if accuracies.max() <= 1.0 and accuracies.min() >= 0.0:
                            validation_results['model_comparison_valid'] = True
                        else:
                            validation_results['issues'].append("模型准确率超出合理范围")
                    else:
                        validation_results['issues'].append("模型比较数据为空")
                else:
                    validation_results['issues'].append("模型比较文件缺少accuracy列")
            else:
                validation_results['issues'].append("模型比较文件为空")
        else:
            validation_results['issues'].append("模型比较文件不存在")
        
        # 2. 检查各个模型的单独结果文件
        model_names = ['Logistic_Regression', 'Random_Forest', 'Gradient_Boosting', 'SVM']
        required_files = [
            'classification_report_{}.csv',
            'confusion_matrix_{}.csv'
        ]
        
        valid_model_files = 0
        for model_name in model_names:
            model_valid = True
            for file_pattern in required_files:
                file_path = os.path.join(output_dir, file_pattern.format(model_name))
                if os.path.exists(file_path):
                    try:
                        model_df = pd.read_csv(file_path)
                        if len(model_df) > 0:
                            logger.info(f"模型文件 {file_pattern.format(model_name)} 包含 {len(model_df)} 行数据")
                        else:
                            validation_results['issues'].append(f"模型文件 {file_pattern.format(model_name)} 为空")
                            model_valid = False
                    except Exception as e:
                        validation_results['issues'].append(f"读取模型文件 {file_pattern.format(model_name)} 失败: {str(e)}")
                        model_valid = False
                else:
                    validation_results['issues'].append(f"模型文件 {file_pattern.format(model_name)} 不存在")
                    model_valid = False
            
            if model_valid:
                valid_model_files += 1
        
        # 至少应该有3个模型文件有效
        if valid_model_files >= 3:
            validation_results['individual_models_valid'] = True
        else:
            validation_results['issues'].append(f"有效的模型文件数量不足: {valid_model_files}/4")
        
        # 3. 检查特征重要性分析
        importance_files = [
            'feature_importance_Logistic_Regression.csv',
            'feature_importance_Random_Forest.csv',
            'feature_importance_Gradient_Boosting.csv'
        ]
        
        valid_importance_files = 0
        for importance_file in importance_files:
            file_path = os.path.join(output_dir, importance_file)
            if os.path.exists(file_path):
                try:
                    importance_df = pd.read_csv(file_path)
                    if len(importance_df) > 0 and 'importance' in importance_df.columns:
                        valid_importance_files += 1
                        logger.info(f"特征重要性文件 {importance_file} 包含 {len(importance_df)} 个特征")
                    else:
                        validation_results['issues'].append(f"特征重要性文件 {importance_file} 数据无效")
                except Exception as e:
                    validation_results['issues'].append(f"读取特征重要性文件 {importance_file} 失败: {str(e)}")
            else:
                validation_results['issues'].append(f"特征重要性文件 {importance_file} 不存在")
        
        # 至少应该有2个特征重要性文件有效
        if valid_importance_files >= 2:
            validation_results['feature_importance_valid'] = True
        else:
            validation_results['issues'].append(f"有效的特征重要性文件数量不足: {valid_importance_files}/3")
        
        # 4. 检查可视化文件
        visualization_files = [
            'model_comparison_plots.png',
            'confusion_matrices.png',
            'feature_importance_classification.png'
        ]
        
        valid_visualization_files = 0
        for viz_file in visualization_files:
            file_path = os.path.join(output_dir, viz_file)
            if os.path.exists(file_path):
                # 检查文件大小是否合理（至少1KB）
                file_size = os.path.getsize(file_path)
                if file_size > 1024:
                    valid_visualization_files += 1
                    logger.info(f"可视化文件 {viz_file} 大小: {file_size} bytes")
                else:
                    validation_results['issues'].append(f"可视化文件 {viz_file} 大小异常: {file_size} bytes")
            else:
                validation_results['issues'].append(f"可视化文件 {viz_file} 不存在")
        
        # 至少应该有2个可视化文件有效
        if valid_visualization_files >= 2:
            validation_results['visualization_valid'] = True
        else:
            validation_results['issues'].append(f"有效的可视化文件数量不足: {valid_visualization_files}/3")
        
        # 5. 总体验证
        if (validation_results['model_comparison_valid'] and 
            validation_results['individual_models_valid'] and
            validation_results['feature_importance_valid'] and
            validation_results['visualization_valid']):
            validation_results['overall_valid'] = True
        
    except Exception as e:
        validation_results['issues'].append(f"验证过程中出现错误: {str(e)}")
        logger.error(f"验证过程中出现错误: {e}")
    
    return validation_results


def compare_with_ml_best_practices(results_dir: str) -> dict:
    """
    与机器学习最佳实践进行对比
    
    Args:
        results_dir: 实验结果目录
    
    Returns:
        dict: 对比结果
    """
    comparison_results = {
        'model_diversity': False,
        'performance_reasonable': False,
        'feature_importance_consistent': False,
        'cross_validation_used': False,
        'overall_best_practices': False,
        'findings': []
    }
    
    try:
        # 1. 检查模型多样性
        comparison_file = os.path.join(results_dir, 'model_comparison.csv')
        if os.path.exists(comparison_file):
            comparison_df = pd.read_csv(comparison_file)
            if len(comparison_df) >= 3:  # 至少应该有3种不同类型的模型
                comparison_results['model_diversity'] = True
                comparison_results['findings'].append(f"使用了 {len(comparison_df)} 种不同类型的模型，模型多样性良好")
        
        # 2. 检查性能是否合理
        if os.path.exists(comparison_file):
            comparison_df = pd.read_csv(comparison_file)
            if 'accuracy' in comparison_df.columns:
                accuracies = comparison_df['accuracy'].dropna()
                if len(accuracies) > 0:
                    # 检查准确率是否在合理范围内（对于性别分类，通常0.6-0.9是合理的）
                    if accuracies.max() <= 0.95 and accuracies.min() >= 0.5:
                        comparison_results['performance_reasonable'] = True
                        comparison_results['findings'].append(f"性别分类模型准确率在合理范围内: {accuracies.min():.3f}-{accuracies.max():.3f}")
                    else:
                        comparison_results['findings'].append(f"性别分类模型准确率可能异常: {accuracies.min():.3f}-{accuracies.max():.3f}")
        
        # 3. 检查特征重要性一致性
        importance_files = [
            'feature_importance_Random_Forest.csv',
            'feature_importance_Gradient_Boosting.csv'
        ]
        
        importance_consistency = True
        for importance_file in importance_files:
            file_path = os.path.join(results_dir, importance_file)
            if os.path.exists(file_path):
                try:
                    importance_df = pd.read_csv(file_path)
                    if len(importance_df) > 0 and 'importance' in importance_df.columns:
                        # 检查是否有非零重要性特征
                        non_zero_importance = importance_df[importance_df['importance'] > 0]
                        if len(non_zero_importance) > 0:
                            comparison_results['findings'].append(f"{importance_file} 包含 {len(non_zero_importance)} 个非零重要性特征")
                        else:
                            importance_consistency = False
                            comparison_results['findings'].append(f"{importance_file} 没有非零重要性特征")
                except Exception as e:
                    importance_consistency = False
                    comparison_results['findings'].append(f"检查特征重要性文件失败: {str(e)}")
        
        if importance_consistency:
            comparison_results['feature_importance_consistent'] = True
        
        # 4. 检查是否使用了交叉验证
        # 这通常可以从模型比较文件中的标准差列推断
        if os.path.exists(comparison_file):
            comparison_df = pd.read_csv(comparison_file)
            if 'std_accuracy' in comparison_df.columns or 'cv_score' in comparison_df.columns:
                comparison_results['cross_validation_used'] = True
                comparison_results['findings'].append("使用了交叉验证评估模型性能")
        
        # 5. 总体最佳实践评估
        if (comparison_results['model_diversity'] and 
            comparison_results['performance_reasonable'] and
            comparison_results['feature_importance_consistent'] and
            comparison_results['cross_validation_used']):
            comparison_results['overall_best_practices'] = True
            comparison_results['findings'].append("总体符合机器学习最佳实践")
        
    except Exception as e:
        comparison_results['findings'].append(f"对比过程中出现错误: {str(e)}")
        logger.error(f"对比过程中出现错误: {e}")
    
    return comparison_results


def analyze_classification_performance(results_dir: str) -> dict:
    """
    分析分类性能
    
    Args:
        results_dir: 实验结果目录
    
    Returns:
        dict: 性能分析结果
    """
    performance_analysis = {
        'best_model': None,
        'model_rankings': [],
        'performance_insights': [],
        'findings': []
    }
    
    try:
        # 读取模型比较结果
        comparison_file = os.path.join(results_dir, 'model_comparison.csv')
        if os.path.exists(comparison_file):
            comparison_df = pd.read_csv(comparison_file)
            
            # 找出最佳模型
            if 'accuracy' in comparison_df.columns:
                best_idx = comparison_df['accuracy'].idxmax()
                best_model = comparison_df.loc[best_idx, 'model']
                performance_analysis['best_model'] = best_model
                performance_analysis['findings'].append(f"最佳模型: {best_model}")
            
            # 模型排名
            if 'accuracy' in comparison_df.columns:
                ranked_models = comparison_df.sort_values('accuracy', ascending=False)
                performance_analysis['model_rankings'] = ranked_models['model'].tolist()
                performance_analysis['findings'].append(f"模型排名: {', '.join(ranked_models['model'].tolist())}")
                
                # 性能洞察
                accuracy_range = ranked_models['accuracy'].max() - ranked_models['accuracy'].min()
                if accuracy_range < 0.1:
                    performance_analysis['performance_insights'].append("模型性能差异较小，可能需要更多特征工程")
                elif accuracy_range > 0.3:
                    performance_analysis['performance_insights'].append("模型性能差异较大，某些模型可能不适合该数据集")
                else:
                    performance_analysis['performance_insights'].append("模型性能差异适中")
        
        # 分析特征重要性
        importance_files = [
            'feature_importance_Random_Forest.csv',
            'feature_importance_Gradient_Boosting.csv'
        ]
        
        for importance_file in importance_files:
            file_path = os.path.join(results_dir, importance_file)
            if os.path.exists(file_path):
                try:
                    importance_df = pd.read_csv(file_path)
                    if len(importance_df) > 0 and 'importance' in importance_df.columns:
                        top_features = importance_df.nlargest(5, 'importance')
                        model_name = importance_file.replace('feature_importance_', '').replace('.csv', '')
                        performance_analysis['findings'].append(f"{model_name} 前5重要特征: {', '.join(top_features['feature'].tolist())}")
                except Exception as e:
                    performance_analysis['findings'].append(f"分析特征重要性文件失败: {str(e)}")
        
    except Exception as e:
        performance_analysis['findings'].append(f"性能分析过程中出现错误: {str(e)}")
        logger.error(f"性能分析过程中出现错误: {e}")
    
    return performance_analysis


def run_classification_test():
    """运行分类实验测试"""
    print("🔬 开始性别分类实验验证测试")
    print("=" * 60)
    
    # 实验参数
    dataset_id = 1  # minimal_harvard数据集
    feature_set_id = 1
    output_dir = "data/experiments/classification_validation"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 运行分类实验
        print("📊 步骤1: 运行性别分类实验...")
        start_time = datetime.now()
        
        result = run_experiment(
            experiment_type='classification',
            dataset_id=dataset_id,
            feature_set_id=feature_set_id,
            output_dir=output_dir,
            extra_args={
                'target_var': 'sex',  # 改为性别分类
                'test_size': 0.2,
                'random_state': 42,
                'n_splits': 5,
                'plot_results': True,
                'plot_feature_importance': True
            }
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ 性别分类实验完成，耗时: {duration:.2f}秒")
        print(f"📁 结果保存在: {output_dir}")
        
        # 2. 验证实验结果
        print("\n🔍 步骤2: 验证实验结果...")
        validation_results = validate_classification_results(output_dir)
        
        print("验证结果:")
        for key, value in validation_results.items():
            if key != 'issues':
                status = "✅ 通过" if value else "❌ 失败"
                print(f"  {key}: {status}")
        
        if validation_results['issues']:
            print("\n⚠️ 发现的问题:")
            for issue in validation_results['issues']:
                print(f"  - {issue}")
        
        # 3. 与机器学习最佳实践对比
        print("\n📚 步骤3: 与机器学习最佳实践对比...")
        comparison_results = compare_with_ml_best_practices(output_dir)
        
        print("最佳实践对比结果:")
        for key, value in comparison_results.items():
            if key != 'findings':
                status = "✅ 符合" if value else "❌ 不符合"
                print(f"  {key}: {status}")
        
        if comparison_results['findings']:
            print("\n📖 发现:")
            for finding in comparison_results['findings']:
                print(f"  - {finding}")
        
        # 4. 分析分类性能
        print("\n📈 步骤4: 分析分类性能...")
        performance_analysis = analyze_classification_performance(output_dir)
        
        if performance_analysis['findings']:
            print("性能分析结果:")
            for finding in performance_analysis['findings']:
                print(f"  - {finding}")
        
        # 5. 生成测试报告
        print("\n📋 步骤5: 生成测试报告...")
        generate_test_report(validation_results, comparison_results, performance_analysis, result, output_dir)
        
        # 6. 总体评估
        print("\n🎯 总体评估:")
        if validation_results['overall_valid'] and comparison_results['overall_best_practices']:
            print("✅ 性别分类实验验证成功！符合机器学习最佳实践。")
        else:
            print("⚠️ 性别分类实验验证部分成功，需要进一步检查。")
        
        return {
            'success': validation_results['overall_valid'] and comparison_results['overall_best_practices'],
            'validation_results': validation_results,
            'comparison_results': comparison_results,
            'performance_analysis': performance_analysis,
            'experiment_result': result
        }
        
    except Exception as e:
        print(f"❌ 性别分类实验测试失败: {e}")
        logger.error(f"性别分类实验测试失败: {e}")
        return {'success': False, 'error': str(e)}


def generate_test_report(validation_results, comparison_results, performance_analysis, experiment_result, output_dir):
    """生成测试报告"""
    report_file = os.path.join(output_dir, 'validation_report.txt')
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("性别分类实验验证报告\n")
        f.write("=" * 50 + "\n\n")
        
        f.write("1. 实验基本信息\n")
        f.write("-" * 20 + "\n")
        f.write(f"实验类型: classification\n")
        f.write(f"数据集ID: 1 \n")
        f.write(f"特征集ID: 1\n")
        f.write(f"目标变量: sex\n")
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
        
        f.write("\n3. 最佳实践对比结果\n")
        f.write("-" * 20 + "\n")
        for key, value in comparison_results.items():
            if key != 'findings':
                status = "符合" if value else "不符合"
                f.write(f"{key}: {status}\n")
        
        if comparison_results['findings']:
            f.write("\n发现:\n")
            for finding in comparison_results['findings']:
                f.write(f"- {finding}\n")
        
        f.write("\n4. 性能分析结果\n")
        f.write("-" * 20 + "\n")
        if performance_analysis['findings']:
            for finding in performance_analysis['findings']:
                f.write(f"- {finding}\n")
        
        f.write("\n5. 总体评估\n")
        f.write("-" * 20 + "\n")
        if validation_results['overall_valid'] and comparison_results['overall_best_practices']:
            f.write("✅ 性别分类实验验证成功！符合机器学习最佳实践。\n")
        else:
            f.write("⚠️ 性别分类实验验证部分成功，需要进一步检查。\n")
    
    print(f"📄 验证报告已保存到: {report_file}")


if __name__ == "__main__":
    run_classification_test() 