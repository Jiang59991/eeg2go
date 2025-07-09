"""
全面实验测试脚本

这个脚本测试所有类型的实验功能：
1. 相关性分析实验
2. 分类分析实验  
3. 特征选择实验
4. 特征统计实验

使用dataset 3 (minimal_harvard) 进行测试，确保所有功能正常运行。
"""

import os
import sys
import logging
import time
from datetime import datetime
from feature_mill.experiment_engine import run_experiment
from feature_mill.feature_experiment_query import FeatureExperimentQuery
from feature_mill.experiment_result_manager import ExperimentResultManager

# 设置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_correlation_experiment():
    """测试相关性分析实验"""
    print("=" * 60)
    print("🧪 测试相关性分析实验")
    print("=" * 60)
    
    try:
        result = run_experiment(
            experiment_type='correlation',
            dataset_id=1,  # minimal_harvard
            feature_set_id=1,
            output_dir='data/experiments/test_correlation',
            extra_args={
                'target_vars': ['age', 'sex'],
                'method': 'pearson',
                'min_corr': 0.1,
                'top_n': 10,
                'plot_corr_matrix': True,
                'plot_scatter': True,
                'save_detailed_results': True
            }
        )
        
        if result['status'] == 'success':
            print(f"✅ 相关性分析实验成功！")
            print(f"   实验ID: {result.get('experiment_result_id', 'N/A')}")
            print(f"   运行时间: {result['duration']:.2f} 秒")
            return True
        else:
            print(f"❌ 相关性分析实验失败: {result}")
            return False
            
    except Exception as e:
        print(f"❌ 相关性分析实验异常: {e}")
        return False


def test_classification_experiment():
    """测试分类分析实验"""
    print("\n" + "=" * 60)
    print("🧪 测试分类分析实验")
    print("=" * 60)
    
    try:
        result = run_experiment(
            experiment_type='classification',
            dataset_id=1,  # minimal_harvard
            feature_set_id=1,
            output_dir='data/experiments/test_classification',
            extra_args={
                'target_var': 'age_group',
                'age_threshold': 50,  # 降低阈值适应小数据集
                'test_size': 0.3,
                'n_splits': 3,  # 减少交叉验证折数
                'random_state': 42,
                'save_model': True,
                'plot_results': True
            }
        )
        
        if result['status'] == 'success':
            print(f"✅ 分类分析实验成功！")
            print(f"   实验ID: {result.get('experiment_result_id', 'N/A')}")
            print(f"   运行时间: {result['duration']:.2f} 秒")
            return True
        else:
            print(f"❌ 分类分析实验失败: {result}")
            return False
            
    except Exception as e:
        print(f"❌ 分类分析实验异常: {e}")
        return False


def test_feature_selection_experiment():
    """测试特征选择实验"""
    print("\n" + "=" * 60)
    print("🧪 测试特征选择实验")
    print("=" * 60)
    
    try:
        result = run_experiment(
            experiment_type='feature_selection',
            dataset_id=1,  # minimal_harvard
            feature_set_id=1,
            output_dir='data/experiments/test_feature_selection',
            extra_args={
                'target_var': 'age',
                'n_features': 10,  # 减少特征数量
                'variance_threshold': 0.01,
                'correlation_threshold': 0.95,
                'random_state': 42,
                'save_results': True,
                'plot_results': True
            }
        )
        
        if result['status'] == 'success':
            print(f"✅ 特征选择实验成功！")
            print(f"   实验ID: {result.get('experiment_result_id', 'N/A')}")
            print(f"   运行时间: {result['duration']:.2f} 秒")
            return True
        else:
            print(f"❌ 特征选择实验失败: {result}")
            return False
            
    except Exception as e:
        print(f"❌ 特征选择实验异常: {e}")
        return False


def test_feature_statistics_experiment():
    """测试特征统计实验"""
    print("\n" + "=" * 60)
    print("🧪 测试特征统计实验")
    print("=" * 60)
    
    try:
        result = run_experiment(
            experiment_type='feature_statistics',
            dataset_id=1,  # minimal_harvard
            feature_set_id=1,
            output_dir='data/experiments/test_feature_statistics',
            extra_args={
                'outlier_method': 'iqr',
                'outlier_threshold': 1.5,
                'top_n_features': 10,
                'save_results': True,
                'plot_results': True,
                'generate_report': True
            }
        )
        
        if result['status'] == 'success':
            print(f"✅ 特征统计实验成功！")
            print(f"   实验ID: {result.get('experiment_result_id', 'N/A')}")
            print(f"   运行时间: {result['duration']:.2f} 秒")
            return True
        else:
            print(f"❌ 特征统计实验失败: {result}")
            return False
            
    except Exception as e:
        print(f"❌ 特征统计实验异常: {e}")
        return False


def test_query_functions():
    """测试查询功能"""
    print("\n" + "=" * 60)
    print("🔍 测试查询功能")
    print("=" * 60)
    
    try:
        query_tool = FeatureExperimentQuery()
        result_manager = ExperimentResultManager()
        
        # 1. 测试实验统计信息
        print("1. 测试实验统计信息...")
        stats = result_manager.get_experiment_statistics()
        if stats:
            print(f"   ✅ 总实验次数: {stats.get('total_experiments', 0)}")
            print(f"   ✅ 特征结果总数: {stats.get('total_feature_results', 0)}")
            if 'experiments_by_type' in stats:
                print("   ✅ 按类型统计:")
                for exp_type in stats['experiments_by_type']:
                    print(f"      {exp_type['experiment_type']}: {exp_type['count']} 次")
        else:
            print("   ❌ 无法获取实验统计信息")
        
        # 2. 测试特征相关性查询
        print("\n2. 测试特征相关性查询...")
        correlation_features = query_tool.search_features_by_correlation(
            target_variable='age',
            min_correlation=0.2,
            significant_only=True
        )
        if len(correlation_features) > 0:
            print(f"   ✅ 找到 {len(correlation_features)} 个相关特征")
            print(f"   ✅ 最相关特征: {correlation_features.iloc[0]['feature_name']} (r={correlation_features.iloc[0]['avg_correlation']:.3f})")
        else:
            print("   ⚠️  未找到显著相关的特征")
        
        # 3. 测试特征重要性查询
        print("\n3. 测试特征重要性查询...")
        important_features = query_tool.search_features_by_importance(
            target_variable='age',
            min_importance=0.01
        )
        if len(important_features) > 0:
            print(f"   ✅ 找到 {len(important_features)} 个重要特征")
            print(f"   ✅ 最重要特征: {important_features.iloc[0]['feature_name']} (重要性={important_features.iloc[0]['avg_importance']:.3f})")
        else:
            print("   ⚠️  未找到重要特征")
        
        # 4. 测试特定特征报告
        print("\n4. 测试特定特征报告...")
        # 查找一个实际存在的特征
        if len(correlation_features) > 0:
            test_feature = correlation_features.iloc[0]['feature_name']
            report = query_tool.get_feature_experiment_report(feature_name=test_feature)
            if 'error' not in report:
                print(f"   ✅ 成功生成特征 {test_feature} 的报告")
                if report['correlation_summaries'].get('age', {}).get('has_correlation_data', False):
                    print(f"   ✅ 该特征与年龄有相关性数据")
            else:
                print(f"   ❌ 生成特征报告失败: {report['error']}")
        else:
            print("   ⚠️  跳过特征报告测试（无可用特征）")
        
        return True
        
    except Exception as e:
        print(f"❌ 查询功能测试异常: {e}")
        return False


def check_dataset_status():
    """检查数据集状态"""
    print("=" * 60)
    print("📊 检查数据集状态")
    print("=" * 60)
    
    try:
        import sqlite3
        conn = sqlite3.connect('database/eeg2go.db')
        c = conn.cursor()
        
        # 检查数据集
        c.execute("SELECT id, name FROM datasets WHERE id = 3")
        dataset = c.fetchone()
        if dataset:
            print(f"✅ 数据集3存在: {dataset[1]}")
        else:
            print("❌ 数据集3不存在")
            return False
        
        # 检查记录数
        c.execute("SELECT COUNT(*) FROM recordings WHERE dataset_id = 3")
        recording_count = c.fetchone()[0]
        print(f"✅ 数据集3包含 {recording_count} 条记录")
        
        if recording_count == 0:
            print("❌ 数据集3没有记录，无法进行实验")
            return False
        
        # 检查特征集
        c.execute("SELECT id, name FROM feature_sets WHERE id = 1")
        feature_set = c.fetchone()
        if feature_set:
            print(f"✅ 特征集1存在: {feature_set[1]}")
        else:
            print("❌ 特征集1不存在")
            return False
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"❌ 检查数据集状态失败: {e}")
        return False


def main():
    """主函数"""
    print("🚀 EEG2Go 全面实验功能测试")
    print("=" * 60)
    print("这个脚本将测试所有类型的实验功能:")
    print("1. 相关性分析实验")
    print("2. 分类分析实验")
    print("3. 特征选择实验")
    print("4. 特征统计实验")
    print("5. 查询功能测试")
    print("=" * 60)
    
    start_time = time.time()
    test_results = {}
    
    try:
        # 检查数据集状态
        if not check_dataset_status():
            print("❌ 数据集状态检查失败，退出测试")
            return
        
        # 测试各种实验
        print("\n开始实验测试...")
        
        # 1. 相关性分析
        test_results['correlation'] = test_correlation_experiment()
        
        # 2. 分类分析
        test_results['classification'] = test_classification_experiment()
        
        # 3. 特征选择
        test_results['feature_selection'] = test_feature_selection_experiment()
        
        # 4. 特征统计
        test_results['feature_statistics'] = test_feature_statistics_experiment()
        
        # 5. 查询功能
        test_results['query_functions'] = test_query_functions()
        
        # 总结测试结果
        print("\n" + "=" * 60)
        print("📋 测试结果总结")
        print("=" * 60)
        
        total_tests = len(test_results)
        passed_tests = sum(test_results.values())
        
        for test_name, result in test_results.items():
            status = "✅ 通过" if result else "❌ 失败"
            print(f"{test_name}: {status}")
        
        print(f"\n总体结果: {passed_tests}/{total_tests} 项测试通过")
        
        if passed_tests == total_tests:
            print("🎉 所有测试通过！实验管理系统功能正常")
        else:
            print("⚠️  部分测试失败，请检查相关功能")
        
        total_time = time.time() - start_time
        print(f"\n总测试时间: {total_time:.2f} 秒")
        
    except Exception as e:
        print(f"❌ 测试过程中出现错误: {e}")
        logger.error(f"测试失败: {e}", exc_info=True)


if __name__ == "__main__":
    main() 