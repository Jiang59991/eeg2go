#!/usr/bin/env python3
"""
分类实验简单验证测试

这个脚本用于验证分类实验的结果是否正确保存到数据库
"""

import os
import sys
import pandas as pd
from datetime import datetime
import logging

# 添加项目根目录到路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from feature_mill.experiment_engine import run_experiment
from feature_mill.experiment_result_manager import ExperimentResultManager
from feature_mill.test_classification_validation import validate_classification_results

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


def test_classification_validation():
    """测试分类实验验证"""
    print("🔬 开始分类实验验证测试")
    print("=" * 60)
    
    # 实验参数
    dataset_id = 3  # minimal_harvard数据集
    feature_set_id = 1
    output_dir = "data/experiments/classification_validation"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    try:
        # 1. 运行分类实验
        print("📊 步骤1: 运行分类实验...")
        start_time = datetime.now()
        
        result = run_experiment(
            experiment_type='classification',
            dataset_id=dataset_id,
            feature_set_id=feature_set_id,
            output_dir=output_dir,
            extra_args={
                'target_var': 'age_group',
                'age_threshold': 65,
                'test_size': 0.2,
                'random_state': 42,
                'n_splits': 5,
                'plot_results': True,
                'plot_feature_importance': True
            }
        )
        
        end_time = datetime.now()
        duration = (end_time - start_time).total_seconds()
        
        print(f"✅ 分类实验完成，耗时: {duration:.2f}秒")
        print(f"📁 结果保存在: {output_dir}")
        print(f"🔢 实验ID: {result.get('experiment_result_id', 'N/A')}")
        
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
        
        # 3. 检查数据库中的结果
        print("\n🗄️ 步骤3: 检查数据库中的结果...")
        result_manager = ExperimentResultManager()
        
        # 获取实验统计信息
        stats = result_manager.get_experiment_statistics()
        print(f"📊 数据库统计:")
        print(f"  总实验数: {stats.get('total_experiments', 0)}")
        print(f"  特征级别结果数: {stats.get('total_feature_results', 0)}")
        
        # 获取分类重要性历史
        importance_history = result_manager.get_feature_importance_history(
            target_variable='age_group',
            result_type='classification_importance'
        )
        print(f"  分类重要性记录: {len(importance_history)} 条")
        
        if len(importance_history) > 0:
            print(f"  示例记录:")
            print(f"    特征: {importance_history.iloc[0]['feature_name']}")
            print(f"    重要性分数: {importance_history.iloc[0]['importance_score']:.4f}")
            print(f"    排名: {importance_history.iloc[0]['rank_position']}")
        
        # 4. 总体评估
        print("\n🎯 总体评估:")
        if validation_results['overall_valid']:
            print("✅ 分类实验验证成功！")
        else:
            print("⚠️ 分类实验验证部分成功，需要进一步检查。")
        
        return {
            'success': validation_results['overall_valid'],
            'validation_results': validation_results,
            'experiment_result': result,
            'database_stats': stats
        }
        
    except Exception as e:
        print(f"❌ 分类实验测试失败: {e}")
        logger.error(f"分类实验测试失败: {e}")
        return {'success': False, 'error': str(e)}


if __name__ == "__main__":
    test_classification_validation() 