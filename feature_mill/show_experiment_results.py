#!/usr/bin/env python3
"""
展示数据库中实验结果的统计信息
"""

import sqlite3
import pandas as pd
from datetime import datetime

def show_experiment_results():
    """展示实验结果统计"""
    conn = sqlite3.connect('database/eeg2go.db')
    
    print("🔬 EEG实验结果管理系统 - 数据库统计")
    print("=" * 50)
    
    # 1. 实验记录统计
    print("\n📊 实验记录统计:")
    total_experiments = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM experiment_results", conn
    ).iloc[0]['count']
    print(f"  总实验记录数: {total_experiments}")
    
    # 按类型统计
    experiments_by_type = pd.read_sql_query("""
        SELECT experiment_type, COUNT(*) as count
        FROM experiment_results
        GROUP BY experiment_type
        ORDER BY count DESC
    """, conn)
    
    print("  按实验类型分布:")
    for _, row in experiments_by_type.iterrows():
        print(f"    {row['experiment_type']}: {row['count']} 次")
    
    # 2. 特征结果统计
    print("\n🎯 特征级别结果统计:")
    total_feature_results = pd.read_sql_query(
        "SELECT COUNT(*) as count FROM experiment_feature_results", conn
    ).iloc[0]['count']
    print(f"  总特征结果数: {total_feature_results}")
    
    # 按结果类型统计
    results_by_type = pd.read_sql_query("""
        SELECT result_type, COUNT(*) as count
        FROM experiment_feature_results
        GROUP BY result_type
        ORDER BY count DESC
    """, conn)
    
    print("  按结果类型分布:")
    for _, row in results_by_type.iterrows():
        print(f"    {row['result_type']}: {row['count']} 条")
    
    # 3. 最近实验记录
    print("\n⏰ 最近实验记录:")
    recent_experiments = pd.read_sql_query("""
        SELECT id, experiment_type, dataset_id, feature_set_id, 
               run_time, duration_seconds
        FROM experiment_results
        ORDER BY run_time DESC
        LIMIT 5
    """, conn)
    
    for _, row in recent_experiments.iterrows():
        print(f"  ID {row['id']}: {row['experiment_type']} "
              f"(数据集{row['dataset_id']}, 特征集{row['feature_set_id']}) "
              f"- {row['run_time']} ({row['duration_seconds']:.1f}s)")
    
    # 4. 相关性分析结果示例
    print("\n📈 相关性分析结果示例:")
    correlation_examples = pd.read_sql_query("""
        SELECT feature_name, target_variable, metric_value, significance_level, rank_position
        FROM experiment_feature_results 
        WHERE result_type = 'correlation' 
        AND metric_name = 'correlation_coefficient'
        AND significance_level != 'ns'
        ORDER BY ABS(metric_value) DESC
        LIMIT 5
    """, conn)
    
    for _, row in correlation_examples.iterrows():
        print(f"  {row['feature_name']} -> {row['target_variable']}: "
              f"{row['metric_value']:.3f} ({row['significance_level']}) "
              f"[排名: {row['rank_position']}]")
    
    # 5. 特征统计
    print("\n🔍 特征统计:")
    unique_features = pd.read_sql_query("""
        SELECT COUNT(DISTINCT feature_name) as count
        FROM experiment_feature_results
    """, conn).iloc[0]['count']
    print(f"  唯一特征数: {unique_features}")
    
    target_variables = pd.read_sql_query("""
        SELECT COUNT(DISTINCT target_variable) as count
        FROM experiment_feature_results
    """, conn).iloc[0]['count']
    print(f"  目标变量数: {target_variables}")
    
    # 6. 数据集信息
    print("\n📁 数据集信息:")
    datasets = pd.read_sql_query("""
        SELECT id, name FROM datasets
    """, conn)
    
    for _, row in datasets.iterrows():
        recordings_count = pd.read_sql_query("""
            SELECT COUNT(*) as count FROM recordings WHERE dataset_id = ?
        """, conn, params=[row['id']]).iloc[0]['count']
        print(f"  数据集 {row['id']} ({row['name']}): {recordings_count} 条记录")
    
    conn.close()
    
    print("\n" + "=" * 50)
    print("✅ 数据库查询完成")

if __name__ == "__main__":
    show_experiment_results() 