#!/usr/bin/env python3
"""
快速清理所有失败的特征记录，包括完全失败和所有epoch值都为nan的特征
"""

import os
import sqlite3
import sys
import json
import numpy as np

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

DB_PATH = os.path.abspath(os.path.join("database", "eeg2go.db"))

def is_all_nan_feature(value_json):
    """
    Check if a feature has all nan values across all epochs.
    
    Args:
        value_json (str): JSON string of the feature value
        
    Returns:
        bool: True if all epoch values are nan
    """
    try:
        if value_json is None or value_json == "null":
            return False
            
        data = json.loads(value_json)
        
        # Check if it's a structured result (dict with channel names)
        if isinstance(data, dict):
            for channel_data in data.values():
                if isinstance(channel_data, list):
                    for epoch_data in channel_data:
                        if isinstance(epoch_data, dict) and "value" in epoch_data:
                            value = epoch_data["value"]
                            # Check if value is not nan
                            if value is not None and not (isinstance(value, float) and np.isnan(value)):
                                return False
                else:
                    # Single value case
                    if channel_data is not None and not (isinstance(channel_data, float) and np.isnan(channel_data)):
                        return False
        elif isinstance(data, list) and len(data) > 0:
            # 列表，每个元素是dict，判断每个dict的"value"字段
            for item in data:
                if isinstance(item, dict) and "value" in item:
                    v = item["value"]
                    if v is not None and not (isinstance(v, float) and np.isnan(v)):
                        return False
                else:
                    # 不是dict或没有value字段，视为非全nan
                    return False
            return True
        elif isinstance(data, list):
            # 兼容老格式：直接list of float
            for value in data:
                if value is not None and not (isinstance(value, float) and np.isnan(value)):
                    return False
        else:
            # Single value case
            if data is not None and not (isinstance(data, float) and np.isnan(data)):
                return False
                
        return True
        
    except (json.JSONDecodeError, TypeError):
        return False

def quick_clean_failed_features():
    """快速清理所有失败的特征记录，包括完全失败和所有epoch值都为nan的特征"""
    print("=== 快速清理所有失败的特征记录 ===")
    
    if not os.path.exists(DB_PATH):
        print(f"错误: 数据库文件不存在: {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 1. 清理完全失败的特征（value为'null'或'[]'）
    fail_condition = "value = 'null' OR value = '[]'"
    
    # 统计要删除的完全失败记录数
    c.execute(f"SELECT COUNT(*) FROM feature_values WHERE {fail_condition}")
    completely_failed_count = c.fetchone()[0]
    
    # 删除完全失败的特征记录
    if completely_failed_count > 0:
        print(f"删除 {completely_failed_count} 条完全失败的特征记录...")
        c.execute(f"DELETE FROM feature_values WHERE {fail_condition}")
        print(f"✓ 成功删除了 {c.rowcount} 条完全失败的特征记录。")
    else:
        print("✓ 没有找到完全失败的特征记录。")
    
    # 2. 清理所有epoch值都为nan的特征
    # 获取所有非空且非完全失败的特征
    c.execute("""
        SELECT fxdef_id, recording_id, value 
        FROM feature_values 
        WHERE value IS NOT NULL 
        AND value != 'null'
        AND value != '[]'
    """)
    all_features = c.fetchall()
    
    # 找出所有epoch值都为nan的特征
    all_nan_features = []
    for fxdef_id, recording_id, value_json in all_features:
        if is_all_nan_feature(value_json):
            all_nan_features.append((fxdef_id, recording_id))
    
    # 删除所有epoch值都为nan的特征
    all_nan_count = len(all_nan_features)
    if all_nan_count > 0:
        print(f"删除 {all_nan_count} 条所有epoch值都为nan的特征记录...")
        
        # 构建删除条件
        delete_conditions = []
        for fxdef_id, recording_id in all_nan_features:
            delete_conditions.append(f"(fxdef_id = {fxdef_id} AND recording_id = {recording_id})")
        
        if delete_conditions:
            delete_sql = f"DELETE FROM feature_values WHERE {' OR '.join(delete_conditions)}"
            c.execute(delete_sql)
            print(f"✓ 成功删除了 {c.rowcount} 条所有epoch值都为nan的特征记录。")
    else:
        print("✓ 没有找到所有epoch值都为nan的特征记录。")
    
    # 提交更改
    conn.commit()
    
    total_deleted = completely_failed_count + all_nan_count
    if total_deleted > 0:
        print(f"\n总共删除了 {total_deleted} 条问题特征记录:")
        print(f"  - 完全失败的特征: {completely_failed_count} 条")
        print(f"  - 所有epoch值都为nan的特征: {all_nan_count} 条")
        
        # VACUUM to reclaim disk space
        print("\n正在整理数据库空间...")
        c.execute("VACUUM")
        conn.commit()
        print("✓ 数据库空间整理完成。")
        
        print("\n清理完成。现在可以重新运行实验，系统会重新生成这些特征。")
    else:
        print("\n✓ 没有找到需要清理的特征记录。")
    
    conn.close()

if __name__ == "__main__":
    quick_clean_failed_features() 