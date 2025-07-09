#!/usr/bin/env python3
"""
快速清理所有失败的特征记录
"""

import os
import sqlite3
import sys

# 添加项目路径
sys.path.append(os.path.dirname(__file__))

DB_PATH = os.path.abspath(os.path.join("database", "eeg2go.db"))

def quick_clean_failed_features():
    """快速清理所有失败的特征记录"""
    print("=== 快速清理所有失败的特征记录 ===")
    
    if not os.path.exists(DB_PATH):
        print(f"错误: 数据库文件不存在: {DB_PATH}")
        return
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 失败的特征在存储时，其 `value` 字段被记为字符串 "null"
    fail_condition = "value = 'null'"
    
    # 统计要删除的记录数
    c.execute(f"SELECT COUNT(*) FROM feature_values WHERE {fail_condition}")
    count = c.fetchone()[0]
    
    if count == 0:
        print("✓ 没有找到失败的特征记录，无需清理。")
        conn.close()
        return
        
    print(f"将要删除 {count} 条失败的特征记录...")
    
    # 删除所有失败的特征记录
    c.execute(f"DELETE FROM feature_values WHERE {fail_condition}")
    
    # 提交更改
    conn.commit()
    
    print(f"✓ 成功删除了 {c.rowcount} 条失败的特征记录。")
    
    # VACUUM to reclaim disk space
    print("正在整理数据库空间...")
    c.execute("VACUUM")
    conn.commit()
    
    conn.close()
    
    print("清理完成。现在可以重新运行实验，系统会重新生成这些特征。")

if __name__ == "__main__":
    quick_clean_failed_features() 