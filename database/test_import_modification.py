#!/usr/bin/env python3
"""
测试修改后的导入逻辑
验证每个subject只导入一条recording
"""

import os
import sqlite3
import sys

# 添加项目根目录到Python路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_import_logic():
    """测试导入逻辑"""
    db_path = "database/eeg2go.db"
    
    if not os.path.exists(db_path):
        print("数据库文件不存在，请先运行导入脚本")
        return
    
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    # 检查harvard数据集
    cursor.execute("""
        SELECT id, name FROM datasets 
        WHERE name LIKE '%harvard%'
    """)
    
    harvard_datasets = cursor.fetchall()
    
    if not harvard_datasets:
        print("没有找到harvard数据集")
        return
    
    print("Harvard数据集统计:")
    print("=" * 50)
    
    for dataset_id, dataset_name in harvard_datasets:
        print(f"\n数据集: {dataset_name} (ID: {dataset_id})")
        
        # 统计subjects数量
        cursor.execute("""
            SELECT COUNT(DISTINCT subject_id) 
            FROM subjects 
            WHERE dataset_id = ?
        """, (dataset_id,))
        subject_count = cursor.fetchone()[0]
        
        # 统计recordings数量
        cursor.execute("""
            SELECT COUNT(*) 
            FROM recordings 
            WHERE dataset_id = ?
        """, (dataset_id,))
        recording_count = cursor.fetchone()[0]
        
        # 检查每个subject的recording数量
        cursor.execute("""
            SELECT subject_id, COUNT(*) as recording_count
            FROM recordings 
            WHERE dataset_id = ?
            GROUP BY subject_id
            ORDER BY recording_count DESC
        """, (dataset_id,))
        
        subject_recording_counts = cursor.fetchall()
        
        print(f"  受试者数量: {subject_count}")
        print(f"  记录数量: {recording_count}")
        print(f"  平均每个受试者的记录数: {recording_count/subject_count:.2f}")
        
        # 检查是否有subject有多个recording
        subjects_with_multiple = [s for s in subject_recording_counts if s[1] > 1]
        
        if subjects_with_multiple:
            print(f"  ⚠️  发现 {len(subjects_with_multiple)} 个受试者有多个记录:")
            for subject_id, count in subjects_with_multiple[:5]:  # 只显示前5个
                print(f"    - {subject_id}: {count} 条记录")
            if len(subjects_with_multiple) > 5:
                print(f"    ... 还有 {len(subjects_with_multiple) - 5} 个受试者")
        else:
            print("  ✅ 每个受试者都只有一条记录")
        
        # 显示记录时长分布
        cursor.execute("""
            SELECT MIN(duration), MAX(duration), AVG(duration), COUNT(*)
            FROM recordings 
            WHERE dataset_id = ?
        """, (dataset_id,))
        
        min_dur, max_dur, avg_dur, total = cursor.fetchone()
        print(f"  记录时长统计:")
        print(f"    最短: {min_dur:.1f}s")
        print(f"    最长: {max_dur:.1f}s")
        print(f"    平均: {avg_dur:.1f}s")
    
    conn.close()

if __name__ == "__main__":
    test_import_logic()



