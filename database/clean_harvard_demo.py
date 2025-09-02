#!/usr/bin/env python3
"""
删除现有的harvard_demo相关数据集和记录的脚本
"""

import sqlite3
import os
from logging_config import logger

def clean_harvard_demo_data(db_path: str = "database/eeg2go.db"):
    """
    删除所有harvard_demo相关的数据集和记录
    
    Args:
        db_path: 数据库文件路径
    """
    try:
        print(f"连接到数据库: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 查找所有harvard_demo相关的数据集
        cursor.execute("""
            SELECT id, name FROM datasets 
            WHERE name LIKE '%harvard%'
        """)
        
        harvard_datasets = cursor.fetchall()
        
        if not harvard_datasets:
            print("没有找到harvard_demo相关的数据集")
            return True
        
        print(f"找到 {len(harvard_datasets)} 个harvard_demo相关数据集:")
        for dataset_id, dataset_name in harvard_datasets:
            print(f"  - {dataset_name} (ID: {dataset_id})")
        
        # 确认删除
        confirm = input("\n确认要删除这些数据集及其所有相关记录吗? [y/N]: ")
        if confirm.lower() not in ['y', 'yes']:
            print("操作已取消")
            return False
        
        deleted_count = 0
        
        for dataset_id, dataset_name in harvard_datasets:
            print(f"\n正在删除数据集: {dataset_name}")
            
            # 1. 删除recording_events (先删除，因为有外键约束)
            cursor.execute("""
                DELETE FROM recording_events 
                WHERE recording_id IN (
                    SELECT id FROM recordings WHERE dataset_id = ?
                )
            """, (dataset_id,))
            events_deleted = cursor.rowcount
            print(f"  删除 {events_deleted} 条recording_events记录")
            
            # 2. 删除recording_metadata
            cursor.execute("""
                DELETE FROM recording_metadata 
                WHERE recording_id IN (
                    SELECT id FROM recordings WHERE dataset_id = ?
                )
            """, (dataset_id,))
            metadata_deleted = cursor.rowcount
            print(f"  删除 {metadata_deleted} 条recording_metadata记录")
            
            # 3. 删除recordings
            cursor.execute("DELETE FROM recordings WHERE dataset_id = ?", (dataset_id,))
            recordings_deleted = cursor.rowcount
            print(f"  删除 {recordings_deleted} 条recordings记录")
            
            # 4. 删除subjects
            cursor.execute("DELETE FROM subjects WHERE dataset_id = ?", (dataset_id,))
            subjects_deleted = cursor.rowcount
            print(f"  删除 {subjects_deleted} 条subjects记录")
            
            # 5. 删除datasets
            cursor.execute("DELETE FROM datasets WHERE id = ?", (dataset_id,))
            print(f"  删除数据集: {dataset_name}")
            
            deleted_count += 1
        
        # 提交更改
        conn.commit()
        
        print(f"\n✅ 成功删除 {deleted_count} 个harvard_demo数据集及其所有相关记录!")
        
        # 验证删除结果
        cursor.execute("""
            SELECT COUNT(*) FROM datasets 
            WHERE name LIKE '%harvard%' AND name LIKE '%demo%'
        """)
        remaining_count = cursor.fetchone()[0]
        
        if remaining_count == 0:
            print("✅ 确认所有harvard_demo相关数据已完全删除")
        else:
            print(f"⚠️  警告: 仍有 {remaining_count} 个harvard_demo相关数据集存在")
        
        conn.close()
        return True
        
    except Exception as e:
        print(f"删除过程中发生错误: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("Harvard Demo 数据清理工具")
    print("=" * 60)
    print("此脚本将删除所有harvard_demo相关的数据集和记录:")
    print("- datasets表中的harvard_demo数据集")
    print("- subjects表中的相关受试者")
    print("- recordings表中的相关记录")
    print("- recording_events表中的相关事件")
    print("- recording_metadata表中的相关元数据")
    print()
    print("⚠️  警告: 此操作不可逆，请确保已备份重要数据!")
    print()
    
    # 获取数据库路径
    db_path = os.getenv('DATABASE_PATH', 'database/eeg2go.db')
    
    # 执行清理
    success = clean_harvard_demo_data(db_path)
    
    if success:
        print("\n✅ 数据清理完成!")
        print("现在可以重新导入新的harvard_1000数据集了")
    else:
        print("\n❌ 数据清理失败!")
        print("请检查错误信息并重试")

if __name__ == "__main__":
    main()
