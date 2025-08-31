#!/usr/bin/env python3
"""
重建experiment相关表格和tasks表格的Python脚本
此脚本会删除现有的experiment和tasks数据，重新创建表格结构
"""

import sqlite3
import os
import sys
from pathlib import Path

def rebuild_experiment_tables(db_path: str = "database/eeg2go.db"):
    """
    重建experiment相关表格和tasks表格
    
    Args:
        db_path: 数据库文件路径
    """
    try:
        # 确保数据库目录存在
        db_dir = os.path.dirname(db_path)
        if db_dir and not os.path.exists(db_dir):
            os.makedirs(db_dir)
            print(f"创建数据库目录: {db_dir}")
        
        # 读取SQL脚本
        script_path = Path(__file__).parent / "rebuild_experiment_tables.sql"
        if not script_path.exists():
            print(f"错误: 找不到SQL脚本文件 {script_path}")
            return False
        
        with open(script_path, 'r', encoding='utf-8') as f:
            sql_script = f.read()
        
        # 连接数据库
        print(f"连接到数据库: {db_path}")
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 执行SQL脚本
        print("开始重建表格...")
        cursor.executescript(sql_script)
        
        # 提交更改
        conn.commit()
        
        # 验证表格是否创建成功
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='table' AND name IN (
                'experiment_definitions', 
                'experiment_results', 
                'experiment_metadata', 
                'experiment_feature_results', 
                'tasks'
            )
        """)
        
        created_tables = [row[0] for row in cursor.fetchall()]
        expected_tables = [
            'experiment_definitions', 
            'experiment_results', 
            'experiment_metadata', 
            'experiment_feature_results', 
            'tasks'
        ]
        
        missing_tables = set(expected_tables) - set(created_tables)
        if missing_tables:
            print(f"警告: 以下表格可能未成功创建: {missing_tables}")
        else:
            print("所有表格创建成功!")
        
        # 检查视图
        cursor.execute("""
            SELECT name FROM sqlite_master 
            WHERE type='view' AND name IN (
                'feature_experiment_summary', 
                'feature_correlation_history', 
                'feature_importance_history'
            )
        """)
        
        created_views = [row[0] for row in cursor.fetchall()]
        expected_views = [
            'feature_experiment_summary', 
            'feature_correlation_history', 
            'feature_importance_history'
        ]
        
        missing_views = set(expected_views) - set(created_views)
        if missing_views:
            print(f"警告: 以下视图可能未成功创建: {missing_views}")
        else:
            print("所有视图创建成功!")
        
        # 检查默认数据
        cursor.execute("SELECT COUNT(*) FROM experiment_definitions")
        def_count = cursor.fetchone()[0]
        print(f"experiment_definitions表中的记录数: {def_count}")
        
        cursor.execute("SELECT COUNT(*) FROM tasks")
        task_count = cursor.fetchone()[0]
        print(f"tasks表中的记录数: {task_count}")
        
        conn.close()
        print("数据库重建完成!")
        return True
        
    except Exception as e:
        print(f"重建过程中发生错误: {e}")
        if 'conn' in locals():
            conn.rollback()
            conn.close()
        return False

def main():
    """主函数"""
    print("=" * 60)
    print("EEG2GO 数据库表格重建工具")
    print("=" * 60)
    print("此脚本将重建以下表格:")
    print("- experiment_definitions")
    print("- experiment_results") 
    print("- experiment_metadata")
    print("- experiment_feature_results")
    print("- tasks")
    print()
    print("警告: 此操作将删除现有的experiment和tasks数据!")
    print()
    
    # 获取数据库路径
    db_path = os.getenv('DATABASE_PATH', 'database/eeg2go.db')
    
    # 确认操作
    confirm = input(f"确认要重建数据库表格吗? (数据库路径: {db_path}) [y/N]: ")
    if confirm.lower() not in ['y', 'yes']:
        print("操作已取消")
        return
    
    # 执行重建
    success = rebuild_experiment_tables(db_path)
    
    if success:
        print("\n✅ 表格重建成功!")
        print("现在可以重新使用experiment和task功能了")
    else:
        print("\n❌ 表格重建失败!")
        print("请检查错误信息并重试")

if __name__ == "__main__":
    main()
