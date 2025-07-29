#!/usr/bin/env python3
"""
系统状态检查脚本

检查当前执行模式和相关系统状态
"""

import os
import subprocess
import sqlite3
from pathlib import Path

def check_execution_mode():
    """检查执行模式"""
    mode = os.getenv('EEG2GO_EXECUTION_MODE', 'local')
    print(f"当前执行模式: {mode}")
    return mode

def check_pbs_system():
    """检查PBS系统状态"""
    try:
        result = subprocess.run(['qstat', '-q'], capture_output=True, text=True, timeout=5)
        if result.returncode == 0:
            print("✅ PBS系统可用")
            return True
        else:
            print("❌ PBS系统不可用")
            return False
    except Exception as e:
        print(f"❌ PBS系统检查失败: {e}")
        return False

def check_database():
    """检查数据库状态"""
    db_path = os.getenv('DATABASE_PATH', 'database/eeg2go.db')
    if os.path.exists(db_path):
        print(f"✅ 数据库文件存在: {db_path}")
        
        # 检查数据库连接
        try:
            conn = sqlite3.connect(db_path)
            c = conn.cursor()
            c.execute("SELECT COUNT(*) FROM tasks")
            task_count = c.fetchone()[0]
            conn.close()
            print(f"✅ 数据库连接正常，任务数量: {task_count}")
            return True
        except Exception as e:
            print(f"❌ 数据库连接失败: {e}")
            return False
    else:
        print(f"❌ 数据库文件不存在: {db_path}")
        return False

def check_directories():
    """检查必要目录"""
    directories = ['database', 'logs', 'pbs_templates']
    for directory in directories:
        if os.path.exists(directory):
            print(f"✅ 目录存在: {directory}")
        else:
            print(f"❌ 目录不存在: {directory}")

def check_python_modules():
    """检查Python模块"""
    modules = [
        'flask',
        'sqlite3',
        'pandas',
        'numpy'
    ]
    
    for module in modules:
        try:
            __import__(module)
            print(f"✅ 模块可用: {module}")
        except ImportError:
            print(f"❌ 模块缺失: {module}")

def main():
    """主函数"""
    print("=== EEG2GO 系统状态检查 ===")
    print()
    
    # 检查执行模式
    mode = check_execution_mode()
    print()
    
    # 检查PBS系统
    if mode == 'pbs':
        check_pbs_system()
    else:
        print("本地模式，跳过PBS检查")
    print()
    
    # 检查数据库
    check_database()
    print()
    
    # 检查目录
    check_directories()
    print()
    
    # 检查Python模块
    check_python_modules()
    print()
    
    print("=== 检查完成 ===")

if __name__ == "__main__":
    main() 