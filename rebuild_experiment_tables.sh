#!/bin/bash

# EEG2GO 数据库表格重建脚本
# 此脚本会重建experiment相关表格和tasks表格

echo "============================================================"
echo "EEG2GO 数据库表格重建工具"
echo "============================================================"
echo "此脚本将重建以下表格:"
echo "- experiment_definitions"
echo "- experiment_results" 
echo "- experiment_metadata"
echo "- experiment_feature_results"
echo "- tasks"
echo ""
echo "警告: 此操作将删除现有的experiment和tasks数据!"
echo ""

# 检查Python是否可用
if ! command -v python3 &> /dev/null; then
    echo "错误: 找不到python3命令"
    exit 1
fi

# 检查脚本文件是否存在
if [ ! -f "database/rebuild_experiment_tables.py" ]; then
    echo "错误: 找不到重建脚本 database/rebuild_experiment_tables.py"
    exit 1
fi

# 设置数据库路径环境变量（如果未设置）
if [ -z "$DATABASE_PATH" ]; then
    export DATABASE_PATH="database/eeg2go.db"
    echo "使用默认数据库路径: $DATABASE_PATH"
else
    echo "使用环境变量数据库路径: $DATABASE_PATH"
fi

# 执行Python脚本
echo "开始执行重建脚本..."
python3 database/rebuild_experiment_tables.py

# 检查执行结果
if [ $? -eq 0 ]; then
    echo ""
    echo "✅ 表格重建完成!"
    echo "现在可以重新使用experiment和task功能了"
else
    echo ""
    echo "❌ 表格重建失败!"
    echo "请检查错误信息并重试"
    exit 1
fi
