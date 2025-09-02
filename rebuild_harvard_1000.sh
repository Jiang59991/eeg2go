#!/bin/bash

# Harvard 1000 数据集重建脚本
# 此脚本会清理旧的harvard_demo数据并导入新的harvard_1000数据

echo "============================================================"
echo "Harvard 1000 数据集重建工具"
echo "============================================================"
echo "此脚本将执行以下操作:"
echo "1. 删除现有的harvard_demo相关数据集和记录"
echo "2. 导入新的harvard_1000数据集"
echo ""

# 检查Python是否可用
if ! command -v python3 &> /dev/null; then
    echo "错误: 找不到python3命令"
    exit 1
fi

# 检查脚本文件是否存在
if [ ! -f "database/clean_harvard_demo.py" ]; then
    echo "错误: 找不到清理脚本 database/clean_harvard_demo.py"
    exit 1
fi

if [ ! -f "database/import_harvard_demo.py" ]; then
    echo "错误: 找不到导入脚本 database/import_harvard_1000.py"
    exit 1
fi

# 设置数据库路径环境变量（如果未设置）
if [ -z "$DATABASE_PATH" ]; then
    export DATABASE_PATH="database/eeg2go.db"
    echo "使用默认数据库路径: $DATABASE_PATH"
else
    echo "使用环境变量数据库路径: $DATABASE_PATH"
fi

# 第一步：清理旧的harvard_demo数据
echo ""
echo "第一步：清理旧的harvard_demo数据..."
python -m database.clean_harvard_demo

if [ $? -ne 0 ]; then
    echo "❌ 清理失败，停止执行"
    exit 1
fi

echo ""
echo "✅ 清理完成!"

# 第二步：导入新的harvard_1000数据
echo ""
echo "第二步：导入新的harvard_1000数据..."
python -m database.import_harvard_demo

if [ $? -ne 0 ]; then
    echo "❌ 导入失败"
    exit 1
fi

echo ""
echo "✅ Harvard 1000 数据集重建完成!"
echo "新的数据集包括:"
echo "- Harvard_S0001_1000"
echo "- Harvard_I0003_1000"
