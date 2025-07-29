#!/bin/bash
# EEG2GO 应用启动脚本

# 设置默认执行模式
export EEG2GO_EXECUTION_MODE=${EEG2GO_EXECUTION_MODE:-local}

# 设置Flask应用
export FLASK_APP=web.app

echo "=== EEG2GO 应用启动 ==="
echo "执行模式: $EEG2GO_EXECUTION_MODE"
echo "数据库路径: ${DATABASE_PATH:-database/eeg2go.db}"
echo "Flask应用: $FLASK_APP"
echo "=========================="

# 检查Python环境
if ! command -v python3 &> /dev/null; then
    echo "错误: 未找到 python3"
    exit 1
fi

# 检查数据库目录
if [ ! -d "database" ]; then
    echo "创建数据库目录..."
    mkdir -p database
fi

# 检查日志目录
if [ ! -d "logs" ]; then
    echo "创建日志目录..."
    mkdir -p logs
fi

# 启动Flask应用
echo "启动Flask应用..."
python3 -m flask run --host=0.0.0.0 --port=5000 