#!/bin/bash

# 清理tmp目录下的所有文件
echo "Cleaning up tmp directory..."

# 删除所有.o文件
rm -f *.o*

# 删除tmp目录下的所有文件
if [ -d "tmp" ]; then
    rm -rf tmp/*
    echo "Cleaned tmp directory"
else
    echo "tmp directory does not exist"
fi

# 删除logs目录下的实验1日志文件（可选）
if [ -d "logs/experiment1" ]; then
    echo "Note: logs/experiment1 directory contains log files"
    echo "To clean logs, run: rm -rf logs/experiment1/*"
fi

echo "Cleanup completed!"
