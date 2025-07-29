#!/usr/bin/env python3
"""
执行模式配置文件

通过环境变量或配置文件设置全局执行模式
"""

import os

# 默认执行模式
DEFAULT_EXECUTION_MODE = 'local'

def get_execution_mode() -> str:
    """获取全局执行模式"""
    # 优先从环境变量读取
    mode = os.getenv('EEG2GO_EXECUTION_MODE', DEFAULT_EXECUTION_MODE)
    
    # 验证模式是否有效
    valid_modes = ['local', 'pbs']
    if mode not in valid_modes:
        print(f"Warning: Invalid execution mode '{mode}', using default '{DEFAULT_EXECUTION_MODE}'")
        mode = DEFAULT_EXECUTION_MODE
    
    return mode

# 全局变量
EXECUTION_MODE = get_execution_mode()

def is_pbs_mode() -> bool:
    """检查是否为PBS模式"""
    return EXECUTION_MODE == 'pbs'

def is_local_mode() -> bool:
    """检查是否为本地模式"""
    return EXECUTION_MODE == 'local' 