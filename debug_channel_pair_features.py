#!/usr/bin/env python3
"""
调试通道对特征失败的原因
"""

import sqlite3
import os
import sys
sys.path.append('.')

from eeg2fx.featureset_grouping import load_fxdefs_for_set
from eeg2fx.steps import load_recording, filter, epoch
from eeg2fx.function_registry import FEATURE_FUNCS
import mne
mne.set_log_level('ERROR')

DB_PATH = "database/eeg2go.db"

def check_recording_channels(recording_id=1):
    """检查recording的通道信息"""
    print(f"=== 检查recording_id={recording_id}的通道信息 ===")
    
    # 加载recording
    try:
        raw = load_recording(recording_id)
        print(f"✓ 成功加载recording_id={recording_id}")
        print(f"  通道数量: {len(raw.ch_names)}")
        print(f"  通道列表: {raw.ch_names}")
        print(f"  采样率: {raw.info['sfreq']} Hz")
        print(f"  数据长度: {raw.n_times / raw.info['sfreq']:.2f} 秒")
        return raw
    except Exception as e:
        print(f"✗ 加载recording失败: {e}")
        return None

def check_failed_features():
    """检查失败的特征定义"""
    print(f"\n=== 检查失败的特征定义 ===")
    
    failed_fxdef_ids = [96, 127, 128, 129, 130, 131, 132]
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    for fxdef_id in failed_fxdef_ids:
        c.execute("""
            SELECT id, shortname, func, chans, params, dim, pipedef_id
            FROM fxdef WHERE id = ?
        """, (fxdef_id,))
        row = c.fetchone()
        if row:
            fxid, shortname, func, chans, params, dim, pipeid = row
            print(f"特征ID {fxid}: {shortname}")
            print(f"  函数: {func}")
            print(f"  通道: {chans}")
            print(f"  参数: {params}")
            print(f"  维度: {dim}")
            print(f"  管道ID: {pipeid}")
            print()
    
    conn.close()

def test_channel_pair_features(raw):
    """测试通道对特征函数"""
    print(f"\n=== 测试通道对特征函数 ===")
    
    if raw is None:
        print("✗ 无法测试，raw为None")
        return
    
    # 按照正确的流程：raw -> filter -> epoch -> 特征函数
    try:
        print("1. 应用滤波器...")
        filtered_raw = filter(raw, hp=1.0, lp=35.0)
        print("   ✓ 滤波器应用成功")
        
        print("2. 分段...")
        epochs = epoch(filtered_raw, duration=2.0)
        print(f"   ✓ 分段成功，生成 {len(epochs)} 个epoch")
        print(f"   每个epoch长度: {epochs.times[-1] - epochs.times[0]:.2f} 秒")
        
    except Exception as e:
        print(f"✗ 预处理失败: {e}")
        return
    
    # 测试的特征和通道对
    test_cases = [
        ("alpha_asymmetry", "C3-C4"),
        ("coherence_band", "C3-C4"),
        ("plv", "C3-C4")
    ]
    
    for func_name, chans in test_cases:
        print(f"\n测试 {func_name} 通道 {chans}:")
        try:
            func = FEATURE_FUNCS[func_name]
            result = func(epochs, chans=chans)
            print(f"  ✓ 成功执行")
            print(f"  结果类型: {type(result)}")
            if hasattr(result, 'keys'):
                print(f"  结果键: {list(result.keys())}")
                if 'value' in result:
                    value = result['value']
                    print(f"  值类型: {type(value)}")
                    if hasattr(value, 'shape'):
                        print(f"  值形状: {value.shape}")
                    if hasattr(value, '__len__'):
                        print(f"  值长度: {len(value)}")
                        if len(value) > 0:
                            print(f"  第一个值: {value[0] if hasattr(value, '__getitem__') else value}")
        except Exception as e:
            print(f"  ✗ 执行失败: {e}")
            import traceback
            traceback.print_exc()

def check_feature_values():
    """检查数据库中这些特征的实际存储值"""
    print(f"\n=== 检查数据库中特征的实际存储值 ===")
    
    failed_fxdef_ids = [96, 127, 128, 129, 130, 131, 132]
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    for fxdef_id in failed_fxdef_ids:
        c.execute("""
            SELECT fv.value, fv.dim, fv.shape, fv.notes, f.shortname
            FROM feature_values fv
            JOIN fxdef f ON fv.fxdef_id = f.id
            WHERE fv.fxdef_id = ? AND fv.recording_id = 1
        """, (fxdef_id,))
        row = c.fetchone()
        if row:
            value, dim, shape, notes, shortname = row
            print(f"特征 {shortname} (ID: {fxdef_id}):")
            print(f"  值: {value}")
            print(f"  维度: {dim}")
            print(f"  形状: {shape}")
            print(f"  备注: {notes}")
            print()
        else:
            print(f"特征ID {fxdef_id}: 未找到记录")
    
    conn.close()

if __name__ == "__main__":
    print("开始调试通道对特征失败问题...")
    
    # 1. 检查recording的通道信息
    raw = check_recording_channels(1)
    
    # 2. 检查失败的特征定义
    check_failed_features()
    
    # 3. 测试通道对特征函数
    test_channel_pair_features(raw)
    
    # 4. 检查数据库中的实际存储值
    check_feature_values()
    
    print("\n调试完成！") 