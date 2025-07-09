#!/usr/bin/env python3
"""
测试特征集缓存检查功能

这个脚本用于测试修改后的 featureset_fetcher.py 中的缓存检查逻辑
"""

import os
import sys
import time
from eeg2fx.featureset_fetcher import run_feature_set, check_all_features_cached
from logging_config import logger

def test_cache_check():
    """测试缓存检查功能"""
    feature_set_id = 1  # 使用特征集ID 1
    recording_id = 1    # 使用录音ID 1
    
    print("=" * 60)
    print("测试特征集缓存检查功能")
    print("=" * 60)
    print(f"特征集ID: {feature_set_id}")
    print(f"录音ID: {recording_id}")
    print()
    
    # 第一次运行：应该会计算特征
    print("第一次运行 - 应该会计算特征...")
    start_time = time.time()
    results1 = run_feature_set(feature_set_id, recording_id)
    duration1 = time.time() - start_time
    
    print(f"第一次运行完成，耗时: {duration1:.2f} 秒")
    print(f"返回特征数量: {len(results1)}")
    print()
    
    # 检查缓存状态
    print("检查缓存状态...")
    all_cached, cached_results = check_all_features_cached(feature_set_id, recording_id)
    print(f"所有特征都已缓存: {all_cached}")
    print(f"缓存结果数量: {len(cached_results)}")
    print()
    
    # 第二次运行：应该直接从缓存读取
    print("第二次运行 - 应该直接从缓存读取...")
    start_time = time.time()
    results2 = run_feature_set(feature_set_id, recording_id)
    duration2 = time.time() - start_time
    
    print(f"第二次运行完成，耗时: {duration2:.2f} 秒")
    print(f"返回特征数量: {len(results2)}")
    print()
    
    # 比较结果
    print("结果比较:")
    print(f"  第一次运行耗时: {duration1:.2f} 秒")
    print(f"  第二次运行耗时: {duration2:.2f} 秒")
    print(f"  性能提升: {duration1/duration2:.1f}x")
    print(f"  结果数量一致: {len(results1) == len(results2)}")
    
    # 检查结果内容是否一致
    if len(results1) == len(results2):
        content_match = True
        for fxid in results1:
            if fxid not in results2:
                content_match = False
                break
            # 简单比较value是否相同（对于复杂对象可能需要更详细的比较）
            if results1[fxid].get('value') != results2[fxid].get('value'):
                content_match = False
                break
        print(f"  结果内容一致: {content_match}")
    
    print()
    print("测试完成!")

if __name__ == "__main__":
    test_cache_check() 