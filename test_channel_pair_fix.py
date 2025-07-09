#!/usr/bin/env python3
"""
测试通道对特征修复
"""

import sys
sys.path.append('.')

from eeg2fx.featureset_fetcher import split_channel

def test_split_channel():
    """测试split_channel函数"""
    print("=== 测试split_channel函数 ===")
    
    # 测试通道对特征
    test_cases = [
        # 通道对特征
        ("C3-C4", {"C3-C4_asymmetry": [1.0, 2.0, 3.0]}),
        ("C3-C4", {"C3-C4_coherence": [0.5, 0.6, 0.7]}),
        ("C3-C4", {"C3-C4_plv": [0.8, 0.9, 1.0]}),
        ("F3-F4", {"F3-F4_asymmetry": [1.1, 2.1, 3.1]}),
        
        # 单通道特征
        ("C3", {"C3": [1.0, 2.0, 3.0]}),
        ("Pz", {"PZ": [0.5, 0.6, 0.7]}),
        
        # 不匹配的情况
        ("C3-C4", {"C3": [1.0, 2.0, 3.0]}),  # 应该返回[]
        ("C3", {"C3-C4_asymmetry": [1.0, 2.0, 3.0]}),  # 应该返回[]
    ]
    
    for chan, result_dict in test_cases:
        result = split_channel(result_dict, chan)
        print(f"通道 '{chan}' 在 {list(result_dict.keys())} 中:")
        print(f"  结果: {result}")
        print()

if __name__ == "__main__":
    test_split_channel() 