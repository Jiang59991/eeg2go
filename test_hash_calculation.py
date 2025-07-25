#!/usr/bin/env python3
"""
验证哈希计算
"""

import hashlib

def test_hash_calculation():
    """测试哈希计算"""
    node_hash = "f3f1e05db168e250d74d4958cb954706"
    parent_hash = ""
    
    # 计算upstream_hash
    upstream_hash = hashlib.md5((parent_hash + node_hash).encode()).hexdigest()
    
    print(f"节点哈希: {node_hash}")
    print(f"父节点哈希: {parent_hash}")
    print(f"计算的upstream_hash: {upstream_hash}")
    print(f"实际的upstream_hash: cdfb2b420f33c59c8277e9d4a72229ac")
    print(f"是否匹配: {upstream_hash == 'cdfb2b420f33c59c8277e9d4a72229ac'}")

if __name__ == "__main__":
    test_hash_calculation() 