#!/usr/bin/env python3
"""
测试API的脚本
"""

import requests
import json

def test_api():
    """测试API"""
    base_url = "http://127.0.0.1:5000"
    
    print("=== 测试API ===")
    
    # 测试获取任务列表
    print("\n1. 测试获取特征提取任务列表:")
    try:
        response = requests.get(f"{base_url}/api/feature_extraction_tasks")
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            tasks = response.json()
            print(f"任务数量: {len(tasks)}")
            for task in tasks:
                print(f"  任务ID: {task.get('id')}, 状态: {task.get('status')}")
        else:
            print(f"错误: {response.text}")
    except Exception as e:
        print(f"请求失败: {e}")
    
    # 测试获取任务详情
    print("\n2. 测试获取任务ID=4的详情:")
    try:
        response = requests.get(f"{base_url}/api/feature_extraction_status/4")
        print(f"状态码: {response.status_code}")
        if response.status_code == 200:
            task = response.json()
            print(f"任务详情: {json.dumps(task[:2], indent=2, default=str)}")
        else:
            print(f"错误: {response.text}")
    except Exception as e:
        print(f"请求失败: {e}")

if __name__ == "__main__":
    test_api()