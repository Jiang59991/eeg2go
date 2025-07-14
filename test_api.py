import requests
import json

# 测试API端点
base_url = 'http://localhost:5000'

def test_api():
    try:
        # 测试datasets API
        print("Testing /api/datasets...")
        response = requests.get(f'{base_url}/api/datasets')
        if response.status_code == 200:
            datasets = response.json()
            print(f"Datasets API OK: {len(datasets)} datasets found")
            for ds in datasets[:3]:  # 显示前3个
                print(f"  - {ds['id']}: {ds['name']}")
        else:
            print(f"Datasets API failed: {response.status_code}")
        
        # 测试feature_sets API
        print("\nTesting /api/feature_sets...")
        response = requests.get(f'{base_url}/api/feature_sets')
        if response.status_code == 200:
            feature_sets = response.json()
            print(f"Feature sets API OK: {len(feature_sets)} feature sets found")
            for fs in feature_sets[:3]:  # 显示前3个
                print(f"  - {fs['id']}: {fs['name']}")
        else:
            print(f"Feature sets API failed: {response.status_code}")
            
    except Exception as e:
        print(f"Error testing API: {e}")

if __name__ == '__main__':
    test_api() 