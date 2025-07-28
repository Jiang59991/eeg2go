#!/usr/bin/env python3
"""
详细诊断slice错误的脚本 - 有选择地输出长内容
"""

import sqlite3
import json
import traceback

def safe_print(content, max_length=200, field_name=""):
    """安全地打印内容，限制长度"""
    if content is None:
        print(f"{field_name}: None")
        return
    
    content_str = str(content)
    if len(content_str) <= max_length:
        print(f"{field_name}: {content_str}")
    else:
        print(f"{field_name}: {content_str[:max_length]}... (总长度: {len(content_str)})")

def diagnose_slice_error():
    """详细诊断slice错误"""
    db_path = "database/eeg2go.db"
    
    print("=== 详细诊断slice错误 ===")
    
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # 检查任务ID=4的所有字段
        print("\n1. 检查任务ID=4的所有字段:")
        c.execute("SELECT * FROM tasks WHERE id = 4")
        task = c.fetchone()
        
        if task:
            print(f"任务ID: {task[0]}")
            print(f"任务类型: {task[1]}")
            print(f"状态: {task[2]}")
            safe_print(task[3], 100, "参数")
            safe_print(task[4], 100, "结果")
            safe_print(task[5], 100, "错误信息")
            print(f"创建时间: {task[6]}")
            print(f"开始时间: {task[7]}")
            print(f"完成时间: {task[8]}")
            print(f"优先级: {task[9]}")
            print(f"数据集ID: {task[10]}")
            print(f"特征集ID: {task[11]}")
            print(f"实验类型: {task[12]}")
            print(f"进度: {task[13]}")
            print(f"已处理: {task[14]}")
            print(f"总数: {task[15]}")
            safe_print(task[16], 100, "备注")
            
            # 逐个测试JSON解析
            print("\n2. 测试JSON解析:")
            
            # 测试parameters
            print("\n测试parameters字段:")
            try:
                if task[3]:
                    params = json.loads(task[3])
                    print(f"Parameters解析成功: {params}")
                else:
                    print("Parameters为空")
            except Exception as e:
                print(f"Parameters解析失败: {e}")
                safe_print(task[3], 200, "Parameters内容")
            
            # 测试result
            print("\n测试result字段:")
            try:
                if task[4]:
                    result = json.loads(task[4])
                    print(f"Result解析成功，类型: {type(result)}")
                    if isinstance(result, dict):
                        print(f"Result键: {list(result.keys())}")
                    elif isinstance(result, list):
                        print(f"Result长度: {len(result)}")
                    else:
                        print(f"Result值: {result}")
                else:
                    print("Result为空")
            except Exception as e:
                print(f"Result解析失败: {e}")
                safe_print(task[4], 300, "Result内容")
                print(f"Result类型: {type(task[4])}")
                
                # 检查是否包含slice
                result_str = str(task[4])
                if 'slice(' in result_str:
                    print("发现slice对象!")
                    # 找到slice的位置
                    import re
                    slice_matches = re.findall(r'slice\([^)]*\)', result_str)
                    print(f"找到的slice对象数量: {len(slice_matches)}")
                    if slice_matches:
                        print(f"前3个slice对象: {slice_matches[:3]}")
                    
                    # 显示slice周围的上下文
                    slice_positions = []
                    for match in re.finditer(r'slice\([^)]*\)', result_str):
                        start = max(0, match.start() - 50)
                        end = min(len(result_str), match.end() + 50)
                        context = result_str[start:end]
                        slice_positions.append(f"...{context}...")
                    
                    print(f"Slice上下文 (前3个):")
                    for i, context in enumerate(slice_positions[:3]):
                        print(f"  {i+1}: {context}")
        
        conn.close()
        
    except Exception as e:
        print(f"错误: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    diagnose_slice_error() 