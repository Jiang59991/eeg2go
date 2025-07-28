#!/usr/bin/env python3
"""
直接修复slice对象的脚本 - 有选择地输出长内容
"""

import sqlite3
import json
import re

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

def fix_slice_direct():
    """直接修复slice对象"""
    db_path = "database/eeg2go.db"
    
    print("=== 直接修复slice对象 ===")
    
    try:
        conn = sqlite3.connect(db_path)
        c = conn.cursor()
        
        # 获取任务ID=4的result字段
        c.execute("SELECT result FROM tasks WHERE id = 4")
        result = c.fetchone()
        
        if result and result[0]:
            safe_print(result[0], 200, "原始result")
            
            # 检查是否包含slice
            result_str = str(result[0])
            if 'slice(' in result_str:
                print("发现slice对象，正在修复...")
                
                # 统计slice对象数量
                slice_matches = re.findall(r'slice\([^)]*\)', result_str)
                print(f"找到 {len(slice_matches)} 个slice对象")
                
                # 方法1: 直接替换slice对象
                def replace_slice(match):
                    slice_str = match.group(0)
                    return f'"{slice_str}"'  # 将slice对象转换为字符串
                
                # 使用正则表达式替换所有slice对象
                fixed_result = re.sub(r'slice\([^)]*\)', replace_slice, result_str)
                
                safe_print(fixed_result, 200, "修复后的result")
                
                # 尝试解析修复后的JSON
                try:
                    parsed_result = json.loads(fixed_result)
                    print("修复后的JSON可以正常解析")
                    print(f"解析后类型: {type(parsed_result)}")
                    
                    # 更新数据库
                    c.execute("UPDATE tasks SET result = ? WHERE id = 4", (fixed_result,))
                    conn.commit()
                    print("已更新数据库")
                    
                except Exception as e:
                    print(f"修复后的JSON仍然无法解析: {e}")
                    
                    # 方法2: 创建全新的安全result
                    safe_result = {
                        "status": "completed",
                        "note": "Original result contained slice objects",
                        "original_content_length": len(result_str),
                        "slice_count": len(slice_matches),
                        "content_preview": result_str[:300]
                    }
                    
                    c.execute("UPDATE tasks SET result = ? WHERE id = 4", 
                             (json.dumps(safe_result),))
                    conn.commit()
                    print("已使用安全result更新数据库")
                    print(f"安全result: {safe_result}")
            else:
                print("未发现slice对象")
        else:
            print("Result字段为空")
        
        conn.close()
        print("修复完成")
        
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    fix_slice_direct()
