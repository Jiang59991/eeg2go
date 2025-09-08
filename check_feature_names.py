#!/usr/bin/env python3
"""
检查数据库中的特征名称
"""

import sqlite3
import json
import sys
import os

def check_feature_names():
    """检查数据库中的特征名称"""
    db_path = "database/eeg2go.db"
    
    if not os.path.exists(db_path):
        print(f"❌ 数据库文件不存在: {db_path}")
        return
    
    print(f"=== 检查数据库中的特征名称: {db_path} ===")
    
    try:
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        
        # 1. 检查fxdef表
        print("\n1. 检查fxdef表:")
        cursor.execute("SELECT id, shortname, chans FROM fxdef LIMIT 10")
        fxdefs = cursor.fetchall()
        print(f"   前10个特征定义:")
        for fxdef in fxdefs:
            print(f"     ID: {fxdef[0]}, 名称: {fxdef[1]}, 通道: {fxdef[2]}")
        
        # 2. 检查feature_values表
        print("\n2. 检查feature_values表:")
        cursor.execute("SELECT DISTINCT fxdef_id FROM feature_values LIMIT 10")
        fxdef_ids = cursor.fetchall()
        print(f"   前10个特征ID: {[f[0] for f in fxdef_ids]}")
        
        # 3. 检查特定特征集
        print("\n3. 检查特征集2:")
        cursor.execute("SELECT fsi.fxdef_id, f.shortname, f.chans FROM feature_set_items fsi JOIN fxdef f ON fsi.fxdef_id = f.id WHERE fsi.feature_set_id = 2 LIMIT 10")
        feature_set_items = cursor.fetchall()
        print(f"   特征集2的前10个特征:")
        for item in feature_set_items:
            print(f"     fxdef_id: {item[0]}, 名称: {item[1]}, 通道: {item[2]}")
        
        # 4. 检查特征值示例
        print("\n4. 检查特征值示例:")
        cursor.execute("SELECT fv.fxdef_id, fv.value, fv.dim, f.shortname, f.chans FROM feature_values fv JOIN fxdef f ON fv.fxdef_id = f.id LIMIT 5")
        feature_values = cursor.fetchall()
        print(f"   前5个特征值:")
        for fv in feature_values:
            print(f"     fxdef_id: {fv[0]}, 维度: {fv[2]}, 名称: {fv[3]}, 通道: {fv[4]}")
            try:
                value = json.loads(fv[1]) if fv[1] else None
                if isinstance(value, list):
                    print(f"       值类型: list, 长度: {len(value)}")
                elif isinstance(value, (int, float)):
                    print(f"       值类型: {type(value).__name__}, 值: {value}")
                else:
                    print(f"       值类型: {type(value).__name__}")
            except:
                print(f"       值解析失败: {fv[1]}")
        
        # 5. 检查数据集7的记录
        print("\n5. 检查数据集7的记录:")
        cursor.execute("SELECT COUNT(*) FROM recordings WHERE dataset_id = 7")
        recording_count = cursor.fetchone()[0]
        print(f"   数据集7的记录数: {recording_count}")
        
        if recording_count > 0:
            cursor.execute("SELECT id FROM recordings WHERE dataset_id = 7 LIMIT 3")
            recording_ids = cursor.fetchall()
            print(f"   前3个记录ID: {[r[0] for r in recording_ids]}")
            
            # 检查第一个记录的特征
            if recording_ids:
                first_recording = recording_ids[0][0]
                cursor.execute("""
                    SELECT fv.fxdef_id, fv.value, fv.dim, f.shortname, f.chans 
                    FROM feature_values fv 
                    JOIN fxdef f ON fv.fxdef_id = f.id 
                    WHERE fv.recording_id = ? 
                    LIMIT 5
                """, (first_recording,))
                first_features = cursor.fetchall()
                print(f"   记录{first_recording}的前5个特征:")
                for feat in first_features:
                    print(f"     fxdef_id: {feat[0]}, 维度: {feat[2]}, 名称: {feat[3]}, 通道: {feat[4]}")
        
        conn.close()
        
    except Exception as e:
        print(f"❌ 检查失败: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    check_feature_names()
