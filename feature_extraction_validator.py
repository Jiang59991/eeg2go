#!/usr/bin/env python3
"""
特征提取验证器

这个脚本用于：
1. 验证所有特征的代码是否正确
2. 对特定数据集提取完整特征
3. 生成验证报告

使用方法：
python feature_extraction_validator.py [dataset_id] [max_recordings]
"""

import os
import sys
import time
import json
import sqlite3
import argparse
from datetime import datetime
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

from eeg2fx.function_registry import FEATURE_FUNCS
from eeg2fx.featureset_fetcher import run_feature_set
from eeg2fx.featureset_grouping import load_fxdefs_for_set
from eeg2fx.feature_saver import get_failed_features_stats
from database.default_featuresets import create_all_features_featureset

DB_PATH = os.path.join(project_root, "database", "eeg2go.db")

class FeatureExtractionValidator:
    """特征提取验证器"""
    
    def __init__(self, db_path=DB_PATH):
        self.db_path = db_path
        self.validation_results = {
            "start_time": datetime.now().isoformat(),
            "feature_validation": {},
            "extraction_validation": {},
            "summary": {}
        }
    
    def validate_feature_functions(self):
        """验证所有特征函数的代码语法和基本逻辑"""
        print("=" * 60)
        print("验证特征函数代码")
        print("=" * 60)
        
        validation_results = {}
        
        for func_name, func in FEATURE_FUNCS.items():
            print(f"\n验证特征函数: {func_name}")
            
            result = {
                "status": "unknown",
                "error": None,
                "function_info": {}
            }
            
            try:
                # 检查函数是否存在
                if func is None:
                    result["status"] = "error"
                    result["error"] = "函数对象为None"
                    print(f"  ✗ 函数对象为None")
                    continue
                
                # 获取函数信息
                result["function_info"] = {
                    "name": func.__name__,
                    "module": func.__module__,
                    "doc": func.__doc__[:100] + "..." if func.__doc__ and len(func.__doc__) > 100 else func.__doc__
                }
                
                # 检查函数签名
                import inspect
                sig = inspect.signature(func)
                result["function_info"]["signature"] = str(sig)
                
                # 检查是否有必要的参数
                params = list(sig.parameters.keys())
                if "epochs" not in params:
                    result["status"] = "error"
                    result["error"] = "缺少必需的'epochs'参数"
                    print(f"  ✗ 缺少必需的'epochs'参数")
                else:
                    result["status"] = "valid"
                    print(f"  ✓ 函数签名有效")
                
            except Exception as e:
                result["status"] = "error"
                result["error"] = str(e)
                print(f"  ✗ 验证失败: {e}")
            
            validation_results[func_name] = result
        
        self.validation_results["feature_validation"] = validation_results
        
        # 统计结果
        valid_count = sum(1 for r in validation_results.values() if r["status"] == "valid")
        error_count = sum(1 for r in validation_results.values() if r["status"] == "error")
        
        print(f"\n特征函数验证结果:")
        print(f"  有效: {valid_count}")
        print(f"  错误: {error_count}")
        print(f"  总计: {len(validation_results)}")
        
        return validation_results
    
    def get_all_features_featureset_id(self):
        """获取或创建包含所有特征的特征集ID"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # 查找是否已存在
        c.execute("SELECT id FROM feature_sets WHERE name = 'all_available_features'")
        row = c.fetchone()
        
        if row:
            set_id = row[0]
            print(f"找到现有特征集 'all_available_features' (ID: {set_id})")
        else:
            print("未找到特征集 'all_available_features'，正在创建...")
            set_id = create_all_features_featureset()
        
        conn.close()
        return set_id
    
    def get_dataset_recordings(self, dataset_id):
        """获取指定数据集的所有记录"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        c.execute("""
            SELECT id, filename, subject_id, duration, channels, sampling_rate
            FROM recordings 
            WHERE dataset_id = ?
            ORDER BY id
        """, (dataset_id,))
        
        recordings = []
        for row in c.fetchall():
            recordings.append({
                "id": row[0],
                "filename": row[1],
                "subject_id": row[2],
                "duration": row[3],
                "channels": row[4],
                "sampling_rate": row[5]
            })
        
        conn.close()
        return recordings
    
    def validate_feature_extraction(self, dataset_id, max_recordings=None):
        """验证特征提取功能"""
        print("\n" + "=" * 60)
        print("验证特征提取功能")
        print("=" * 60)
        
        # 获取特征集ID
        feature_set_id = self.get_all_features_featureset_id()
        if not feature_set_id:
            print("✗ 无法获取特征集ID")
            return None
        
        # 获取数据集记录
        recordings = self.get_dataset_recordings(dataset_id)
        if not recordings:
            print(f"✗ 数据集 {dataset_id} 中没有找到记录")
            return None
        
        print(f"找到 {len(recordings)} 条记录")
        
        # 限制处理的记录数量
        if max_recordings and len(recordings) > max_recordings:
            recordings = recordings[:max_recordings]
            print(f"限制处理前 {max_recordings} 条记录")
        
        # 获取特征定义
        try:
            fxdefs = load_fxdefs_for_set(feature_set_id)
            print(f"特征集包含 {len(fxdefs)} 个特征定义")
        except Exception as e:
            print(f"✗ 加载特征定义失败: {e}")
            return None
        
        extraction_results = {
            "feature_set_id": feature_set_id,
            "dataset_id": dataset_id,
            "recordings_processed": 0,
            "recordings_successful": 0,
            "recordings_failed": 0,
            "total_features": len(fxdefs),
            "recording_results": []
        }
        
        # 处理每个记录
        for i, recording in enumerate(recordings, 1):
            print(f"\n处理记录 {i}/{len(recordings)}: {recording['filename']}")
            print(f"  ID: {recording['id']}, 时长: {recording['duration']:.1f}s, 通道: {recording['channels']}")
            
            start_time = time.time()
            
            try:
                # 运行特征提取
                results = run_feature_set(feature_set_id, recording['id'])
                
                # 统计结果
                successful_features = sum(1 for r in results.values() if r["value"] is not None)
                failed_features = len(results) - successful_features
                
                processing_time = time.time() - start_time
                
                recording_result = {
                    "recording_id": recording['id'],
                    "filename": recording['filename'],
                    "status": "success",
                    "processing_time": processing_time,
                    "successful_features": successful_features,
                    "failed_features": failed_features,
                    "error": None
                }
                
                extraction_results["recordings_successful"] += 1
                print(f"  ✓ 成功提取 {successful_features}/{len(results)} 个特征 ({processing_time:.2f}s)")
                
            except Exception as e:
                processing_time = time.time() - start_time
                recording_result = {
                    "recording_id": recording['id'],
                    "filename": recording['filename'],
                    "status": "failed",
                    "processing_time": processing_time,
                    "successful_features": 0,
                    "failed_features": len(fxdefs),
                    "error": str(e)
                }
                
                extraction_results["recordings_failed"] += 1
                print(f"  ✗ 特征提取失败: {e}")
            
            extraction_results["recording_results"].append(recording_result)
            extraction_results["recordings_processed"] += 1
        
        # 生成统计信息
        total_successful = sum(r["successful_features"] for r in extraction_results["recording_results"])
        total_failed = sum(r["failed_features"] for r in extraction_results["recording_results"])
        
        extraction_results["summary"] = {
            "total_successful_features": total_successful,
            "total_failed_features": total_failed,
            "success_rate": extraction_results["recordings_successful"] / extraction_results["recordings_processed"] if extraction_results["recordings_processed"] > 0 else 0,
            "feature_success_rate": total_successful / (total_successful + total_failed) if (total_successful + total_failed) > 0 else 0
        }
        
        print(f"\n特征提取验证结果:")
        print(f"  处理记录: {extraction_results['recordings_processed']}")
        print(f"  成功记录: {extraction_results['recordings_successful']}")
        print(f"  失败记录: {extraction_results['recordings_failed']}")
        print(f"  记录成功率: {extraction_results['summary']['success_rate']:.2%}")
        print(f"  特征成功率: {extraction_results['summary']['feature_success_rate']:.2%}")
        
        self.validation_results["extraction_validation"] = extraction_results
        return extraction_results
    
    def generate_validation_report(self, output_file=None):
        """生成验证报告"""
        print("\n" + "=" * 60)
        print("生成验证报告")
        print("=" * 60)
        
        # 添加结束时间
        self.validation_results["end_time"] = datetime.now().isoformat()
        
        # 生成失败特征统计
        try:
            failed_stats = get_failed_features_stats()
            self.validation_results["failed_features_stats"] = failed_stats
            print("✓ 获取失败特征统计")
        except Exception as e:
            print(f"✗ 获取失败特征统计失败: {e}")
        
        # 保存报告
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"feature_validation_report_{timestamp}.json"
        
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(self.validation_results, f, indent=2, ensure_ascii=False)
        
        print(f"✓ 验证报告已保存到: {output_file}")
        
        # 打印摘要
        self.print_summary()
        
        return output_file
    
    def print_summary(self):
        """打印验证摘要"""
        print("\n" + "=" * 60)
        print("验证摘要")
        print("=" * 60)
        
        # 特征函数验证摘要
        feature_validation = self.validation_results.get("feature_validation", {})
        if feature_validation:
            valid_funcs = sum(1 for r in feature_validation.values() if r["status"] == "valid")
            error_funcs = sum(1 for r in feature_validation.values() if r["status"] == "error")
            print(f"特征函数验证:")
            print(f"  ✓ 有效函数: {valid_funcs}")
            print(f"  ✗ 错误函数: {error_funcs}")
        
        # 特征提取验证摘要
        extraction_validation = self.validation_results.get("extraction_validation", {})
        if extraction_validation:
            summary = extraction_validation.get("summary", {})
            print(f"特征提取验证:")
            print(f"  处理记录: {extraction_validation.get('recordings_processed', 0)}")
            print(f"  ✓ 成功记录: {extraction_validation.get('recordings_successful', 0)}")
            print(f"  ✗ 失败记录: {extraction_validation.get('recordings_failed', 0)}")
            print(f"  📈 记录成功率: {summary.get('success_rate', 0):.2%}")
            print(f"  📈 特征成功率: {summary.get('feature_success_rate', 0):.2%}")
        
        # 失败特征统计
        failed_stats = self.validation_results.get("failed_features_stats", {})
        if failed_stats:
            print(f"失败特征统计:")
            print(f"  📉 总失败特征: {failed_stats.get('total_failed', 0)}")
            if failed_stats.get('common_errors'):
                print(f"  🔍 常见错误:")
                for error, count in list(failed_stats['common_errors'].items())[:3]:
                    print(f"    - {error[:50]}... ({count}次)")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="EEG特征提取验证工具")
    parser.add_argument("dataset_id", type=int, help="数据集ID")
    parser.add_argument("--max_recordings", type=int, default=None, 
                       help="最大处理记录数（默认处理所有记录）")
    parser.add_argument("--output", type=str, default=None,
                       help="输出报告文件名（默认自动生成）")
    
    args = parser.parse_args()
    
    print("EEG特征提取验证工具")
    print("=" * 60)
    print(f"数据集ID: {args.dataset_id}")
    print(f"最大记录数: {args.max_recordings or '全部'}")
    
    # 创建验证器
    validator = FeatureExtractionValidator()
    
    # 验证特征函数
    validator.validate_feature_functions()
    
    # 验证特征提取
    validator.validate_feature_extraction(args.dataset_id, args.max_recordings)
    
    # 生成报告
    report_file = validator.generate_validation_report(args.output)
    
    print(f"\n验证完成！详细报告已保存到: {report_file}")

if __name__ == "__main__":
    main() 