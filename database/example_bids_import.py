#!/usr/bin/env python3
"""
BIDS数据集导入示例

这个文件展示了如何使用BIDSImporter类来导入BIDS格式的EEG数据集。
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.import_bids_dataset import BIDSImporter, import_bids_dataset
from logging_config import logger

def example_import_harvard_bids():
    """
    示例：导入Harvard BIDS数据集
    """
    # Harvard BIDS数据集路径
    harvard_bids_path = "/rds/general/user/zj724/ephemeral"
    
    if not os.path.exists(harvard_bids_path):
        logger.error(f"Harvard BIDS路径不存在: {harvard_bids_path}")
        return
    
    try:
        # 方法1：使用便捷函数
        logger.info("使用便捷函数导入Harvard BIDS数据集...")
        dataset_id = import_bids_dataset(
            bids_root=harvard_bids_path,
            dataset_name="Harvard_BIDS_1000",
            max_import_count=100  # 限制导入100条记录
        )
        logger.info(f"Harvard BIDS数据集导入成功，数据集ID: {dataset_id}")
        
    except Exception as e:
        logger.error(f"Harvard BIDS数据集导入失败: {e}")

def example_import_custom_bids():
    """
    示例：导入自定义BIDS数据集
    """
    # 自定义BIDS数据集路径
    custom_bids_path = "/path/to/your/bids/dataset"
    
    if not os.path.exists(custom_bids_path):
        logger.warning(f"自定义BIDS路径不存在: {custom_bids_path}")
        logger.info("请修改custom_bids_path变量指向您的BIDS数据集")
        return
    
    try:
        # 方法2：使用BIDSImporter类
        logger.info("使用BIDSImporter类导入自定义BIDS数据集...")
        
        importer = BIDSImporter(custom_bids_path)
        
        # 查看数据集信息
        logger.info(f"数据集名称: {importer.dataset_description.get('Name', 'Unknown')}")
        logger.info(f"数据集描述: {importer.dataset_description.get('Description', 'No description')}")
        logger.info(f"参与者数量: {len(importer.participants_info)}")
        
        # 导入数据集
        dataset_id = importer.import_dataset(
            dataset_name="Custom_BIDS_Dataset",
            max_import_count=50
        )
        
        logger.info(f"自定义BIDS数据集导入成功，数据集ID: {dataset_id}")
        
    except Exception as e:
        logger.error(f"自定义BIDS数据集导入失败: {e}")

def example_validate_bids_structure():
    """
    示例：验证BIDS目录结构
    """
    # 测试路径
    test_paths = [
        "/rds/general/user/zj724/ephemeral",
        "/path/to/test/bids/dataset"
    ]
    
    for test_path in test_paths:
        if os.path.exists(test_path):
            logger.info(f"验证BIDS结构: {test_path}")
            try:
                importer = BIDSImporter(test_path)
                logger.info(f"✓ {test_path} 是有效的BIDS数据集")
                
                # 显示数据集信息
                logger.info(f"  数据集名称: {importer.dataset_description.get('Name', 'Unknown')}")
                logger.info(f"  参与者数量: {len(importer.participants_info)}")
                
                # 检查主题目录
                subject_dirs = [d for d in importer.bids_root.iterdir() 
                              if d.is_dir() and d.name.startswith("sub-")]
                logger.info(f"  主题目录数量: {len(subject_dirs)}")
                
                # 检查第一个主题的EEG文件
                if subject_dirs:
                    first_subject = subject_dirs[0]
                    eeg_files = importer._find_eeg_files(first_subject)
                    logger.info(f"  第一个主题({first_subject.name})的EEG文件数量: {len(eeg_files)}")
                
            except Exception as e:
                logger.error(f"✗ {test_path} 不是有效的BIDS数据集: {e}")
        else:
            logger.warning(f"路径不存在: {test_path}")

def main():
    """主函数"""
    logger.info("=== BIDS数据集导入示例 ===")
    
    # 示例1：验证BIDS结构
    logger.info("\n1. 验证BIDS目录结构")
    example_validate_bids_structure()
    
    # 示例2：导入Harvard BIDS数据集
    logger.info("\n2. 导入Harvard BIDS数据集")
    example_import_harvard_bids()
    
    # 示例3：导入自定义BIDS数据集
    logger.info("\n3. 导入自定义BIDS数据集")
    example_import_custom_bids()
    
    logger.info("\n=== 示例完成 ===")

if __name__ == "__main__":
    main()


