#!/usr/bin/env python3
"""
测试Harvard BIDS导入器
"""

import os
import sys
from pathlib import Path

# 添加项目根目录到Python路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from database.import_harvard_bids import HarvardBIDSImporter
from logging_config import logger

def test_harvard_bids_importer():
    """测试Harvard BIDS导入器"""
    
    # Harvard BIDS数据集路径
    harvard_bids_path = "/rds/general/user/zj724/ephemeral/S0001"
    
    if not os.path.exists(harvard_bids_path):
        logger.error(f"Harvard BIDS路径不存在: {harvard_bids_path}")
        return False
    
    try:
        logger.info("=== 测试Harvard BIDS导入器 ===")
        
        # 创建导入器实例
        importer = HarvardBIDSImporter(harvard_bids_path)
        
        # 显示数据集信息
        logger.info(f"数据集路径: {harvard_bids_path}")
        logger.info(f"主题数量: {len(importer.subjects_info)}")
        
        # 显示前几个主题的信息
        for i, (subject_id, subject_info) in enumerate(importer.subjects_info.items()):
            if i >= 3:  # 只显示前3个
                break
            logger.info(f"主题 {i+1}: {subject_id}")
            logger.info(f"  会话数量: {len(subject_info['sessions'])}")
            logger.info(f"  EEG文件数量: {len(subject_info['eeg_files'])}")
            
            # 显示第一个EEG文件的信息
            if subject_info['eeg_files']:
                first_eeg = subject_info['eeg_files'][0]
                logger.info(f"  第一个EEG文件: {first_eeg['file_name']}")
                logger.info(f"    采样率: {first_eeg['sfreq']} Hz")
                logger.info(f"    通道数: {first_eeg['channels']}")
                logger.info(f"    时长: {first_eeg['duration']:.1f} 秒")
        
        logger.info("=== 测试完成 ===")
        return True
        
    except Exception as e:
        logger.error(f"测试失败: {e}")
        return False

def test_small_import():
    """测试小规模导入"""
    
    harvard_bids_path = "/rds/general/user/zj724/ephemeral/S0001"
    
    if not os.path.exists(harvard_bids_path):
        logger.error(f"Harvard BIDS路径不存在: {harvard_bids_path}")
        return False
    
    try:
        logger.info("=== 测试小规模导入 ===")
        
        # 导入少量记录进行测试
        from database.import_harvard_bids import import_harvard_bids_dataset
        
        dataset_id = import_harvard_bids_dataset(
            bids_root=harvard_bids_path,
            dataset_name="Harvard_BIDS_Test",
            max_import_count=5  # 只导入5条记录
        )
        
        logger.info(f"测试导入成功，数据集ID: {dataset_id}")
        return True
        
    except Exception as e:
        logger.error(f"测试导入失败: {e}")
        return False

def main():
    """主函数"""
    logger.info("开始测试Harvard BIDS导入器...")
    
    # 测试1：验证导入器
    success1 = test_harvard_bids_importer()
    
    if success1:
        # 测试2：小规模导入
        success2 = test_small_import()
        
        if success2:
            logger.info("所有测试通过！")
        else:
            logger.error("导入测试失败")
    else:
        logger.error("导入器验证失败")
    
    logger.info("测试完成")

if __name__ == "__main__":
    main()


