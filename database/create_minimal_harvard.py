"""
创建 minimal_harvard 数据集

这个脚本从现有的 Harvard_S0001_demo 数据集中复制30条记录，
创建一个新的 minimal_harvard 数据集用于快速测试。
"""

import os
import sqlite3
import random
from datetime import datetime
from logging_config import logger

DB_PATH = os.path.join(os.path.dirname(__file__), "eeg2go.db")
SOURCE_DATASET_ID = 1  # Harvard_S0001_demo
TARGET_DATASET_NAME = "minimal_harvard"
RECORDINGS_TO_COPY = 30


def create_minimal_harvard_dataset():
    """创建 minimal_harvard 数据集"""
    logger.info(f"开始创建 {TARGET_DATASET_NAME} 数据集...")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # 1. 创建新的数据集
        logger.info(f"创建数据集: {TARGET_DATASET_NAME}")
        c.execute("""
            INSERT INTO datasets (name, description, source_type, path) 
            VALUES (?, ?, ?, ?)
        """, (
            TARGET_DATASET_NAME,
            f"Minimal Harvard dataset for fast testing - {RECORDINGS_TO_COPY} recordings",
            "edf",
            "data/harvard_EEG"  # 使用相同的路径
        ))
        target_dataset_id = c.lastrowid
        logger.info(f"数据集创建成功，ID: {target_dataset_id}")
        
        # 2. 从源数据集随机选择30条记录
        logger.info(f"从源数据集选择 {RECORDINGS_TO_COPY} 条记录...")
        c.execute("""
            SELECT id, subject_id, filename, path, duration, channels, sampling_rate
            FROM recordings 
            WHERE dataset_id = ?
            ORDER BY RANDOM()
            LIMIT ?
        """, (SOURCE_DATASET_ID, RECORDINGS_TO_COPY))
        
        selected_recordings = c.fetchall()
        logger.info(f"选择了 {len(selected_recordings)} 条记录")
        
        if len(selected_recordings) < RECORDINGS_TO_COPY:
            logger.warning(f"源数据集只有 {len(selected_recordings)} 条记录，少于请求的 {RECORDINGS_TO_COPY} 条")
        
        # 3. 复制选中的记录到新数据集
        copied_recordings = 0
        copied_subjects = set()
        
        for recording in selected_recordings:
            rec_id, subject_id, filename, path, duration, channels, sampling_rate = recording
            
            # 直接复制recording记录，不创建新的subject记录
            c.execute("""
                INSERT INTO recordings (dataset_id, subject_id, filename, path, 
                                      duration, channels, sampling_rate)
                VALUES (?, ?, ?, ?, ?, ?, ?)
            """, (target_dataset_id, subject_id, filename, path, 
                  duration, channels, sampling_rate))
            
            new_recording_id = c.lastrowid
            
            # 复制recording_metadata（如果存在）
            c.execute("""
                SELECT age_days, sex, start_time, end_time, seizure, spindles, 
                       status, normal, abnormal
                FROM recording_metadata 
                WHERE recording_id = ?
            """, (rec_id,))
            
            metadata = c.fetchone()
            if metadata:
                c.execute("""
                    INSERT INTO recording_metadata (
                        recording_id, age_days, sex, start_time, end_time, 
                        seizure, spindles, status, normal, abnormal
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (new_recording_id,) + metadata)
            
            copied_recordings += 1
            logger.info(f"复制记录 {copied_recordings}/{len(selected_recordings)}: {filename}")
        
        # 4. 提交事务
        conn.commit()
        
        # 5. 验证结果
        c.execute("SELECT COUNT(*) FROM recordings WHERE dataset_id = ?", (target_dataset_id,))
        final_count = c.fetchone()[0]
        
        c.execute("SELECT COUNT(DISTINCT subject_id) FROM recordings WHERE dataset_id = ?", (target_dataset_id,))
        subject_count = c.fetchone()[0]
        
        logger.info(f"✅ minimal_harvard 数据集创建完成！")
        logger.info(f"   数据集ID: {target_dataset_id}")
        logger.info(f"   记录数量: {final_count}")
        logger.info(f"   受试者数量: {subject_count}")
        logger.info(f"   源数据集: Harvard_S0001_demo (ID: {SOURCE_DATASET_ID})")
        
        return target_dataset_id
        
    except Exception as e:
        logger.error(f"创建数据集失败: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def verify_minimal_dataset(dataset_id):
    """验证minimal数据集"""
    logger.info(f"验证数据集 {dataset_id}...")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # 检查数据集信息
        c.execute("SELECT name, description FROM datasets WHERE id = ?", (dataset_id,))
        dataset_info = c.fetchone()
        if dataset_info:
            logger.info(f"数据集名称: {dataset_info[0]}")
            logger.info(f"数据集描述: {dataset_info[1]}")
        
        # 检查记录数量
        c.execute("SELECT COUNT(*) FROM recordings WHERE dataset_id = ?", (dataset_id,))
        recording_count = c.fetchone()[0]
        logger.info(f"记录数量: {recording_count}")
        
        # 检查受试者数量
        c.execute("SELECT COUNT(DISTINCT subject_id) FROM recordings WHERE dataset_id = ?", (dataset_id,))
        subject_count = c.fetchone()[0]
        logger.info(f"受试者数量: {subject_count}")
        
        # 检查元数据数量
        c.execute("""
            SELECT COUNT(*) FROM recording_metadata rm
            JOIN recordings r ON rm.recording_id = r.id
            WHERE r.dataset_id = ?
        """, (dataset_id,))
        metadata_count = c.fetchone()[0]
        logger.info(f"元数据记录数量: {metadata_count}")
        
        # 显示一些示例记录
        c.execute("""
            SELECT r.filename, r.subject_id, r.duration, r.channels, r.sampling_rate
            FROM recordings r
            WHERE r.dataset_id = ?
            LIMIT 5
        """, (dataset_id,))
        
        sample_records = c.fetchall()
        logger.info("示例记录:")
        for record in sample_records:
            filename, subject_id, duration, channels, sampling_rate = record
            logger.info(f"  {filename} | {subject_id} | {duration:.1f}s | {channels}ch | {sampling_rate}Hz")
        
        return {
            'recording_count': recording_count,
            'subject_count': subject_count,
            'metadata_count': metadata_count
        }
        
    finally:
        conn.close()


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("创建 minimal_harvard 数据集")
    logger.info("=" * 60)
    
    try:
        # 创建数据集
        dataset_id = create_minimal_harvard_dataset()
        
        # 验证数据集
        logger.info("\n" + "=" * 60)
        logger.info("验证数据集")
        logger.info("=" * 60)
        stats = verify_minimal_dataset(dataset_id)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ 数据集创建成功！")
        logger.info("=" * 60)
        logger.info(f"数据集ID: {dataset_id}")
        logger.info(f"数据集名称: {TARGET_DATASET_NAME}")
        logger.info(f"记录数量: {stats['recording_count']}")
        logger.info(f"受试者数量: {stats['subject_count']}")
        logger.info(f"元数据记录: {stats['metadata_count']}")
        logger.info("\n现在可以使用这个数据集进行快速测试了！")
        
    except Exception as e:
        logger.error(f"❌ 创建数据集失败: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main() 