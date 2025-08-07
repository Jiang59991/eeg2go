"""
创建 minimal_harvard 数据集（小内存版本）

这个脚本从现有的 Harvard_S0001_demo 数据集中筛选出内存较小的30条记录，
创建一个新的 minimal_harvard 数据集用于快速测试。
参考 import_harvard_demo.py 的内存检查逻辑。
"""

import os
import sqlite3
import mne
from datetime import datetime
from logging_config import logger

DB_PATH = os.path.join(os.path.dirname(__file__), "eeg2go.db")
SOURCE_DATASET_ID = 1  # Harvard_S0001_demo
TARGET_DATASET_NAME = "minimal_harvard"
RECORDINGS_TO_COPY = 30
MAX_MEMORY_GB = 2  # 设置更小的内存限制，只选择小文件

# 设置MNE日志级别
mne.set_log_level('WARNING')


def estimate_file_memory_size(file_path):
    """
    估算文件的内存使用量（MB）
    
    Args:
        file_path: EDF文件路径
    
    Returns:
        float: 估算的内存使用量（MB），如果无法读取则返回None
    """
    try:
        raw = mne.io.read_raw_edf(file_path, preload=False, verbose='ERROR')
        channels = len(raw.info['ch_names'])
        n_times = raw.n_times
        
        # 估算内存使用量：channels * n_times * 8 bytes (double precision)
        estimated_mb = (channels * n_times * 8) / (1024 * 1024)
        return estimated_mb
        
    except Exception as e:
        logger.warning(f"无法读取文件 {file_path}: {e}")
        return None


def get_small_files_from_dataset(dataset_id, max_memory_gb=2, target_count=30):
    """
    从数据集中筛选出内存较小的文件
    
    Args:
        dataset_id: 源数据集ID
        max_memory_gb: 最大内存限制（GB）
        target_count: 目标文件数量
    
    Returns:
        list: 筛选出的记录列表
    """
    logger.info(f"从数据集 {dataset_id} 中筛选内存小于 {max_memory_gb}GB 的文件...")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # 获取所有记录
        c.execute("""
            SELECT id, subject_id, filename, path, duration, channels, sampling_rate
            FROM recordings 
            WHERE dataset_id = ?
        """, (dataset_id,))
        
        all_recordings = c.fetchall()
        logger.info(f"源数据集共有 {len(all_recordings)} 条记录")
        
        # 筛选小文件
        small_files = []
        max_memory_mb = max_memory_gb * 1024
        
        for recording in all_recordings:
            rec_id, subject_id, filename, path, duration, channels, sampling_rate = recording
            
            # 检查文件是否存在
            if not os.path.exists(path):
                logger.warning(f"文件不存在: {path}")
                continue
            
            # 估算内存使用量
            estimated_mb = estimate_file_memory_size(path)
            
            if estimated_mb is None:
                continue
            
            if estimated_mb <= max_memory_mb:
                small_files.append({
                    'recording': recording,
                    'memory_mb': estimated_mb
                })
                logger.info(f"符合条件: {filename} - {estimated_mb:.1f}MB")
            else:
                logger.debug(f"文件过大: {filename} - {estimated_mb:.1f}MB > {max_memory_mb}MB")
        
        # 按内存大小排序，选择最小的文件
        small_files.sort(key=lambda x: x['memory_mb'])
        
        # 选择前N个最小的文件
        selected_files = small_files[:target_count]
        
        logger.info(f"筛选完成: 找到 {len(small_files)} 个小文件，选择最小的 {len(selected_files)} 个")
        
        if len(selected_files) < target_count:
            logger.warning(f"只找到 {len(selected_files)} 个符合条件的文件，少于目标数量 {target_count}")
        
        return selected_files
        
    finally:
        conn.close()


def create_minimal_harvard_dataset_small():
    """创建小内存版本的 minimal_harvard 数据集"""
    logger.info(f"开始创建小内存版本的 {TARGET_DATASET_NAME} 数据集...")
    
    # 1. 筛选小文件
    small_files = get_small_files_from_dataset(
        SOURCE_DATASET_ID, 
        max_memory_gb=MAX_MEMORY_GB, 
        target_count=RECORDINGS_TO_COPY
    )
    
    if not small_files:
        logger.error("没有找到符合条件的文件")
        return None
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # 2. 检查数据集是否已存在，如果存在则删除
        c.execute("SELECT id FROM datasets WHERE name = ?", (TARGET_DATASET_NAME,))
        existing_dataset = c.fetchone()
        if existing_dataset:
            logger.info(f"删除已存在的数据集: {TARGET_DATASET_NAME}")
            # 删除相关的recording_metadata
            c.execute("""
                DELETE FROM recording_metadata 
                WHERE recording_id IN (
                    SELECT id FROM recordings WHERE dataset_id = ?
                )
            """, (existing_dataset[0],))
            # 删除recordings
            c.execute("DELETE FROM recordings WHERE dataset_id = ?", (existing_dataset[0],))
            # 删除dataset
            c.execute("DELETE FROM datasets WHERE id = ?", (existing_dataset[0],))
        
        # 3. 创建新的数据集
        logger.info(f"创建数据集: {TARGET_DATASET_NAME}")
        c.execute("""
            INSERT INTO datasets (name, description, source_type, path) 
            VALUES (?, ?, ?, ?)
        """, (
            TARGET_DATASET_NAME,
            f"Minimal Harvard dataset (small files) for fast testing - {len(small_files)} recordings, max {MAX_MEMORY_GB}GB each",
            "edf",
            "data/harvard_EEG"
        ))
        target_dataset_id = c.lastrowid
        logger.info(f"数据集创建成功，ID: {target_dataset_id}")
        
        # 4. 复制选中的记录到新数据集
        copied_recordings = 0
        total_memory = 0
        
        for file_info in small_files:
            recording = file_info['recording']
            memory_mb = file_info['memory_mb']
            
            rec_id, subject_id, filename, path, duration, channels, sampling_rate = recording
            
            # 复制recording记录
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
            total_memory += memory_mb
            logger.info(f"复制记录 {copied_recordings}/{len(small_files)}: {filename} ({memory_mb:.1f}MB)")
        
        # 5. 提交事务
        conn.commit()
        
        # 6. 验证结果
        c.execute("SELECT COUNT(*) FROM recordings WHERE dataset_id = ?", (target_dataset_id,))
        final_count = c.fetchone()[0]
        
        c.execute("SELECT COUNT(DISTINCT subject_id) FROM recordings WHERE dataset_id = ?", (target_dataset_id,))
        subject_count = c.fetchone()[0]
        
        logger.info(f"✅ 小内存版本 minimal_harvard 数据集创建完成！")
        logger.info(f"   数据集ID: {target_dataset_id}")
        logger.info(f"   记录数量: {final_count}")
        logger.info(f"   受试者数量: {subject_count}")
        logger.info(f"   总内存估算: {total_memory:.1f}MB ({total_memory/1024:.2f}GB)")
        logger.info(f"   平均文件大小: {total_memory/final_count:.1f}MB")
        logger.info(f"   最大文件限制: {MAX_MEMORY_GB}GB")
        
        return target_dataset_id
        
    except Exception as e:
        logger.error(f"创建数据集失败: {e}")
        conn.rollback()
        raise
    finally:
        conn.close()


def verify_minimal_dataset_small(dataset_id):
    """验证小内存数据集"""
    logger.info(f"验证小内存数据集 {dataset_id}...")
    
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    try:
        # 检查数据集信息
        c.execute("SELECT name, description FROM datasets WHERE id = ?", (dataset_id,))
        dataset_info = c.fetchone()
        if dataset_info:
            logger.info(f"数据集名称: {dataset_info[0]}")
            logger.info(f"数据集描述: {dataset_info[1]}")
        
        # 检查记录数量和内存使用情况
        c.execute("""
            SELECT r.filename, r.path, r.duration, r.channels, r.sampling_rate
            FROM recordings r
            WHERE r.dataset_id = ?
        """, (dataset_id,))
        
        recordings = c.fetchall()
        logger.info(f"记录数量: {len(recordings)}")
        
        # 计算总内存使用量
        total_memory = 0
        logger.info("文件内存使用情况:")
        for filename, path, duration, channels, sampling_rate in recordings:
            if os.path.exists(path):
                memory_mb = estimate_file_memory_size(path)
                if memory_mb:
                    total_memory += memory_mb
                    logger.info(f"  {filename}: {memory_mb:.1f}MB ({channels}ch, {duration:.1f}s)")
        
        logger.info(f"总内存估算: {total_memory:.1f}MB ({total_memory/1024:.2f}GB)")
        logger.info(f"平均文件大小: {total_memory/len(recordings):.1f}MB")
        
        return {
            'recording_count': len(recordings),
            'total_memory_mb': total_memory,
            'avg_memory_mb': total_memory / len(recordings) if recordings else 0
        }
        
    finally:
        conn.close()


def main():
    """主函数"""
    logger.info("=" * 60)
    logger.info("创建小内存版本的 minimal_harvard 数据集")
    logger.info("=" * 60)
    logger.info(f"目标文件数量: {RECORDINGS_TO_COPY}")
    logger.info(f"最大文件大小: {MAX_MEMORY_GB}GB")
    logger.info("=" * 60)
    
    try:
        # 创建数据集
        dataset_id = create_minimal_harvard_dataset_small()
        
        if dataset_id is None:
            logger.error("数据集创建失败")
            return False
        
        # 验证数据集
        logger.info("\n" + "=" * 60)
        logger.info("验证数据集")
        logger.info("=" * 60)
        stats = verify_minimal_dataset_small(dataset_id)
        
        logger.info("\n" + "=" * 60)
        logger.info("✅ 小内存数据集创建成功！")
        logger.info("=" * 60)
        logger.info(f"数据集ID: {dataset_id}")
        logger.info(f"数据集名称: {TARGET_DATASET_NAME}")
        logger.info(f"记录数量: {stats['recording_count']}")
        logger.info(f"总内存: {stats['total_memory_mb']:.1f}MB ({stats['total_memory_mb']/1024:.2f}GB)")
        logger.info(f"平均文件大小: {stats['avg_memory_mb']:.1f}MB")
        logger.info("\n现在可以使用这个优化的小内存数据集进行快速测试了！")
        
    except Exception as e:
        logger.error(f"❌ 创建数据集失败: {e}")
        return False
    
    return True


if __name__ == "__main__":
    main() 