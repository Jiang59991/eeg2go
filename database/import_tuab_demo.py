#!/usr/bin/env python3
import os
import sqlite3
import mne
import re
from logging_config import logger

DATA_DIR = "/rds/general/user/zj724/ephemeral/TUAB_v3_0_1_edf"
DB_PATH = os.path.join(os.path.dirname(__file__), "eeg2go.db")
mne.set_log_level('WARNING')

def extract_subject_info_from_path(file_path):
    """
    从TUAB v3.0.1文件路径中提取subject信息
    文件路径格式: eval/abnormal/01_tcp_ar/aaaaabdo_s003_t000.edf
    """
    path_parts = file_path.split('/')
    if len(path_parts) >= 4:
        split_type = path_parts[0]  # eval 或 train
        label = path_parts[1]       # normal 或 abnormal
        config = path_parts[2]      # 01_tcp_ar
        filename = path_parts[3]    # aaaaabdo_s003_t000.edf
        
        # 解析文件名: aaaaabdo_s003_t000.edf
        # subject_id: aaaaabdo
        # session: s003
        # token: t000
        filename_parts = filename.replace('.edf', '').split('_')
        if len(filename_parts) >= 3:
            subject_id = filename_parts[0]
            session = filename_parts[1]
            token = filename_parts[2]
            
            # 创建唯一的recording_id: subject_id_session_token
            recording_id = f"{subject_id}_{session}_{token}"
            
            return {
                'subject_id': subject_id,
                'session': session,
                'token': token,
                'recording_id': recording_id,
                'split_type': split_type,
                'label': label,
                'config': config,
                'filename': filename
            }
    return None

def import_demo_data(data_dir, max_files=10):
    """
    导入demo数据集，只包含指定数量的文件
    """
    logger.info(f"Importing demo dataset from {data_dir} (max {max_files} files)")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 1. 确保 demo 数据集存在
    c.execute("SELECT id FROM datasets WHERE name = ?", ("demo",))
    row = c.fetchone()
    if row is None:
        c.execute("INSERT INTO datasets (name, description, source_type, path) VALUES (?, ?, ?, ?)",
                  ("demo", "Demo dataset with 10 recordings from TUAB v3.0.1", "edf", data_dir))
        dataset_id = c.lastrowid
        logger.info(f"Created demo dataset with ID: {dataset_id}")
    else:
        dataset_id = row[0]
        logger.info(f"Using existing demo dataset with ID: {dataset_id}")

    # 2. 遍历所有 .edf 文件，但只处理前max_files个
    edf_files = []
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith(".edf"):
                fpath = os.path.join(root, fname)
                # 获取相对于data_dir的路径
                rel_path = os.path.relpath(fpath, data_dir)
                edf_files.append((fpath, rel_path))
                if len(edf_files) >= max_files:
                    break
        if len(edf_files) >= max_files:
            break
    
    logger.info(f"Found {len(edf_files)} EDF files (limited to {max_files})")
    
    imported_count = 0
    skipped_count = 0
    
    for fpath, rel_path in edf_files:
        info = extract_subject_info_from_path(rel_path)
        
        if not info:
            logger.warning(f"Could not extract subject info from path: {rel_path}")
            skipped_count += 1
            continue

        # 检查是否已经导入
        c.execute("SELECT id FROM recordings WHERE filename = ? AND path = ? AND dataset_id = ?", 
                 (info['filename'], fpath, dataset_id))
        if c.fetchone():
            logger.debug(f"Already imported: {info['filename']}")
            skipped_count += 1
            continue

        # 为demo数据集创建唯一的subject_id
        demo_subject_id = f"demo_{info['subject_id']}"
        
        # 插入或查询 subject
        c.execute("SELECT subject_id FROM subjects WHERE subject_id = ? AND dataset_id = ?", 
                 (demo_subject_id, dataset_id))
        if not c.fetchone():
            c.execute("INSERT INTO subjects (subject_id, dataset_id) VALUES (?, ?)", 
                     (demo_subject_id, dataset_id))
            logger.debug(f"Created new subject: {demo_subject_id}")

        try:
            # 读取EDF文件获取基本信息
            raw = mne.io.read_raw_edf(fpath, preload=False, verbose='ERROR')
            sfreq = raw.info['sfreq']
            channels = len(raw.info['ch_names'])
            duration = raw.n_times / sfreq
            
            # 获取通道名称
            ch_names = raw.info['ch_names']
            
        except Exception as e:
            logger.error(f"Failed to read {fpath}: {e}")
            sfreq = channels = duration = None
            ch_names = []

        # 插入recording记录
        c.execute("""
            INSERT INTO recordings (
                dataset_id, subject_id, filename, path, duration, 
                channels, sampling_rate, recording_type, placement_scheme
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (dataset_id, demo_subject_id, info['filename'], fpath, duration, 
              channels, sfreq, "continuous", "10-20"))

        # 插入recording_metadata记录，包含normal/abnormal标签
        recording_id = c.lastrowid
        c.execute("""
            INSERT INTO recording_metadata (
                recording_id, status, normal, abnormal
            )
            VALUES (?, ?, ?, ?)
        """, (recording_id, info['label'], 
              '1' if info['label'] == 'normal' else '0',
              '1' if info['label'] == 'abnormal' else '0'))

        imported_count += 1
        
        logger.info(f"Imported {imported_count}/{max_files}: {info['filename']} | Subject: {demo_subject_id} | Label: {info['label']} | Duration: {duration:.1f}s" if sfreq else f"Imported {imported_count}/{max_files} (basic): {info['filename']}")

    conn.commit()
    conn.close()
    
    logger.info(f"Demo import completed. Imported: {imported_count}, Skipped: {skipped_count}")
    return dataset_id

def get_demo_statistics():
    """
    获取demo数据集统计信息
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    
    # 获取demo数据集ID
    c.execute("SELECT id FROM datasets WHERE name = ?", ("demo",))
    row = c.fetchone()
    if not row:
        logger.error("Demo dataset not found.")
        return None
    
    dataset_id = row[0]
    
    # 统计标签分布
    c.execute("""
        SELECT rm.abnormal, COUNT(*) as count
        FROM recordings r
        JOIN recording_metadata rm ON r.id = rm.recording_id
        WHERE r.dataset_id = ?
        GROUP BY rm.abnormal
    """, (dataset_id,))
    
    stats = {}
    for row in c.fetchall():
        is_abnormal = bool(row[0])
        count = row[1]
        stats['abnormal' if is_abnormal else 'normal'] = count
    
    # 获取所有recordings的详细信息
    c.execute("""
        SELECT r.id, r.subject_id, r.filename, r.duration, r.channels, r.sampling_rate, rm.abnormal
        FROM recordings r
        JOIN recording_metadata rm ON r.id = rm.recording_id
        WHERE r.dataset_id = ?
        ORDER BY r.id
    """, (dataset_id,))
    
    recordings = []
    for row in c.fetchall():
        recordings.append({
            'id': row[0],
            'subject_id': row[1],
            'filename': row[2],
            'duration': row[3],
            'channels': row[4],
            'sampling_rate': row[5],
            'is_abnormal': bool(row[6])
        })
    
    conn.close()
    
    return {
        'overall': stats,
        'recordings': recordings
    }

if __name__ == "__main__":
    # 导入demo数据（只10条）
    dataset_id = import_demo_data(DATA_DIR, max_files=10)
    
    # 获取统计信息
    stats = get_demo_statistics()
    if stats:
        print("\n=== Demo Dataset Statistics ===")
        print(f"Overall distribution: {stats['overall']}")
        print(f"Total recordings: {len(stats['recordings'])}")
        print("\nRecordings:")
        for i, rec in enumerate(stats['recordings']):
            label = "ABNORMAL" if rec['is_abnormal'] else "NORMAL"
            print(f"  {i+1}. ID: {rec['id']}, Subject: {rec['subject_id']}, File: {rec['filename']}")
            print(f"      Duration: {rec['duration']:.1f}s, Channels: {rec['channels']}, Sample Rate: {rec['sampling_rate']:.0f}Hz, Label: {label}")
        
        # 统计标签分布
        abnormal_count = sum(1 for r in stats['recordings'] if r['is_abnormal'])
        normal_count = len(stats['recordings']) - abnormal_count
        print(f"\nLabel distribution:")
        print(f"  Normal: {normal_count} ({normal_count/len(stats['recordings'])*100:.1f}%)")
        print(f"  Abnormal: {abnormal_count} ({abnormal_count/len(stats['recordings'])*100:.1f}%)")
