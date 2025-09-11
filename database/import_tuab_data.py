#!/usr/bin/env python3
import os
import sqlite3
import mne
import re
from typing import Optional, Dict, Any, List
from logging_config import logger

DATA_DIR = "/rds/general/user/zj724/ephemeral/TUAB_v3_0_1_edf"
DB_PATH = os.path.join(os.path.dirname(__file__), "eeg2go.db")
mne.set_log_level('WARNING')

def extract_subject_info_from_path(file_path: str) -> Optional[Dict[str, Any]]:
    """
    Extract subject information from TUAB v3.0.1 file path.

    Args:
        file_path (str): Relative path of the .edf file, e.g. eval/abnormal/01_tcp_ar/aaaaabdo_s003_t000.edf

    Returns:
        Optional[Dict[str, Any]]: Dictionary with subject/session/token/recording_id/split_type/label/config/filename, or None if parsing fails.
    """
    path_parts = file_path.split('/')
    if len(path_parts) >= 4:
        split_type = path_parts[0]
        label = path_parts[1]
        config = path_parts[2]
        filename = path_parts[3]
        filename_parts = filename.replace('.edf', '').split('_')
        if len(filename_parts) >= 3:
            subject_id = filename_parts[0]
            session = filename_parts[1]
            token = filename_parts[2]
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

def import_tuab_data(data_dir: str) -> Optional[int]:
    """
    Import TUAB v3.0.1 data into the database.

    Args:
        data_dir (str): Directory containing TUAB EDF files.

    Returns:
        Optional[int]: Dataset ID of the imported TUAB dataset.
    """
    logger.info(f"Importing TUAB v3.0.1 data from {data_dir}")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # Ensure TUAB dataset exists
    c.execute("SELECT id FROM datasets WHERE name = ?", ("TUAB_v3.0.1",))
    row = c.fetchone()
    if row is None:
        c.execute("INSERT INTO datasets (name, description, source_type, path) VALUES (?, ?, ?, ?)",
                  ("TUAB_v3.0.1", "Temple University Hospital Abnormal EEG Corpus v3.0.1", "edf", data_dir))
        dataset_id = c.lastrowid
        logger.info(f"Created TUAB v3.0.1 dataset with ID: {dataset_id}")
    else:
        dataset_id = row[0]
        logger.info(f"Using existing TUAB v3.0.1 dataset with ID: {dataset_id}")

    edf_files = []
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith(".edf"):
                fpath = os.path.join(root, fname)
                rel_path = os.path.relpath(fpath, data_dir)
                edf_files.append((fpath, rel_path))
    
    logger.info(f"Found {len(edf_files)} EDF files")
    imported_count = 0
    skipped_count = 0
    
    for fpath, rel_path in edf_files:
        info = extract_subject_info_from_path(rel_path)
        if not info:
            logger.warning(f"Could not extract subject info from path: {rel_path}")
            skipped_count += 1
            continue

        c.execute("SELECT id FROM recordings WHERE filename = ? AND path = ?", (info['filename'], fpath))
        if c.fetchone():
            logger.debug(f"Already imported: {info['filename']}")
            skipped_count += 1
            continue

        c.execute("SELECT subject_id FROM subjects WHERE subject_id = ? AND dataset_id = ?", (info['subject_id'], dataset_id))
        if not c.fetchone():
            c.execute("INSERT INTO subjects (subject_id, dataset_id) VALUES (?, ?)", (info['subject_id'], dataset_id))
            logger.debug(f"Created new subject: {info['subject_id']}")

        try:
            raw = mne.io.read_raw_edf(fpath, preload=False, verbose='ERROR')
            sfreq = raw.info['sfreq']
            channels = len(raw.info['ch_names'])
            duration = raw.n_times / sfreq
            ch_names = raw.info['ch_names']
        except Exception as e:
            logger.error(f"Failed to read {fpath}: {e}")
            sfreq = channels = duration = None
            ch_names = []

        c.execute("""
            INSERT INTO recordings (
                dataset_id, subject_id, filename, path, duration, 
                channels, sampling_rate, recording_type, placement_scheme
            )
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (dataset_id, info['subject_id'], info['filename'], fpath, duration, 
              channels, sfreq, "continuous", "10-20"))

        # Insert recording_metadata with normal/abnormal label
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
        
        if imported_count % 100 == 0:
            logger.info(f"Imported {imported_count} files...")
            conn.commit()

        logger.debug(f"Imported: {info['filename']} | Subject: {info['subject_id']} | Label: {info['label']} | Duration: {duration:.1f}s" if sfreq else f"Imported (basic): {info['filename']}")

    conn.commit()
    conn.close()
    logger.info(f"Import completed. Imported: {imported_count}, Skipped: {skipped_count}")
    return dataset_id

def create_tuab_subset_for_experiment1() -> Optional[List[Dict[str, Any]]]:
    """
    Create TUAB subset for Experiment 1: select one recording per subject,
    preferring abnormal label, otherwise the longest recording.

    Returns:
        Optional[List[Dict[str, Any]]]: List of selected recordings with keys: recording_id, subject_id, filename, path, duration, is_abnormal.
    """
    logger.info("Creating TUAB subset for Experiment 1...")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM datasets WHERE name = ?", ("TUAB_v3.0.1",))
    row = c.fetchone()
    if not row:
        logger.error("TUAB v3.0.1 dataset not found. Please import TUAB data first.")
        return None
    dataset_id = row[0]
    c.execute("""
        SELECT DISTINCT r.subject_id 
        FROM recordings r
        WHERE r.dataset_id = ? 
        ORDER BY r.subject_id
    """, (dataset_id,))
    subjects = [row[0] for row in c.fetchall()]
    logger.info(f"Found {len(subjects)} subjects in TUAB v3.0.1 dataset")
    selected_recordings = []
    for subject_id in subjects:
        c.execute("""
            SELECT r.id, r.filename, r.path, r.duration, rm.abnormal
            FROM recordings r
            JOIN recording_metadata rm ON r.id = rm.recording_id
            WHERE r.dataset_id = ? AND r.subject_id = ?
            ORDER BY rm.abnormal DESC, r.duration DESC
            LIMIT 1
        """, (dataset_id, subject_id))
        row = c.fetchone()
        if row:
            selected_recordings.append({
                'recording_id': row[0],
                'subject_id': subject_id,
                'filename': row[1],
                'path': row[2],
                'duration': row[3],
                'is_abnormal': row[4] == '1' if isinstance(row[4], str) else bool(row[4])
            })
    conn.close()
    logger.info(f"Selected {len(selected_recordings)} recordings for Experiment 1")
    abnormal_count = sum(1 for r in selected_recordings if r['is_abnormal'])
    normal_count = len(selected_recordings) - abnormal_count
    logger.info(f"Label distribution: Normal={normal_count}, Abnormal={abnormal_count}")
    return selected_recordings

def get_label_statistics() -> Optional[Dict[str, Any]]:
    """
    Get label statistics for the TUAB dataset.

    Returns:
        Optional[Dict[str, Any]]: Dictionary with overall and by_split label statistics.
    """
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM datasets WHERE name = ?", ("TUAB_v3.0.1",))
    row = c.fetchone()
    if not row:
        logger.error("TUAB v3.0.1 dataset not found.")
        return None
    dataset_id = row[0]
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
    c.execute("""
        SELECT 
            CASE 
                WHEN r.path LIKE '%/eval/%' THEN 'eval'
                WHEN r.path LIKE '%/train/%' THEN 'train'
                ELSE 'unknown'
            END as split_type,
            rm.abnormal,
            COUNT(*) as count
        FROM recordings r
        JOIN recording_metadata rm ON r.id = rm.recording_id
        WHERE r.dataset_id = ?
        GROUP BY split_type, rm.abnormal
    """, (dataset_id,))
    split_stats = {}
    for row in c.fetchall():
        split_type = row[0]
        is_abnormal = bool(row[1])
        count = row[2]
        if split_type not in split_stats:
            split_stats[split_type] = {'normal': 0, 'abnormal': 0}
        split_stats[split_type]['abnormal' if is_abnormal else 'normal'] = count
    conn.close()
    return {
        'overall': stats,
        'by_split': split_stats
    }

if __name__ == "__main__":
    dataset_id = import_tuab_data(DATA_DIR)
    stats = get_label_statistics()
    if stats:
        print("\n=== TUAB v3.0.1 Dataset Statistics ===")
        print(f"Overall distribution: {stats['overall']}")
        print("By split:")
        for split_type, counts in stats['by_split'].items():
            print(f"  {split_type}: {counts}")
    subset = create_tuab_subset_for_experiment1()
    if subset:
        print(f"\nExperiment 1 subset created with {len(subset)} recordings")
        print("Sample recordings:")
        for i, rec in enumerate(subset[:5]):
            label = "ABNORMAL" if rec['is_abnormal'] else "NORMAL"
            print(f"  {i+1}. {rec['subject_id']}: {rec['filename']} ({rec['duration']:.1f}s, {label})")
        if len(subset) > 5:
            print(f"  ... and {len(subset)-5} more")
        abnormal_count = sum(1 for r in subset if r['is_abnormal'])
        normal_count = len(subset) - abnormal_count
        print(f"\nExperiment 1 subset label distribution:")
        print(f"  Normal: {normal_count} ({normal_count/len(subset)*100:.1f}%)")
        print(f"  Abnormal: {abnormal_count} ({abnormal_count/len(subset)*100:.1f}%)")
