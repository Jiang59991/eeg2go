import os
import sqlite3
import mne
from logging_config import logger

DATA_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data"))
DB_PATH = os.path.join(os.path.dirname(__file__), "eeg2go.db")
mne.set_log_level('WARNING')

def update_subject_info_from_txt(dataset_id, tuh_demo_dir, conn):
    c = conn.cursor()

    def parse_subject_file(file_path, epilepsy_label):
        with open(file_path, "r") as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue
                subj_id = parts[0].lower()
                age = int(parts[1].strip('[]').replace("Age:", ""))
                sex = parts[2].strip('[]')
                c.execute("""
                    UPDATE subjects
                    SET age = ?, sex = ?, epilepsy = ?
                    WHERE subject_id = ? AND dataset_id = ?
                """, (age, sex, epilepsy_label, subj_id, dataset_id))

    logger.info("Updating subject info from TXT...")
    parse_subject_file(os.path.join(tuh_demo_dir, "00_subject_ids_epilepsy.list.txt"), epilepsy_label=1)
    parse_subject_file(os.path.join(tuh_demo_dir, "01_subject_ids_no_epilepsy.list.txt"), epilepsy_label=0)
    conn.commit()
    logger.info("Subject info updated from text files.")

def import_tuh_demo_data(data_dir):
    logger.info(f"Importing TUH demo data from {data_dir}")
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()

    # 1. 确保 TUH_demo 数据集存在
    c.execute("SELECT id FROM datasets WHERE name = ?", ("TUH_demo",))
    row = c.fetchone()
    if row is None:
        c.execute("INSERT INTO datasets (name, description, source_type, path) VALUES (?, ?, ?, ?)",
                  ("TUH_demo", "Temple University Hospital demo EEG data", "edf", data_dir))
        dataset_id = c.lastrowid
    else:
        dataset_id = row[0]

    # 2. 遍历所有 .edf 文件
    for root, dirs, files in os.walk(data_dir):
        for fname in files:
            if fname.endswith(".edf"):
                fpath = os.path.join(root, fname)
                subject_code = fname.split("_")[0].lower()
                if not subject_code.isalpha() or len(subject_code) < 5:
                    logger.warning(f"Skipped {fname} (invalid subject ID: {subject_code})")
                    continue

                # 插入或查询 subject
                c.execute("SELECT id FROM subjects WHERE subject_id = ? AND dataset_id = ?", (subject_code, dataset_id))
                row = c.fetchone()
                if row is None:
                    c.execute("INSERT INTO subjects (subject_id, dataset_id) VALUES (?, ?)", (subject_code, dataset_id))
                    subject_id = c.lastrowid
                else:
                    subject_id = row[0]

                c.execute("SELECT id FROM recordings WHERE filename = ? AND path = ?", (fname, fpath))
                if c.fetchone():
                    continue

                try:
                    raw = mne.io.read_raw_edf(fpath, preload=False, verbose='ERROR')
                    sfreq = raw.info['sfreq']
                    channels = len(raw.info['ch_names'])
                    duration = raw.n_times / sfreq
                except Exception as e:
                    logger.error(f"Failed to read {fname}: {e}")
                    sfreq = channels = duration = None

                c.execute("""
                    INSERT INTO recordings (dataset_id, subject_id, filename, path, duration, channels, sampling_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                """, (dataset_id, subject_id, fname, fpath, duration, channels, sfreq))

                logger.info(f"{fname} | Subject: {subject_code} | Duration: {duration:.1f}s" if sfreq else f"Imported (basic): {fname}")

    update_subject_info_from_txt(dataset_id, data_dir, conn)
    conn.close()
    logger.info("All data imported successfully.")

if __name__ == "__main__":
    import_tuh_demo_data(os.path.join(DATA_DIR, 'tuh_demo'))
