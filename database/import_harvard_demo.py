import os
import sqlite3
import mne
import pandas as pd
from logging_config import logger
import json
import numpy as np

# Base directory for metadata (original location)
# BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "harvard_EEG"))
METADATA_BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "harvard_EEG"))
DB_PATH = os.path.join(os.path.dirname(__file__), "eeg2go.db")

BIDS_DIR = "/rds/general/user/zj724/ephemeral"
# BIDS_DIR = os.path.join(BASE_DIR, "bids")
METADATA_DIR = os.path.join(METADATA_BASE_DIR, "HEEDB_Metadata")
PATIENT_CSV = os.path.join(METADATA_DIR, "HEEDB_patients.csv")

MAX_MEMORY_GB = 1  # Set the memory usage limit for a single recording file (GB)
mne.utils.set_log_level('WARNING')

def make_subject_id(hospital_id, bdsp_id):
    return f"sub-{hospital_id}{bdsp_id}"

def detect_events(raw):
    """
    检测EEG文件中的事件信息
    
    Parameters:
        raw : mne.io.Raw
            EEG数据对象
    
    Returns:
        tuple : (has_events, event_types, events_data)
            has_events: 是否有事件
            event_types: 事件类型列表的JSON字符串
            events_data: 详细事件数据列表，用于插入recording_events表
    """
    try:
        # 尝试多种方式检测事件
        events = None
        
        # 方法1: 自动检测
        try:
            events = mne.find_events(raw, verbose='ERROR')
        except:
            pass
        
        # 方法2: 如果有STIM通道，手动指定
        if events is None or len(events) == 0:
            stim_channels = [ch for ch in raw.ch_names if 'STI' in ch.upper() or 'TRIG' in ch.upper()]
            if stim_channels:
                try:
                    events = mne.find_events(raw, stim_channel=stim_channels[0], verbose='ERROR')
                    logger.info(f"Found events using stim channel: {stim_channels[0]}")
                except:
                    pass
        
        # 方法3: 检查所有通道是否有事件标记
        if events is None or len(events) == 0:
            # 检查是否有任何通道包含事件信息
            for ch_name in raw.ch_names:
                if any(keyword in ch_name.upper() for keyword in ['STIM', 'TRIG', 'EVENT', 'MARKER']):
                    try:
                        events = mne.find_events(raw, stim_channel=ch_name, verbose='ERROR')
                        if len(events) > 0:
                            logger.info(f"Found events using channel: {ch_name}")
                            break
                    except:
                        continue
        
        if events is not None and len(events) > 0:
            event_ids = np.unique(events[:, 2])
            event_types = event_ids.tolist()
            
            # 准备详细事件数据
            events_data = []
            for event in events:
                onset = event[0] / raw.info['sfreq']  # 转换为秒
                event_type = str(event[2])  # 事件ID作为类型
                events_data.append({
                    'event_type': event_type,
                    'onset': onset,
                    'duration': 0.0,  # 默认持续时间
                    'value': str(event[2])  # 事件值
                })
            
            logger.info(f"Found {len(events)} events with IDs: {event_types}")
            return True, json.dumps(event_types), events_data
        else:
            logger.info("No events found in recording")
            return False, None, []
            
    except Exception as e:
        logger.warning(f"Error detecting events: {e}")
        return False, None, []

def import_harvard_edf_for_hospital(conn, hospital_id, dataset_name, max_import_count=None):
    """
    Import EDF files for a specified hospital ID into a specified dataset
    Each subject will have only one recording imported (the longest one) to ensure experimental diversity
    """
    c = conn.cursor()
    
    if max_import_count is not None:
        logger.info(f"Importing EDF recordings for {hospital_id} to dataset '{dataset_name}'... (max: {max_import_count})")
    else:
        logger.info(f"Importing EDF recordings for {hospital_id} to dataset '{dataset_name}'... (no limit)")

    # Create or get the dataset
    c.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,))
    row = c.fetchone()
    if row is None:
        c.execute("INSERT INTO datasets (name, description, source_type, path) VALUES (?, ?, ?, ?)",
                  (dataset_name, f"Harvard EEG 1000 data - {hospital_id}", "edf", BIDS_DIR))
        dataset_id = c.lastrowid
    else:
        dataset_id = row[0]

    hospital_path = os.path.join(BIDS_DIR, hospital_id)
    if not os.path.exists(hospital_path):
        logger.warning(f"Hospital directory {hospital_path} does not exist, skipping.")
        return dataset_id

    imported_count = 0
    for subj_folder in os.listdir(hospital_path):
        if not subj_folder.startswith("sub-"):
            continue
        
        if max_import_count is not None and imported_count >= max_import_count:
            logger.info(f"Reached import limit of {max_import_count}, stopping import.")
            break
        
        subject_id = subj_folder
        subject_path = os.path.join(hospital_path, subj_folder)

        # Insert subject if not exists (use INSERT OR IGNORE to avoid constraint violations)
        c.execute("INSERT OR IGNORE INTO subjects (subject_id, dataset_id) VALUES (?, ?)", 
                 (subject_id, dataset_id))

        # 1. First check if there is a direct EEG folder
        direct_eeg_path = os.path.join(subject_path, "eeg")
        if os.path.exists(direct_eeg_path):
            eeg_paths = [direct_eeg_path]
        else:
            # 2. Otherwise, iterate through all session EEG folders
            eeg_paths = []
            for ses_folder in os.listdir(subject_path):
                if not ses_folder.startswith("ses-"):
                    continue
                ses_path = os.path.join(subject_path, ses_folder, "eeg")
                if os.path.exists(ses_path):
                    eeg_paths.append(ses_path)

        # 收集该subject的所有EDF文件信息
        subject_recordings = []
        
        for eeg_path in eeg_paths:
            for fname in os.listdir(eeg_path):
                if not fname.endswith("_eeg.edf"):
                    continue
                fpath = os.path.join(eeg_path, fname)
                
                # 读取BIDS JSON文件
                json_fname = fname.replace("_eeg.edf", "_eeg.json")
                json_fpath = os.path.join(eeg_path, json_fname)
                json_info = {}
                if os.path.exists(json_fpath):
                    with open(json_fpath, "r", encoding="utf-8") as f:
                        json_info = json.load(f)
                
                # 读取EDF获取基本信息
                try:
                    raw = mne.io.read_raw_edf(fpath, preload=False, verbose='ERROR')
                    sfreq = raw.info['sfreq']
                    channels = len(raw.info['ch_names'])
                    duration = raw.n_times / sfreq
                    
                    # 收集recording信息
                    subject_recordings.append({
                        'fname': fname,
                        'fpath': fpath,
                        'eeg_path': eeg_path,
                        'json_info': json_info,
                        'sfreq': sfreq,
                        'channels': channels,
                        'duration': duration,
                        'raw': raw
                    })
                    
                except Exception as e:
                    logger.error(f"SKIPPING (cannot read): {fname}: {e}")
                    continue
        
        # 为每个subject选择一条recording（优先选择最长的）
        if subject_recordings:
            # 按duration排序，选择最长的recording
            subject_recordings.sort(key=lambda x: x['duration'], reverse=True)
            selected_recording = subject_recordings[0]
            
            if max_import_count is not None and imported_count >= max_import_count:
                logger.info(f"Reached import limit of {max_import_count}, stopping import.")
                break
            
            if max_import_count is not None:
                logger.info(f"Processing {subject_id} - selected recording: {selected_recording['fname']} (duration: {selected_recording['duration']:.1f}s) ... (imported: {imported_count}/{max_import_count})")
            else:
                logger.info(f"Processing {subject_id} - selected recording: {selected_recording['fname']} (duration: {selected_recording['duration']:.1f}s) ... (imported: {imported_count})")
            
            # 2. 提取字段（有则用，无则用默认）
            original_reference   = json_info.get("EEGReference", "n/a")
            recording_type       = json_info.get("RecordingType", "n/a")
            eeg_ground           = json_info.get("EEGGround", "n/a")
            placement_scheme     = json_info.get("EEGPlacementScheme", "n/a")
            manufacturer         = json_info.get("Manufacturer", "n/a")
            powerline_frequency  = json_info.get("PowerLineFrequency", "n/a")
            software_filters     = json_info.get("SoftwareFilters", "n/a")

            # 3. 检测事件信息
            has_events, event_types, events_data = detect_events(raw)
                
            # 4. 插入recordings表
            c.execute("""INSERT INTO recordings
                (dataset_id, subject_id, filename, path, duration, channels, sampling_rate,
                 original_reference, recording_type, eeg_ground, placement_scheme, manufacturer, 
                 powerline_frequency, software_filters, has_events, event_types)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (dataset_id, subject_id, selected_recording['fname'], selected_recording['fpath'], 
                 selected_recording['duration'], selected_recording['channels'], selected_recording['sfreq'],
                 original_reference, recording_type, eeg_ground, placement_scheme, manufacturer, 
                 powerline_frequency, software_filters, has_events, event_types))
                
            recording_id = c.lastrowid
            
            # 5. 如果有事件，插入recording_events表
            if has_events and events_data:
                for event in events_data:
                    c.execute("""INSERT INTO recording_events
                        (recording_id, event_type, onset, duration, value)
                        VALUES (?, ?, ?, ?, ?)""",
                        (recording_id, event['event_type'], event['onset'], 
                         event['duration'], event['value']))
                
                logger.info(f"Inserted {len(events_data)} events for recording {recording_id}")
            
            imported_count += 1
            if max_import_count is not None:
                logger.info(f"Imported: {selected_recording['fname']} ({imported_count}/{max_import_count})")
            else:
                logger.info(f"Imported: {selected_recording['fname']} (total: {imported_count})")

    conn.commit()
    if max_import_count is not None:
        logger.info(f"EDF import complete for {hospital_id}: {imported_count} recordings imported (limit: {max_import_count}).")
    else:
        logger.info(f"EDF import complete for {hospital_id}: {imported_count} recordings imported (no limit).")
    return dataset_id

def import_recording_metadata_for_hospital(conn, hospital_id):
    """
    Import EEG report metadata for a specified hospital ID
    """
    logger.info(f"Importing EEG report findings metadata for {hospital_id}...")
    c = conn.cursor()

    all_meta = []
    for fname in os.listdir(METADATA_DIR):
        if fname.endswith("_EEG_reports_findings.csv") or fname.endswith("_EEG__reports_findings.csv"):
            file_hospital_id = fname.split("_")[0]
            if file_hospital_id != hospital_id:
                continue
            df = pd.read_csv(os.path.join(METADATA_DIR, fname), dtype=str, low_memory=False)
            df["hospital_id"] = file_hospital_id
            all_meta.append(df)

    if not all_meta:
        logger.warning(f"No metadata CSV files found for {hospital_id}.")
        return

    meta_df = pd.concat(all_meta, ignore_index=True)

    # Get the dataset ID corresponding to the hospital
    dataset_name = f"Harvard_{hospital_id}_1000"
    c.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,))
    row = c.fetchone()
    if not row:
        logger.warning(f"Dataset '{dataset_name}' not found, skipping metadata import.")
        return
    
    dataset_id = row[0]

    # Get all recordings for the dataset
    rec_df = pd.read_sql_query("SELECT id, path, subject_id FROM recordings WHERE dataset_id = ?", 
                               conn, params=(dataset_id,))
    rec_df["path"] = rec_df["path"].astype(str)

    inserted = 0
    for _, row in meta_df.iterrows():
        subj_code = make_subject_id(row["hospital_id"], row["BDSPPatientID"])

        matches = rec_df[(rec_df["subject_id"] == subj_code)]

        if matches.empty:
            continue

        rec = matches.iloc[0]
        rec_id = rec["id"]

        c.execute("""INSERT OR REPLACE INTO recording_metadata (
            recording_id, subject_id, age_days, sex, start_time, end_time,
            seizure, spindles, status, normal, abnormal
        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""", (
            rec_id, subj_code,
            row.get("AgeInDaysAtVisit"),
            row.get("SexDSC"),
            row.get("StartTime(EEG)"),
            row.get("EndTime(EEG)"),
            row.get("seizure", 0),
            row.get("spindles", 0),
            row.get("status", 0),
            row.get("normal", 0),
            row.get("abnormal", 0)
        ))
        inserted += 1

    conn.commit()
    logger.info(f"Metadata import complete for {hospital_id}: {inserted} entries.")

def import_patient_metadata_for_hospital(conn, hospital_id):
    """
    Import patient-level metadata for a specified hospital ID
    """
    logger.info(f"Importing patient-level metadata for {hospital_id}...")
    c = conn.cursor()

    # Get the dataset ID corresponding to the hospital
    dataset_name = f"Harvard_{hospital_id}_1000"
    c.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,))
    row = c.fetchone()
    if not row:
        logger.warning(f"Dataset '{dataset_name}' not found, skipping patient metadata import.")
        return
    
    dataset_id = row[0]

    # Get all subjects for the dataset
    subj_set = set(pd.read_sql_query("SELECT subject_id FROM subjects WHERE dataset_id = ?", 
                                    conn, params=(dataset_id,))["subject_id"])

    df = pd.read_csv(PATIENT_CSV, dtype=str)
    inserted = 0

    for _, row in df.iterrows():
        # Only process records for the specified hospital
        if row["SiteID"] != hospital_id:
            continue
            
        subj_code = make_subject_id(row["SiteID"], row["BDSPPatientID"])
        if subj_code not in subj_set:
            continue

        c.execute("""UPDATE subjects SET
            sex = ?, age = ?, race = ?, ethnicity = ?,
            visit_count = ?, icd10_count = ?, medication_count = ?
            WHERE subject_id = ? AND dataset_id = ?""", (
            row.get("Sex"),
            float(row.get("AgeAtVisitAvg")) if row.get("AgeAtVisitAvg") else None,
            row.get("RaceAndEthnicity"),
            row.get("RaceAndEthnicityDSC"),
            int(float(row.get("VisitCount"))) if row.get("VisitCount") else None,
            int(row.get("ICD10Count")) if row.get("ICD10Count") else None,
            int(row.get("MedicationCount")) if row.get("MedicationCount") else None,
            subj_code, dataset_id
        ))
        inserted += 1

    conn.commit()
    logger.info(f"Patient metadata updated for {hospital_id}: {inserted} subjects.")

def main():
    """
    Main function: Import the S0001 and I0003 datasets separately
    """
    conn = sqlite3.connect(DB_PATH)
    
    # Set import limit (set to None for unlimited import, or a number like 2000 for limited import)
    IMPORT_LIMIT = 1000  # Change this to None for unlimited import, or a number for limited import
    
    # Import the S0001 dataset
    logger.info("============================================================")
    logger.info("Importing Harvard_S0001_1000 dataset")
    logger.info("============================================================")
    import_harvard_edf_for_hospital(conn, "S0001", "Harvard_S0001_1000", max_import_count=IMPORT_LIMIT)
    import_patient_metadata_for_hospital(conn, "S0001")
    
    # Import the I0003 dataset
    # logger.info("============================================================")
    # logger.info("Importing Harvard_I0003_1000 dataset")
    # logger.info("============================================================")
    # import_harvard_edf_for_hospital(conn, "I0003", "Harvard_I0003_1000", max_import_count=IMPORT_LIMIT)
    # # import_recording_metadata_for_hospital(conn, "I0003")
    # import_patient_metadata_for_hospital(conn, "I0003")
    
    conn.close()
    logger.info("All datasets imported successfully!")

if __name__ == "__main__":
    main()
