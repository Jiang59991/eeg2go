import os
import sqlite3
import mne
import pandas as pd
from logging_config import logger

BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "harvard_EEG"))
DB_PATH = os.path.join(os.path.dirname(__file__), "eeg2go.db")

BIDS_DIR = os.path.join(BASE_DIR, "bids")
METADATA_DIR = os.path.join(BASE_DIR, "HEEDB_Metadata")
PATIENT_CSV = os.path.join(METADATA_DIR, "HEEDB_patients.csv")

MAX_MEMORY_GB = 8  # Set the memory usage limit for a single recording file (GB)
mne.set_log_level('WARNING')

def make_subject_id(hospital_id, bdsp_id):
    return f"sub-{hospital_id}{bdsp_id}"

def import_harvard_edf_for_hospital(conn, hospital_id, dataset_name):
    """
    Import EDF files for a specified hospital ID into a specified dataset
    """
    c = conn.cursor()
    logger.info(f"Importing EDF recordings for {hospital_id} to dataset '{dataset_name}'...")

    # Create or get the dataset
    c.execute("SELECT id FROM datasets WHERE name = ?", (dataset_name,))
    row = c.fetchone()
    if row is None:
        c.execute("INSERT INTO datasets (name, description, source_type, path) VALUES (?, ?, ?, ?)",
                  (dataset_name, f"Harvard EEG demo data - {hospital_id}", "edf", BASE_DIR))
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
        
        subject_id = subj_folder
        subject_path = os.path.join(hospital_path, subj_folder)

        # Check if the subject already exists (in the same dataset)
        c.execute("SELECT subject_id FROM subjects WHERE subject_id = ? AND dataset_id = ?", 
                 (subject_id, dataset_id))
        if not c.fetchone():
            c.execute("INSERT INTO subjects (subject_id, dataset_id) VALUES (?, ?)", 
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

        for eeg_path in eeg_paths:
            logger.info(f"Processing {subject_id} - {eeg_path} ...")
            for fname in os.listdir(eeg_path):
                if not fname.endswith("_eeg.edf"):
                    continue
                fpath = os.path.join(eeg_path, fname)
                
                c.execute("SELECT id FROM recordings WHERE filename = ? AND path = ?", (fname, fpath))
                if c.fetchone():
                    logger.info(f"SKIPPING (already exists): {fname}")
                    continue

                try:
                    raw = mne.io.read_raw_edf(fpath, preload=False, verbose='ERROR')
                    sfreq = raw.info['sfreq']
                    channels = len(raw.info['ch_names'])
                    duration = raw.n_times / sfreq
                    
                    # --- Memory usage pre-check ---
                    estimated_mb = (channels * raw.n_times * 8) / (1024**2)
                    if estimated_mb > (MAX_MEMORY_GB * 1024):
                        logger.warning(f"SKIPPING (too large): {fname} ({estimated_mb:.1f}MB > {MAX_MEMORY_GB}GB)")
                        continue # If the file is too large, skip this file

                except Exception as e:
                    logger.error(f"SKIPPING (cannot read): {fname}: {e}")
                    continue

                c.execute("""INSERT INTO recordings
                    (dataset_id, subject_id, filename, path, duration, channels, sampling_rate)
                    VALUES (?, ?, ?, ?, ?, ?, ?)""",
                    (dataset_id, subject_id, fname, fpath, duration, channels, sfreq))
                
                imported_count += 1
                logger.info(f"Imported: {fname}")

    conn.commit()
    logger.info(f"EDF import complete for {hospital_id}: {imported_count} recordings imported.")
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
    dataset_name = f"Harvard_{hospital_id}_demo"
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
    dataset_name = f"Harvard_{hospital_id}_demo"
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
    
    # Import the S0001 dataset
    logger.info("============================================================")
    logger.info("Importing Harvard_S0001_demo dataset")
    logger.info("============================================================")
    import_harvard_edf_for_hospital(conn, "S0001", "Harvard_S0001_demo")
    import_patient_metadata_for_hospital(conn, "S0001")
    
    # # Import the I0003 dataset
    # print("=" * 60)
    # print("Importing Harvard_I0003_demo dataset")
    # print("=" * 60)
    # import_harvard_edf_for_hospital(conn, "I0003", "Harvard_I0003_demo")
    # # import_recording_metadata_for_hospital(conn, "I0003")
    # import_patient_metadata_for_hospital(conn, "I0003")
    
    conn.close()
    logger.info("All datasets imported successfully!")

if __name__ == "__main__":
    main()
