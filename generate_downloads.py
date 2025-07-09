import csv
import os
import subprocess

CSV_PATIENTS = "data/harvard_EEG/HEEDB_Metadata/HEEDB_patients.csv"
CSV_ICD10 = "data/harvard_EEG/HEEDB_Metadata/HEEDB_ICD10_for_Neurology.csv"
CSV_MEDS = "data/harvard_EEG/HEEDB_Metadata/HEEDB_Medication_ATC.csv"
SITE_ID = "S0001"

s3_prefix = "s3://arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-eeg-access-point/EEG/bids/S0001/"
# local_prefix = "data/harvard_EEG/bids/S0001/"
EPHEMERAL_PATH = "/rds/general/user/zj724/ephemeral/S0001/"

# Ensure base local folder exists
os.makedirs(EPHEMERAL_PATH, exist_ok=True)
# os.makedirs(local_prefix, exist_ok=True)

def load_table(path):
    with open(path, newline='') as f:
        rdr = csv.DictReader(f)
        return {row["BDSPPatientID"]: row for row in rdr}

pts  = load_table(CSV_PATIENTS)
dx   = load_table(CSV_ICD10)
meds = load_table(CSV_MEDS)

FLAG = 0

for pid, row_pt in pts.items():

    if row_pt["SiteID"] != SITE_ID:
        continue

    row_dx   = dx.get(pid, {})
    row_meds = meds.get(pid, {})

    # ---------- customise your filter here ----------
    if not (
        row_pt["Sex"] == "M" and
        row_dx.get("Seizure Disorders") and
        row_meds.get("Nervous System Drugs")
    ):
        continue
    # -----------------------------------------------

    FLAG += 1
    s3_path     = f"{s3_prefix}sub-{SITE_ID}{pid}/"
    local_path  = os.path.join(EPHEMERAL_PATH, f"sub-{SITE_ID}{pid}/")

    print(f"Downloading {s3_path} → {local_path}")
    try:
        subprocess.run(["aws","s3","cp",s3_path,local_path,"--recursive"], check=True)
    except subprocess.CalledProcessError as e:
        print(f"Failed on {pid}: {e}")

    if FLAG == 100:          # stop after 100 subjects
        break
