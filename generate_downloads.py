import csv
import os
import subprocess
import json

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

def get_s3_folder_size(s3_path):
    """获取S3文件夹的总大小（字节）"""
    try:
        # 使用aws s3 ls命令获取文件夹内容
        result = subprocess.run(
            ["aws", "s3", "ls", s3_path, "--recursive", "--summarize"],
            capture_output=True, text=True, check=True
        )
        
        # 解析输出找到总大小
        for line in result.stdout.split('\n'):
            if line.strip().startswith('Total Size:'):
                size_str = line.split(':')[1].strip()
                # 移除 "bytes" 并转换为整数
                size_bytes = int(size_str.replace(' bytes', ''))
                return size_bytes
        return 0
    except subprocess.CalledProcessError:
        print(f"Failed to get size for {s3_path}")
        return float('inf')  # 返回无穷大，表示无法获取大小

pts  = load_table(CSV_PATIENTS)
dx   = load_table(CSV_ICD10)
meds = load_table(CSV_MEDS)

FLAG = 0
MAX_SIZE_GB = 1
MAX_SIZE_BYTES = MAX_SIZE_GB * 1024 * 1024 * 1024  # 1GB in bytes

print(f"只下载小于 {MAX_SIZE_GB}GB 的数据...")

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

    s3_path = f"{s3_prefix}sub-{SITE_ID}{pid}/"
    local_path = os.path.join(EPHEMERAL_PATH, f"sub-{SITE_ID}{pid}/")

    # 检查文件夹大小
    print(f"检查 {s3_path} 的大小...")
    folder_size = get_s3_folder_size(s3_path)
    
    if folder_size > MAX_SIZE_BYTES:
        print(f"跳过 {pid}: 文件夹大小 {folder_size / (1024**3):.2f}GB > {MAX_SIZE_GB}GB")
        continue
    
    if folder_size == 0:
        print(f"跳过 {pid}: 文件夹为空或不存在")
        continue

    FLAG += 1
    print(f"下载 {s3_path} → {local_path} (大小: {folder_size / (1024**3):.2f}GB)")
    
    try:
        subprocess.run(["aws","s3","cp",s3_path,local_path,"--recursive"], check=True)
        print(f"成功下载 {pid}")
    except subprocess.CalledProcessError as e:
        print(f"下载失败 {pid}: {e}")

    if FLAG == 1000:          # stop after 50 subjects
        print(f"已达到50个受试者限制")
        break

print(f"总共下载了 {FLAG} 个受试者的数据")
