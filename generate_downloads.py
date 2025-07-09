import csv
import os
import subprocess
import json
import logging
from datetime import datetime

CSV_PATH = "data/harvard_EEG/HEEDB_Metadata/HEEDB_patients.csv"
SITE_ID = "S0001"
MAX_SIZE_GB = 5  # Maximum file size limit (GB)

s3_prefix = "s3://arn:aws:s3:us-east-1:184438910517:accesspoint/bdsp-eeg-access-point/EEG/bids/S0001/"
# local_prefix = "data/harvard_EEG/bids/S0001/"
EPHEMERAL_PATH = "/rds/general/user/zj724/ephemeral/S0001/"

# Setup logging
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)
log_filename = f"download_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
log_path = os.path.join(log_dir, log_filename)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(log_path),
        logging.StreamHandler()  # Also print to console
    ]
)
logger = logging.getLogger(__name__)

def get_s3_folder_size(s3_path):
    """Get total size of S3 folder (bytes)"""
    try:
        # Use aws s3 ls command to get folder size
        result = subprocess.run([
            "aws", "s3", "ls", s3_path, "--recursive", "--summarize"
        ], capture_output=True, text=True, check=True)
        
        # Parse output to find total size
        for line in result.stdout.split('\n'):
            if line.strip().startswith('Total Size:'):
                size_str = line.split(':')[1].strip()
                # Remove "bytes" and convert to integer
                size_bytes = int(size_str.replace(' bytes', ''))
                return size_bytes
        return 0
    except subprocess.CalledProcessError as e:
        logger.error(f"Failed to get folder size {s3_path}: {e}")
        return 0

# Ensure base local folder exists
os.makedirs(EPHEMERAL_PATH, exist_ok=True)
# os.makedirs(local_prefix, exist_ok=True)

logger.info(f"Starting download process. Log file: {log_path}")
logger.info(f"Max file size limit: {MAX_SIZE_GB} GB")

# FLAG = 0
# sub-I0003175086301

with open(CSV_PATH, newline='') as csvfile:
    reader = csv.DictReader(csvfile)
    for row in reader:
        if row["SiteID"] == SITE_ID:
            # FLAG = FLAG + 1
            subject_id = row["BDSPPatientID"]
            # if FLAG== False and subject_id == "175160615":
            #     print("HIT")
            #     FLAG = True
            
            # if FLAG == False:
            #     continue
            
            s3_path = f"{s3_prefix}sub-{SITE_ID}{subject_id}/"
            # local_path = f"{local_prefix}sub-{SITE_ID}{subject_id}/"
            local_path = os.path.join(EPHEMERAL_PATH, f"sub-{SITE_ID}{subject_id}/")
            # print(f"aws s3 cp {s3_path} {local_path} --recursive")

            # Check folder size
            folder_size_bytes = get_s3_folder_size(s3_path)
            folder_size_gb = folder_size_bytes / (1024**3)
            
            logger.info(f"Checking {s3_path} - Size: {folder_size_gb:.2f} GB")
            
            if folder_size_gb > MAX_SIZE_GB:
                logger.warning(f"Skipping {subject_id}: File size ({folder_size_gb:.2f} GB) exceeds limit ({MAX_SIZE_GB} GB)")
                continue

            logger.info(f"Downloading {s3_path} → {local_path}")
            try:
                subprocess.run([
                    "aws", "s3", "cp",
                    s3_path,
                    local_path,
                    "--recursive"
                ], check=True)
                logger.info(f"Successfully downloaded {subject_id}")
            except subprocess.CalledProcessError as e:
                logger.error(f"Failed to download {subject_id}: {e}")

            # if FLAG == 200:
            #     break

logger.info("Download process completed")
