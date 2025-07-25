import mne
mne.set_log_level('WARNING')
import numpy as np
import sqlite3
import os
import gc
from eeg2fx.feature.common import auto_gc
from logging_config import logger
import csv

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))
MAX_MEMORY_GB = 1  # 统一内存限制常量

class RecordingTooLargeError(Exception):
    """当录音文件过大时抛出的异常"""
    pass

def build_channel_rename_dict(lookup_csv_path):
    rename_dict = {}
    with open(lookup_csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # 跳过表头
        for row in reader:
            dest = row[0].strip()
            for src in row[1:]:
                src = src.strip()
                if src:
                    rename_dict[src] = dest
    return rename_dict

# ====== 只在模块加载时读取一次csv，生成全局变量 ======
LOOKUP_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chanlookup.csv"))
RENAME_DICT = build_channel_rename_dict(LOOKUP_CSV_PATH)
# =====================================================

def load_recording(recording_id):
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT path, duration, channels, sampling_rate FROM recordings WHERE id = ?", (recording_id,))
    row = c.fetchone()
    conn.close()

    if not row:
        raise ValueError(f"Recording id {recording_id} not found in recordings table.")

    filepath, duration, channels, sampling_rate = row
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"EEG file not found at path: {filepath}")

    # 计算预估内存使用量
    if duration and channels and sampling_rate:
        estimated_memory_mb = (duration * sampling_rate * channels * 4) / (1024 * 1024)  # 4 bytes per float32
        logger.info(f"[load_recording] Recording {recording_id}: {duration:.1f}s, {channels} channels, {sampling_rate} Hz")
        logger.info(f"[load_recording] Estimated memory usage: {estimated_memory_mb:.1f} MB")
        
        # 检查是否超过内存限制
        if estimated_memory_mb > MAX_MEMORY_GB * 1024:
            logger.warning(f"[load_recording] Recording {recording_id} is too large ({estimated_memory_mb:.2f} MB), exceeding limit of {MAX_MEMORY_GB*1024:.2f} MB. Skipping.")
            raise RecordingTooLargeError(f"Recording too large to process ({estimated_memory_mb:.2f} MB)")
        

        # raw = mne.io.read_raw_edf(filepath, preload=True, verbose='ERROR')


        # 根据文件大小决定是否使用内存映射
        if estimated_memory_mb > 900:
            logger.info(f"[load_recording] Large file detected, using memory mapping")
            raw = mne.io.read_raw_edf(filepath, preload='auto', verbose='ERROR')
        else:
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose='ERROR')
    else:
        logger.info(f"[load_recording] Missing metadata, using memory mapping")
        raw = mne.io.read_raw_edf(filepath, preload='auto', verbose='ERROR')

    # 1. 只重命名raw中存在的通道
    channel_rename_map = {}
    for ch in raw.ch_names:
        if ch in RENAME_DICT:
            channel_rename_map[ch] = RENAME_DICT[ch]
    if channel_rename_map:
        logger.info(f"[load_recording] Renaming channels: {channel_rename_map}")
        raw.rename_channels(channel_rename_map)
    else:
        logger.info(f"[load_recording] No channels to rename according to lookup table.")

    return raw

@auto_gc
def filter(raw, hp=1.0, lp=40.0):
    raw_copy = raw.copy()
    raw_copy.filter(hp, lp, fir_design='firwin', verbose='ERROR')
    return raw_copy

@auto_gc
def notch_filter(raw, freq=50.0, sfreq=None):
    return raw.copy().notch_filter(freqs=[freq])

@auto_gc
def reref(raw, sfreq=None):
    raw_ref = raw.copy().set_eeg_reference('average')
    return raw_ref

@auto_gc
def resample(raw, sfreq=128.0):
    return raw.copy().resample(sfreq=sfreq)

@auto_gc
def ica(raw, n_components=20, sfreq=None):
    ica_inst = mne.preprocessing.ICA(n_components=n_components, random_state=97, max_iter='auto')
    ica_inst.fit(raw)
    return ica_inst.apply(raw.copy())

@auto_gc
def epoch(raw, duration=5.0):
    """
    Segment raw data into fixed-length epochs.
    """
    events = mne.make_fixed_length_events(raw, duration=duration)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=duration, baseline=None, preload=True, verbose='ERROR')
    return epochs

@auto_gc
def reject_high_amplitude(epochs, threshold=150e-6, sfreq=None):
    return epochs.copy().drop_bad(reject=dict(eeg=threshold))

@auto_gc
def zscore(epochs):
    """
    Standardize epochs across time for each channel.
    """
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    mean = np.mean(data, axis=2, keepdims=True)
    std = np.std(data, axis=2, keepdims=True)
    std[std == 0] = 1e-6  
    data_z = (data - mean) / std

    zscored = epochs.copy()
    zscored._data = data_z
    return zscored
