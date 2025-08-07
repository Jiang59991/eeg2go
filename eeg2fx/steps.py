import mne
mne.set_log_level('WARNING')
import numpy as np
import sqlite3
import os
import gc
from .feature.common import auto_gc
from logging_config import logger
import csv

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))
MAX_MEMORY_GB = 1  # 统一内存限制常量 (1GB)

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
def filter(raw, hp, lp):
    if hp is None or lp is None:
        raise ValueError("High-pass and low-pass cutoff frequencies must be explicitly provided.")
    if lp <= hp:
        raise ValueError(f"Low-pass frequency ({lp}) must be greater than high-pass ({hp}).")
    return raw.copy().filter(l_freq=hp, h_freq=lp, fir_design='firwin', verbose='ERROR')

@auto_gc
def notch_filter(raw, freq):
    if freq is None:
        raise ValueError("Notch filter frequency must be specified.")
    sfreq = raw.info.get('sfreq')
    if freq >= sfreq / 2:
        raise ValueError(f"Notch frequency {freq} is above Nyquist frequency.")
    return raw.copy().notch_filter(freqs=[freq])

@auto_gc
def reref(raw, method, original_reference=None):
    """
    Apply EEG re-referencing strategy.

    Parameters:
        raw: mne.io.Raw
            The raw EEG recording.
        method: str
            The re-reference strategy. Options: 'average', 'linked_mastoid', 'none'.
        original_reference: str or None
            The original reference used at recording time. Can be 'Cz', 'average', 'CMS/DRL', etc.

    Returns:
        raw_ref: mne.io.Raw
            The re-referenced raw object.
    """

    original_reference = (original_reference or "").lower()
    method = method.lower()

    # Catch dangerous or nonsensical combinations
    if method == "average" and original_reference == "average":
        raise ValueError("Recording is already average referenced; do not apply average again.")

    raw_copy = raw.copy()

    if method == "average":
        raw_copy.set_eeg_reference('average')
    elif method == "linked_mastoid":
        try:
            raw_copy.set_eeg_reference(['M1', 'M2'])
        except ValueError:
            raise ValueError("M1/M2 not found in channel list; cannot apply linked mastoid reference.")
    else:
        raise ValueError(f"Unknown re-reference method: {method}")

    return raw_copy


@auto_gc
def resample(raw, sfreq):
    if sfreq is None:
        raise ValueError("Target sampling rate must be explicitly specified.")
    original_sfreq = raw.info.get('sfreq')
    if np.isclose(sfreq, original_sfreq):
        return raw.copy()  # Skip if already at target sfreq
    return raw.copy().resample(sfreq=sfreq)

@auto_gc
def ica(raw, n_components, detect_artifacts):
    """
    Fully automated ICA step for EEG pipelines. No manual interaction allowed.

    Parameters:
        raw : mne.io.Raw
            Continuous EEG recording.
        n_components : int or float
            Number of components to retain (int or PCA variance ratio).
        detect_artifacts : str
            Artifact detection strategy: 'eog', 'ecg', or 'none'.
        random_state : int
            For reproducibility of ICA.

    Returns:
        raw_clean : mne.io.Raw
            EEG data after ICA cleaning (if any components excluded).
    """

    if isinstance(n_components, int) and n_components > raw.info['nchan']:
        raise ValueError(f"n_components={n_components} exceeds number of channels.")
    
    ica_inst = mne.preprocessing.ICA(n_components=n_components, random_state=97, max_iter='auto')
    ica_inst.fit(raw)

    exclude = []
    if detect_artifacts == "eog":
        exclude, _ = ica_inst.find_bads_eog(raw)
    elif detect_artifacts == "ecg":
        exclude, _ = ica_inst.find_bads_ecg(raw)

    raw_clean = ica_inst.apply(raw.copy(), exclude=exclude)

    return raw_clean

@auto_gc
def epoch(raw, duration=5.0):
    """
    Segment raw data into fixed-length epochs.
    """
    events = mne.make_fixed_length_events(raw, duration=duration)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=duration, baseline=None, preload=True, verbose='ERROR')
    return epochs

@auto_gc
def epoch_by_event(raw, event_type, tmin, tmax, recording_id):
    if recording_id is None:
        raise ValueError("recording_id must be provided")

    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT DISTINCT value FROM recording_events WHERE recording_id = ? AND event_type = ?",
        (recording_id, event_type)
    )
    rows = c.fetchall()
    conn.close()

    if not rows:
        raise ValueError(f"No events of type '{event_type}' found for recording {recording_id}")

    event_values = [row[0] for row in rows]
    try:
        event_ids = [int(v) for v in event_values]
    except Exception:
        raise ValueError(f"Event values must be convertible to int: {event_values}")

    event_id_map = {f"{event_type}_{v}": v for v in event_ids}

    events = mne.find_events(raw, verbose='ERROR')
    if len(events) == 0:
        raise ValueError("No events found in the recording.")

    # 检查 event_id 是否存在
    available_event_ids = np.unique(events[:, 2])
    for name, val in event_id_map.items():
        if val not in available_event_ids:
            logger.warning(f"Event ID {val} ({name}) not found in recording. Available: {available_event_ids}")

    # 创建 epochs（不进行 baseline 或 reject）
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id_map,
        tmin=tmin,
        tmax=tmax,
        baseline=None,
        reject=None,
        flat=None,
        preload=True,
        verbose='ERROR'
    )

    logger.info(f"Created {len(epochs)} epochs from {len(events)} events using event_ids={event_id_map}")

    return epochs

@auto_gc
def reject_high_amplitude(epochs, threshold_uv):
    """
    Reject epochs with any EEG channel exceeding the given amplitude.

    Parameters:
        epochs : mne.Epochs
            Input EEG epochs.
        threshold_uv : float
            Absolute amplitude threshold (in microvolts). Default is 150 µV.

    Returns:
        clean_epochs : mne.Epochs
            Epochs after rejecting high-amplitude artifacts.
    """
    threshold_v = threshold_uv * 1e-6  # convert µV to V
    clean_epochs = epochs.copy().drop_bad(reject={"eeg": threshold_v})
    return clean_epochs

@auto_gc
def zscore(epochs, mode):
    """
    Standardize epochs across time for each channel.

    Parameters:
        epochs : mne.Epochs
            Input EEG epochs.
        mode : str
            Normalization strategy. 
            'per_epoch': normalize each epoch × channel individually (default).
            'global': normalize across all epochs per channel.
    """
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    
    if mode == "per_epoch":
        mean = np.mean(data, axis=2, keepdims=True)
        std = np.std(data, axis=2, keepdims=True)
    elif mode == "global":
        mean = np.mean(data, axis=(0, 2), keepdims=True)
        std = np.std(data, axis=(0, 2), keepdims=True)
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Choose from 'per_epoch' or 'global'.")

    std[std == 0] = 1e-6  # floor to prevent divide-by-zero
    data_z = (data - mean) / std

    zscored = epochs.copy()
    zscored._data = data_z
    return zscored

@auto_gc
def detect_bad_channels(raw, flat_thresh=1e-7, noisy_thresh=1e-4, correlation_thresh=0.4):
    data, _ = raw.get_data(picks="eeg", return_times=False)
    ch_names = raw.ch_names
    n_channels = data.shape[0]

    bads = set()
    stds = np.std(data, axis=1)

    for i, std in enumerate(stds):
        if std < flat_thresh or std > noisy_thresh:
            bads.add(ch_names[i])

    corr_matrix = np.corrcoef(data)
    for i in range(n_channels):
        corrs = np.delete(corr_matrix[i], i)
        if np.nanmean(np.abs(corrs)) < correlation_thresh:
            bads.add(ch_names[i])

    raw.info['bads'] = list(bads)
    return list(bads)