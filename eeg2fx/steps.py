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
def filter(raw, hp=None, lp=None):
    out = raw.copy()

    hp = None if (hp is None or hp <= 0) else float(hp)
    lp = None if (lp is None or lp <= 0) else float(lp)

    if hp is None and lp is None:
        return out

    sfreq = float(out.info["sfreq"])
    nyq = sfreq / 2.0

    if hp is not None and hp >= nyq:
        raise ValueError(f"High-pass ({hp} Hz) must be < Nyquist ({nyq} Hz).")
    if lp is not None and lp >= nyq:
        raise ValueError(f"Low-pass ({lp} Hz) must be < Nyquist ({nyq} Hz).")
    if hp is not None and lp is not None and lp <= hp:
        raise ValueError(f"Low-pass ({lp}) must be greater than high-pass ({hp}).")

    out.filter(l_freq=hp, h_freq=lp, fir_design='firwin', verbose='ERROR')

    return out

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

    if isinstance(n_components, int):
        if n_components <= 0:
            raise ValueError("n_components must be positive.")
        if n_components > raw.info['nchan']:
            raise ValueError(f"n_components={n_components} exceeds number of channels ({raw.info['nchan']}).")
    elif isinstance(n_components, float):
        if not (0.0 < n_components <= 1.0):
            raise ValueError("When float, n_components must be in (0, 1].")
    else:
        raise TypeError("n_components must be int or float.")
    
    # Fit ICA
    ica_inst = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=97,
        max_iter='auto'
    )
    ica_inst.fit(raw)

    # Helper: try EOG with fallback proxies if no EOG channel exists
    def _find_bads_eog_with_fallback(ica_obj, raw_obj):
        eog_picks = mne.pick_types(raw_obj.info, eeg=False, eog=True)
        if len(eog_picks) > 0:
            eog_inds, eog_scores = ica_obj.find_bads_eog(raw_obj)
            return eog_inds, eog_scores

        # Fallback: use frontal proxies if available
        proxies = {'fp1', 'fp2', 'fpz'}
        ch_lower = {ch.lower(): ch for ch in raw_obj.ch_names}
        proxy_name = next((ch_lower[k] for k in proxies if k in ch_lower), None)
        if proxy_name is not None:
            eog_inds, eog_scores = ica_obj.find_bads_eog(raw_obj, ch_name=proxy_name)
            return eog_inds, eog_scores

        # Nothing found / no proxies
        return [], None

    # Helper: ECG only if ECG channel exists (safer in EEG-only recordings)
    def _find_bads_ecg_safe(ica_obj, raw_obj):
        ecg_picks = mne.pick_types(raw_obj.info, eeg=False, ecg=True)
        if len(ecg_picks) == 0:
            return [], None
        ecg_inds, ecg_scores = ica_obj.find_bads_ecg(raw_obj, method='correlation')
        return ecg_inds, ecg_scores

    mode = (detect_artifacts or "none").strip().lower()
    exclude = set()

    if mode == "none":
        pass
    elif mode == "eog":
        inds, _ = _find_bads_eog_with_fallback(ica_inst, raw)
        exclude.update(inds)
    elif mode == "ecg":
        inds, _ = _find_bads_ecg_safe(ica_inst, raw)
        exclude.update(inds)
    elif mode == "auto":
        eog_inds, _ = _find_bads_eog_with_fallback(ica_inst, raw)
        ecg_inds, _ = _find_bads_ecg_safe(ica_inst, raw)
        exclude.update(eog_inds)
        exclude.update(ecg_inds)
    else:
        raise ValueError("detect_artifacts must be one of {'auto','eog','ecg','none'}.")

    # Apply ICA (on a copy)
    raw_clean = raw.copy()
    if len(exclude) > 0:
        ica_inst.apply(raw_clean, exclude=list(exclude))

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
def epoch_by_event(
    raw,
    event_type,
    recording_id,
    subepoch_len=10.0,
    drop_partial=True,
    min_overlap=0.8,
    include_values=None,
):
    """
    根据数据库 `recording_events` 中的事件区间，切分为定长子窗并返回 mne.Epochs。

    典型用法：基于睡眠阶段（30s hypnogram），将每段阶段切成 10s 无重叠子窗。

    Parameters
    ----------
    raw : mne.io.Raw
        连续 EEG。
    event_type : str
        事件类型（如 'sleep_stage'）。
    recording_id : int
        对应 `recordings.id`。
    subepoch_len : float
        子窗长度（秒）。默认 10.0。
    drop_partial : bool
        True 时仅保留完全落入事件区间的子窗；False 时允许部分重叠，配合 `min_overlap` 使用。
    min_overlap : float
        当 `drop_partial=False` 时，子窗与事件区间的最小重叠比例阈值（0-1）。
    include_values : list[str] or None
        仅保留这些取值（例如 ['W','N1','N2','N3','REM']）。None 表示不过滤。

    Returns
    -------
    mne.Epochs
        每个子窗一个 epoch，`event_id` 对应事件 value 的枚举映射。
    """

    if recording_id is None:
        raise ValueError("recording_id must be provided")

    # 读取该 recording 的事件（包含 onset / duration / value）
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute(
        "SELECT onset, duration, value FROM recording_events WHERE recording_id = ? AND event_type = ? ORDER BY onset",
        (recording_id, event_type),
    )
    rows = c.fetchall()
    conn.close()

    if not rows:
        raise ValueError(f"No events of type '{event_type}' found for recording {recording_id}")

    sfreq = float(raw.info["sfreq"])

    # 可选按取值过滤（例如仅保留 AASM 五阶段）
    if include_values is not None:
        include_values_set = set([str(v) for v in include_values])
        rows = [r for r in rows if str(r[2]) in include_values_set]
        if not rows:
            raise ValueError(
                f"No events left after filtering include_values={include_values} for recording {recording_id}"
            )

    # 生成子窗事件（样本点）
    # events: (n_events, 3) 数组，第三列为整数标签；event_id 为 {label: code}
    value_to_code = {}
    events_list = []

    for onset, duration, value in rows:
        try:
            onset = float(onset)
            duration = float(duration) if duration is not None else 0.0
        except Exception:
            continue

        if duration <= 0:
            # 无持续时长就跳过（睡眠阶段通常有 duration=30s）
            continue

        window_start = onset
        window_end = onset + duration

        # 按 subepoch_len 滚动切分
        t = window_start
        while t + 1e-9 < window_end:  # 容忍浮点边界
            candidate_start = t
            candidate_end = t + subepoch_len

            # 计算与事件区间的重叠
            inter_start = max(candidate_start, window_start)
            inter_end = min(candidate_end, window_end)
            overlap = max(0.0, inter_end - inter_start)

            keep = False
            if drop_partial:
                keep = (candidate_start >= window_start - 1e-9) and (candidate_end <= window_end + 1e-9)
            else:
                keep = (overlap / subepoch_len) >= float(min_overlap)

            if keep:
                if value not in value_to_code:
                    value_to_code[value] = len(value_to_code) + 1  # 从1开始编码
                code = value_to_code[value]

                sample = int(round(candidate_start * sfreq))
                events_list.append([sample, 0, code])

            # 子窗无重叠滚动
            t += subepoch_len

    if len(events_list) == 0:
        raise ValueError("No sub-epochs generated from events. Check parameters and event table.")

    events = np.array(events_list, dtype=int)
    event_id = {str(k): v for k, v in value_to_code.items()}

    # 用 mne.Epochs 切分，tmin=0, tmax=subepoch_len
    epochs = mne.Epochs(
        raw,
        events,
        event_id=event_id,
        tmin=0.0,
        tmax=subepoch_len,
        baseline=None,
        reject=None,
        flat=None,
        preload=True,
        verbose='ERROR',
    )

    logger.info(
        f"Created {len(epochs)} sub-epochs (len={subepoch_len}s) from {len(rows)} '{event_type}' events; labels={list(event_id.keys())}"
    )

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