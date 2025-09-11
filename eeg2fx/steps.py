import mne
mne.set_log_level('WARNING')
import numpy as np
import sqlite3
import os
import gc
from .feature.common import auto_gc
from logging_config import logger
import csv
from typing import Optional, List, Dict, Any

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))
MAX_MEMORY_GB = 1  # Memory limit in GB

class RecordingTooLargeError(Exception):
    """Exception raised when a recording file is too large to process."""
    pass

def build_channel_rename_dict(lookup_csv_path: str) -> Dict[str, str]:
    """
    Build a dictionary for channel renaming from a CSV lookup table.

    Args:
        lookup_csv_path (str): Path to the channel lookup CSV.

    Returns:
        Dict[str, str]: Mapping from source channel names to destination names.
    """
    rename_dict = {}
    with open(lookup_csv_path, newline='', encoding='utf-8') as csvfile:
        reader = csv.reader(csvfile)
        next(reader)  # Skip header
        for row in reader:
            dest = row[0].strip()
            for src in row[1:]:
                src = src.strip()
                if src:
                    rename_dict[src] = dest
    return rename_dict

LOOKUP_CSV_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "chanlookup.csv"))
RENAME_DICT = build_channel_rename_dict(LOOKUP_CSV_PATH)

def load_recording(recording_id: int) -> mne.io.Raw:
    """
    Load an EEG recording from the database by its ID.

    Args:
        recording_id (int): The ID of the recording.

    Returns:
        mne.io.Raw: The loaded EEG recording.
    """
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

    if duration and channels and sampling_rate:
        estimated_memory_mb = (duration * sampling_rate * channels * 4) / (1024 * 1024)  # 4 bytes per float32
        logger.info(f"[load_recording] Recording {recording_id}: {duration:.1f}s, {channels} channels, {sampling_rate} Hz")
        logger.info(f"[load_recording] Estimated memory usage: {estimated_memory_mb:.1f} MB")
        if estimated_memory_mb > MAX_MEMORY_GB * 1024:
            logger.warning(f"[load_recording] Recording {recording_id} is too large ({estimated_memory_mb:.2f} MB), exceeding limit of {MAX_MEMORY_GB*1024:.2f} MB. Skipping.")
            raise RecordingTooLargeError(f"Recording too large to process ({estimated_memory_mb:.2f} MB)")
        if estimated_memory_mb > 900:
            logger.info(f"[load_recording] Large file detected, using memory mapping")
            raw = mne.io.read_raw_edf(filepath, preload='auto', verbose='ERROR')
        else:
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose='ERROR')
    else:
        logger.info(f"[load_recording] Missing metadata, using memory mapping")
        raw = mne.io.read_raw_edf(filepath, preload='auto', verbose='ERROR')

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
def filter(raw: mne.io.Raw, hp: Optional[float] = None, lp: Optional[float] = None) -> mne.io.Raw:
    """
    Apply bandpass filtering to the raw EEG data.

    Args:
        raw (mne.io.Raw): The raw EEG data.
        hp (Optional[float]): High-pass frequency in Hz.
        lp (Optional[float]): Low-pass frequency in Hz.

    Returns:
        mne.io.Raw: Filtered EEG data.
    """
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
def notch_filter(raw: mne.io.Raw, freq: float) -> mne.io.Raw:
    """
    Apply a notch filter to the raw EEG data.

    Args:
        raw (mne.io.Raw): The raw EEG data.
        freq (float): Frequency to notch filter (Hz).

    Returns:
        mne.io.Raw: Notch-filtered EEG data.
    """
    if freq is None:
        raise ValueError("Notch filter frequency must be specified.")
    sfreq = raw.info.get('sfreq')
    if freq >= sfreq / 2:
        raise ValueError(f"Notch frequency {freq} is above Nyquist frequency.")
    return raw.copy().notch_filter(freqs=[freq])

@auto_gc
def reref(raw: mne.io.Raw, method: str, original_reference: Optional[str] = None) -> mne.io.Raw:
    """
    Apply EEG re-referencing strategy.

    Args:
        raw (mne.io.Raw): The raw EEG recording.
        method (str): The re-reference strategy. Options: 'average', 'linked_mastoid'.
        original_reference (Optional[str]): The original reference used at recording time.

    Returns:
        mne.io.Raw: The re-referenced raw object.
    """
    original_reference = (original_reference or "").lower()
    method = method.lower()

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
def resample(raw: mne.io.Raw, sfreq: float) -> mne.io.Raw:
    """
    Resample the raw EEG data to a new sampling frequency.

    Args:
        raw (mne.io.Raw): The raw EEG data.
        sfreq (float): Target sampling frequency in Hz.

    Returns:
        mne.io.Raw: Resampled EEG data.
    """
    if sfreq is None:
        raise ValueError("Target sampling rate must be explicitly specified.")
    original_sfreq = raw.info.get('sfreq')
    if np.isclose(sfreq, original_sfreq):
        return raw.copy()
    return raw.copy().resample(sfreq=sfreq)

@auto_gc
def ica(raw: mne.io.Raw, n_components: Any, detect_artifacts: str) -> mne.io.Raw:
    """
    Perform fully automated ICA for artifact removal.

    Args:
        raw (mne.io.Raw): Continuous EEG recording.
        n_components (int or float): Number of components to retain.
        detect_artifacts (str): Artifact detection strategy: 'eog', 'ecg', 'auto', or 'none'.

    Returns:
        mne.io.Raw: EEG data after ICA cleaning.
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

    ica_inst = mne.preprocessing.ICA(
        n_components=n_components,
        random_state=97,
        max_iter='auto'
    )
    ica_inst.fit(raw)

    def _find_bads_eog_with_fallback(ica_obj, raw_obj):
        eog_picks = mne.pick_types(raw_obj.info, eeg=False, eog=True)
        if len(eog_picks) > 0:
            eog_inds, eog_scores = ica_obj.find_bads_eog(raw_obj)
            return eog_inds, eog_scores
        proxies = {'fp1', 'fp2', 'fpz'}
        ch_lower = {ch.lower(): ch for ch in raw_obj.ch_names}
        proxy_name = next((ch_lower[k] for k in proxies if k in ch_lower), None)
        if proxy_name is not None:
            eog_inds, eog_scores = ica_obj.find_bads_eog(raw_obj, ch_name=proxy_name)
            return eog_inds, eog_scores
        return [], None

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

    raw_clean = raw.copy()
    if len(exclude) > 0:
        ica_inst.apply(raw_clean, exclude=list(exclude))

    return raw_clean

@auto_gc
def epoch(raw: mne.io.Raw, duration: float = 5.0) -> mne.Epochs:
    """
    Segment raw data into fixed-length epochs.

    Args:
        raw (mne.io.Raw): The raw EEG data.
        duration (float): Length of each epoch in seconds.

    Returns:
        mne.Epochs: Epoched EEG data.
    """
    events = mne.make_fixed_length_events(raw, duration=duration)
    epochs = mne.Epochs(raw, events, tmin=0, tmax=duration, baseline=None, preload=True, verbose='ERROR')
    return epochs

@auto_gc
def epoch_by_event(
    raw: mne.io.Raw,
    event_type: str,
    recording_id: int,
    subepoch_len: float = 10.0,
    drop_partial: bool = True,
    min_overlap: float = 0.8,
    include_values: Optional[List[str]] = None,
) -> mne.Epochs:
    """
    Segment raw data into fixed-length sub-epochs based on event intervals from the database.

    Args:
        raw (mne.io.Raw): Continuous EEG.
        event_type (str): Event type (e.g., 'sleep_stage').
        recording_id (int): Recording ID.
        subepoch_len (float): Sub-epoch length in seconds.
        drop_partial (bool): If True, only keep sub-epochs fully within event intervals.
        min_overlap (float): Minimum overlap ratio for partial sub-epochs.
        include_values (Optional[List[str]]): Only keep these event values.

    Returns:
        mne.Epochs: Epoched EEG data, one epoch per sub-epoch.
    """
    if recording_id is None:
        raise ValueError("recording_id must be provided")

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

    if include_values is not None:
        include_values_set = set([str(v) for v in include_values])
        rows = [r for r in rows if str(r[2]) in include_values_set]
        if not rows:
            raise ValueError(
                f"No events left after filtering include_values={include_values} for recording {recording_id}"
            )

    value_to_code = {}
    events_list = []

    for onset, duration, value in rows:
        try:
            onset = float(onset)
            duration = float(duration) if duration is not None else 0.0
        except Exception:
            continue

        if duration <= 0:
            continue

        window_start = onset
        window_end = onset + duration

        t = window_start
        while t + 1e-9 < window_end:
            candidate_start = t
            candidate_end = t + subepoch_len

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
                    value_to_code[value] = len(value_to_code) + 1
                code = value_to_code[value]
                sample = int(round(candidate_start * sfreq))
                events_list.append([sample, 0, code])
            t += subepoch_len

    if len(events_list) == 0:
        raise ValueError("No sub-epochs generated from events. Check parameters and event table.")

    events = np.array(events_list, dtype=int)
    event_id = {str(k): v for k, v in value_to_code.items()}

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
def reject_high_amplitude(epochs: mne.Epochs, threshold_uv: float) -> mne.Epochs:
    """
    Reject epochs with any EEG channel exceeding the given amplitude.

    Args:
        epochs (mne.Epochs): Input EEG epochs.
        threshold_uv (float): Absolute amplitude threshold (in microvolts).

    Returns:
        mne.Epochs: Cleaned epochs after rejecting high-amplitude artifacts.
    """
    threshold_v = threshold_uv * 1e-6  # convert µV to V
    clean_epochs = epochs.copy().drop_bad(reject={"eeg": threshold_v})
    return clean_epochs

@auto_gc
def zscore(epochs: mne.Epochs, mode: str) -> mne.Epochs:
    """
    Standardize epochs across time for each channel.

    Args:
        epochs (mne.Epochs): Input EEG epochs.
        mode (str): Normalization strategy. 'per_epoch' or 'global'.

    Returns:
        mne.Epochs: Z-scored epochs.
    """
    data = epochs.get_data()
    if mode == "per_epoch":
        mean = np.mean(data, axis=2, keepdims=True)
        std = np.std(data, axis=2, keepdims=True)
    elif mode == "global":
        mean = np.mean(data, axis=(0, 2), keepdims=True)
        std = np.std(data, axis=(0, 2), keepdims=True)
    else:
        raise ValueError(f"Unsupported mode '{mode}'. Choose from 'per_epoch' or 'global'.")

    std[std == 0] = 1e-6
    data_z = (data - mean) / std

    zscored = epochs.copy()
    zscored._data = data_z
    return zscored

@auto_gc
def detect_bad_channels(
    raw: mne.io.Raw,
    flat_thresh: float = 1e-7,
    noisy_thresh: float = 1e-4,
    correlation_thresh: float = 0.4
) -> List[str]:
    """
    Detect bad EEG channels based on flatness, noise, and correlation.

    Args:
        raw (mne.io.Raw): The raw EEG data.
        flat_thresh (float): Threshold for flat channels (std below this).
        noisy_thresh (float): Threshold for noisy channels (std above this).
        correlation_thresh (float): Minimum mean absolute correlation with other channels.

    Returns:
        List[str]: List of detected bad channel names.
    """
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