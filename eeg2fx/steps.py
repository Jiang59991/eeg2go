import mne
import numpy as np
import sqlite3
import os
import gc
from eeg2fx.feature.common import auto_gc

DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db"))

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

    # Calculate estimated memory usage
    if duration and channels and sampling_rate:
        estimated_memory_mb = (duration * sampling_rate * channels * 4) / (1024 * 1024)  # 4 bytes per float32
        print(f"[load_recording] Recording {recording_id}: {duration:.1f}s, {channels} channels, {sampling_rate} Hz")
        print(f"[load_recording] Estimated memory usage: {estimated_memory_mb:.1f} MB")
        
        # Use preload=False for large files (>500MB estimated)
        if estimated_memory_mb > 500:
            print(f"[load_recording] Large file detected, using preload=False to save memory")
            raw = mne.io.read_raw_edf(filepath, preload=False, verbose='ERROR')
        else:
            raw = mne.io.read_raw_edf(filepath, preload=True, verbose='ERROR')
    else:
        # Fallback to preload=False if metadata is missing
        print(f"[load_recording] Missing metadata, using preload=False")
        raw = mne.io.read_raw_edf(filepath, preload=False, verbose='ERROR')
    
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
