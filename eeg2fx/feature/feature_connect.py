"""
TODO: 
绝大多数连接性特征是“通道对”（channel-pair）特征，因此 必须指定两个通道。
针对这种情况制定允许两个channel被生成的解决方案
"""
import numpy as np
from eeg2fx.feature.common import standardize_channel_name, wrap_structured_result, auto_gc
from scipy.signal import coherence, hilbert

@auto_gc
def coherence_band(epochs, chans=None, band=(8, 13)):
    """
    Compute coherence in a given band between two channels.
    chans should be exactly two channels: [ch1, ch2]
    Returns one value per epoch
    """
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    sfreq = epochs.info["sfreq"]
    ch_names = [standardize_channel_name(ch) for ch in epochs.info["ch_names"]]
    n_epochs = data.shape[0]

    if chans is None or len(chans) != 2:
        raise ValueError("Coherence requires exactly two channels.")

    ch1, ch2 = chans
    if ch1 not in ch_names or ch2 not in ch_names:
        raise ValueError(f"Channels not found: {ch1}, {ch2}")

    idx1 = ch_names.index(ch1)
    idx2 = ch_names.index(ch2)

    values = []
    for i in range(n_epochs):
        f, Cxy = coherence(data[i, idx1], data[i, idx2], fs=sfreq)
        band_mask = (f >= band[0]) & (f <= band[1])
        mean_coh = np.mean(Cxy[band_mask])
        values.append([mean_coh])

    values = np.array(values)  # shape (n_epochs, 1)
    return wrap_structured_result(values, epochs, [f"{ch1}-{ch2}"])

@auto_gc
def plv(epochs, chans=None):
    """
    Compute Phase-Locking Value (PLV) between two channels across time.
    """
    data = epochs.get_data()
    n_epochs = data.shape[0]
    sfreq = epochs.info["sfreq"]
    ch_names = [standardize_channel_name(ch) for ch in epochs.info["ch_names"]]

    if chans is None or len(chans) != 2:
        raise ValueError("PLV requires exactly two channels.")

    ch1, ch2 = chans
    if ch1 not in ch_names or ch2 not in ch_names:
        raise ValueError(f"Channels not found: {ch1}, {ch2}")

    idx1 = ch_names.index(ch1)
    idx2 = ch_names.index(ch2)

    values = []
    for i in range(n_epochs):
        sig1 = data[i, idx1, :]
        sig2 = data[i, idx2, :]

        pha1 = np.angle(hilbert(sig1))
        pha2 = np.angle(hilbert(sig2))

        complex_phase_diff = np.exp(1j * (pha1 - pha2))
        plv_val = np.abs(np.sum(complex_phase_diff)) / len(complex_phase_diff)
        values.append([plv_val])

    values = np.array(values)  # shape (n_epochs, 1)
    return wrap_structured_result(values, epochs, [f"{ch1}-{ch2}"])
