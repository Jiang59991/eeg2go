import numpy as np
import pywt
from scipy.signal import stft
from eeg2fx.feature.common import wrap_structured_result, auto_gc
from logging_config import logger
import mne
mne.set_log_level('WARNING')


@auto_gc
def wavelet_entropy(epochs, chans=None, wavelet="db4", level=4):
    """
    Compute wavelet entropy for each epoch and channel.
    """
    data = epochs.get_data()
    n_epochs = data.shape[0]
    ch_names = list(epochs.info["ch_names"])

    if chans is None:
        chans = ch_names
    elif isinstance(chans, str):
        chans = [chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in ch_names:
            idx = ch_names.index(ch)
            we_vals = []
            for epoch in data[:, idx, :]:
                coeffs = pywt.wavedec(epoch, wavelet=wavelet, level=level)
                energies = np.array([np.sum(c ** 2) for c in coeffs])
                probs = energies / np.sum(energies)
                ent = -np.sum(probs * np.log2(probs + 1e-12))
                we_vals.append(ent)
            values.append(we_vals)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)


@auto_gc
def stft_power(epochs, chans=None, band=(8, 13)):
    """
    Compute average STFT power in a given frequency band per epoch and channel.
    """
    data = epochs.get_data()
    n_epochs = data.shape[0]
    sfreq = epochs.info["sfreq"]
    ch_names = list(epochs.info["ch_names"])

    if chans is None:
        chans = ch_names
    elif isinstance(chans, str):
        chans = [chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in ch_names:
            idx = ch_names.index(ch)
            power_vals = []
            for epoch in data[:, idx, :]:
                f, t, Zxx = stft(epoch, fs=sfreq, nperseg=128)
                band_mask = (f >= band[0]) & (f <= band[1])
                power = np.abs(Zxx) ** 2
                avg_power = np.mean(power[band_mask])
                power_vals.append(avg_power)
            values.append(power_vals)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)
