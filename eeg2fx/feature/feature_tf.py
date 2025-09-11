import numpy as np
import pywt
from scipy.signal import stft
from eeg2fx.feature.common import wrap_structured_result, auto_gc
from logging_config import logger
import mne
from typing import Any, Dict, List, Optional, Tuple
mne.set_log_level('WARNING')


@auto_gc
def wavelet_entropy(
    epochs, 
    chans: Optional[List[str]] = None, 
    wavelet: str = "db4", 
    level: int = 4
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute wavelet entropy for each epoch and channel.

    Args:
        epochs: MNE Epochs object.
        chans: List of channel names or None for all channels.
        wavelet: Wavelet type to use for decomposition.
        level: Decomposition level.

    Returns:
        Dictionary mapping channel names to a list of wavelet entropy values per epoch.
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
                # Wavelet decomposition and entropy calculation
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
def stft_power(
    epochs, 
    chans: Optional[List[str]] = None, 
    band: Tuple[float, float] = (8, 13)
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute average STFT power in a given frequency band per epoch and channel.

    Args:
        epochs: MNE Epochs object.
        chans: List of channel names or None for all channels.
        band: Tuple specifying the frequency band (fmin, fmax).

    Returns:
        Dictionary mapping channel names to a list of average STFT power values per epoch.
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
                # STFT and band power calculation
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
