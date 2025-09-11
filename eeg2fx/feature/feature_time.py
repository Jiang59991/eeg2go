# EEG feature module: Time-domain features

import numpy as np
import gc
from eeg2fx.feature.common import wrap_structured_result, auto_gc
from logging_config import logger
import mne
from scipy import signal
from scipy.stats import skew, kurtosis, entropy
from sklearn.preprocessing import StandardScaler
mne.set_log_level('WARNING')


@auto_gc
def mean_amplitude(epochs, chans=None) -> np.recarray:
    """
    Calculate the mean absolute amplitude for each epoch and channel.

    Args:
        epochs: MNE Epochs object containing EEG data.
        chans: List of channel names or a single channel name. If None, use all channels.

    Returns:
        np.recarray: Structured array with mean amplitude for each epoch and channel.
    """
    data = epochs.get_data()
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            amp = np.abs(data[:, idx, :])
            mean_amp = np.mean(amp, axis=1)
            values.append(mean_amp)
            valid_chans.append(ch)
            del amp, mean_amp
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)
        gc.collect()

    values = np.stack(values, axis=1)
    del data
    gc.collect()

    return wrap_structured_result(values, epochs, valid_chans)


@auto_gc
def rms(epochs, chans=None) -> np.recarray:
    """
    Calculate the root mean square (RMS) value for each epoch and channel.

    Args:
        epochs: MNE Epochs object containing EEG data.
        chans: List of channel names or a single channel name. If None, use all channels.

    Returns:
        np.recarray: Structured array with RMS value for each epoch and channel.
    """
    data = epochs.get_data()
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            sq = data[:, idx, :] ** 2
            rms_val = np.sqrt(np.mean(sq, axis=1))
            values.append(rms_val)
            valid_chans.append(ch)
            del sq, rms_val
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)
        gc.collect()

    values = np.stack(values, axis=1)
    del data
    gc.collect()

    return wrap_structured_result(values, epochs, valid_chans)


@auto_gc
def zero_crossings(epochs, chans=None) -> np.recarray:
    """
    Count the number of zero crossings in each epoch and channel.

    Args:
        epochs: MNE Epochs object containing EEG data.
        chans: List of channel names or a single channel name. If None, use all channels.

    Returns:
        np.recarray: Structured array with zero crossing counts for each epoch and channel.
    """
    data = epochs.get_data()
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]

    def count_zero_crossings(signal):
        return np.sum(np.diff(np.signbit(signal)))

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            zero_cnt = []
            for epoch in data[:, idx, :]:
                zero_cnt.append(count_zero_crossings(epoch))
                del epoch
            values.append(zero_cnt)
            valid_chans.append(ch)
            del zero_cnt
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)
        gc.collect()

    values = np.stack(values, axis=1)
    del data
    gc.collect()

    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def signal_variance(epochs, chans=None) -> np.recarray:
    """
    Calculate the variance of the signal for each epoch and channel.

    Args:
        epochs: MNE Epochs object containing EEG data.
        chans: List of channel names or a single channel name. If None, use all channels.

    Returns:
        np.recarray: Structured array with variance for each epoch and channel.
    """
    data = epochs.get_data()
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            variance = np.var(data[:, idx, :], axis=1)
            values.append(variance)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def signal_skewness(epochs, chans=None) -> np.recarray:
    """
    Calculate the skewness of the signal for each epoch and channel.

    Args:
        epochs: MNE Epochs object containing EEG data.
        chans: List of channel names or a single channel name. If None, use all channels.

    Returns:
        np.recarray: Structured array with skewness for each epoch and channel.
    """
    data = epochs.get_data()
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            skewness_vals = []
            for epoch in data[:, idx, :]:
                skewness_vals.append(skew(epoch))
            values.append(skewness_vals)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def signal_kurtosis(epochs, chans=None) -> np.recarray:
    """
    Calculate the kurtosis of the signal for each epoch and channel.

    Args:
        epochs: MNE Epochs object containing EEG data.
        chans: List of channel names or a single channel name. If None, use all channels.

    Returns:
        np.recarray: Structured array with kurtosis for each epoch and channel.
    """
    data = epochs.get_data()
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            kurtosis_vals = []
            for epoch in data[:, idx, :]:
                kurtosis_vals.append(kurtosis(epoch))
            values.append(kurtosis_vals)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def peak_to_peak_amplitude(epochs, chans=None) -> np.recarray:
    """
    Calculate the peak-to-peak amplitude (max - min) for each epoch and channel.

    Args:
        epochs: MNE Epochs object containing EEG data.
        chans: List of channel names or a single channel name. If None, use all channels.

    Returns:
        np.recarray: Structured array with peak-to-peak amplitude for each epoch and channel.
    """
    data = epochs.get_data()
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            ptp_vals = np.ptp(data[:, idx, :], axis=1)
            values.append(ptp_vals)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def crest_factor(epochs, chans=None) -> np.recarray:
    """
    Calculate the crest factor (peak value divided by RMS) for each epoch and channel.

    Args:
        epochs: MNE Epochs object containing EEG data.
        chans: List of channel names or a single channel name. If None, use all channels.

    Returns:
        np.recarray: Structured array with crest factor for each epoch and channel.
    """
    data = epochs.get_data()
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            crest_factors = []
            for epoch in data[:, idx, :]:
                peak = np.max(np.abs(epoch))
                rms_val = np.sqrt(np.mean(epoch**2))
                if rms_val > 0:
                    crest_factors.append(peak / rms_val)
                else:
                    crest_factors.append(np.nan)
            values.append(crest_factors)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def shape_factor(epochs, chans=None) -> np.recarray:
    """
    Calculate the shape factor (RMS divided by mean absolute value) for each epoch and channel.

    Args:
        epochs: MNE Epochs object containing EEG data.
        chans: List of channel names or a single channel name. If None, use all channels.

    Returns:
        np.recarray: Structured array with shape factor for each epoch and channel.
    """
    data = epochs.get_data()
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            shape_factors = []
            for epoch in data[:, idx, :]:
                rms_val = np.sqrt(np.mean(epoch**2))
                mean_abs = np.mean(np.abs(epoch))
                if mean_abs > 0:
                    shape_factors.append(rms_val / mean_abs)
                else:
                    shape_factors.append(np.nan)
            values.append(shape_factors)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def impulse_factor(epochs, chans=None) -> np.recarray:
    """
    Calculate the impulse factor (peak value divided by mean absolute value) for each epoch and channel.

    Args:
        epochs: MNE Epochs object containing EEG data.
        chans: List of channel names or a single channel name. If None, use all channels.

    Returns:
        np.recarray: Structured array with impulse factor for each epoch and channel.
    """
    data = epochs.get_data()
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            impulse_factors = []
            for epoch in data[:, idx, :]:
                peak = np.max(np.abs(epoch))
                mean_abs = np.mean(np.abs(epoch))
                if mean_abs > 0:
                    impulse_factors.append(peak / mean_abs)
                else:
                    impulse_factors.append(np.nan)
            values.append(impulse_factors)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def margin_factor(epochs, chans=None) -> np.recarray:
    """
    Calculate the margin factor (peak value divided by RMS) for each epoch and channel.

    Args:
        epochs: MNE Epochs object containing EEG data.
        chans: List of channel names or a single channel name. If None, use all channels.

    Returns:
        np.recarray: Structured array with margin factor for each epoch and channel.
    """
    data = epochs.get_data()
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            margin_factors = []
            for epoch in data[:, idx, :]:
                peak = np.max(np.abs(epoch))
                rms_val = np.sqrt(np.mean(epoch**2))
                if rms_val > 0:
                    margin_factors.append(peak / rms_val)
                else:
                    margin_factors.append(np.nan)
            values.append(margin_factors)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def signal_entropy(epochs, chans=None, bins=50) -> np.recarray:
    """
    Calculate the entropy of the signal for each epoch and channel.

    Args:
        epochs: MNE Epochs object containing EEG data.
        chans: List of channel names or a single channel name. If None, use all channels.
        bins: Number of bins for histogram calculation.

    Returns:
        np.recarray: Structured array with entropy for each epoch and channel.
    """
    data = epochs.get_data()
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            entropy_vals = []
            for epoch in data[:, idx, :]:
                hist, _ = np.histogram(epoch, bins=bins, density=True)
                hist = hist[hist > 0]
                if len(hist) > 0:
                    entropy_vals.append(entropy(hist))
                else:
                    entropy_vals.append(np.nan)
            values.append(entropy_vals)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def signal_complexity(epochs, chans=None) -> np.recarray:
    """
    Calculate the signal complexity for each epoch and channel.
    Complexity is defined as the product of the standard deviation of the first-order difference and the entropy.

    Args:
        epochs: MNE Epochs object containing EEG data.
        chans: List of channel names or a single channel name. If None, use all channels.

    Returns:
        np.recarray: Structured array with complexity for each epoch and channel.
    """
    data = epochs.get_data()
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            complexity_vals = []
            for epoch in data[:, idx, :]:
                diff = np.diff(epoch)
                change_rate = np.std(diff)
                hist, _ = np.histogram(epoch, bins=50, density=True)
                hist = hist[hist > 0]
                if len(hist) > 0:
                    signal_entropy_val = entropy(hist)
                else:
                    signal_entropy_val = 0
                complexity = change_rate * signal_entropy_val
                complexity_vals.append(complexity)
            values.append(complexity_vals)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def signal_regularity(epochs, chans=None) -> np.recarray:
    """
    Calculate the regularity of the signal for each epoch and channel.
    Regularity is based on the decay rate of the autocorrelation function.

    Args:
        epochs: MNE Epochs object containing EEG data.
        chans: List of channel names or a single channel name. If None, use all channels.

    Returns:
        np.recarray: Structured array with regularity for each epoch and channel.
    """
    data = epochs.get_data()
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            regularity_vals = []
            for epoch in data[:, idx, :]:
                autocorr = np.correlate(epoch, epoch, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                autocorr = autocorr / autocorr[0]
                # Use first 10 points to estimate decay rate
                if len(autocorr) > 10:
                    decay_rate = np.mean(np.abs(np.diff(autocorr[:10])))
                    regularity_vals.append(1.0 / (1.0 + decay_rate))
                else:
                    regularity_vals.append(np.nan)
            values.append(regularity_vals)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def signal_stability(epochs, chans=None) -> np.recarray:
    """
    Calculate the stability of the signal for each epoch and channel.
    Stability is defined as 1 / (1 + coefficient of variation).

    Args:
        epochs: MNE Epochs object containing EEG data.
        chans: List of channel names or a single channel name. If None, use all channels.

    Returns:
        np.recarray: Structured array with stability for each epoch and channel.
    """
    data = epochs.get_data()
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            stability_vals = []
            for epoch in data[:, idx, :]:
                std_val = np.std(epoch)
                mean_val = np.mean(epoch)
                # Stability = 1 / (1 + coefficient of variation)
                if mean_val != 0:
                    cv = std_val / abs(mean_val)
                    stability = 1.0 / (1.0 + cv)
                else:
                    stability = 0.0
                stability_vals.append(stability)
            values.append(stability_vals)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)
