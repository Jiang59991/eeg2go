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
def mean_amplitude(epochs, chans=None):
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
def rms(epochs, chans=None):
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
def zero_crossings(epochs, chans=None):
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
def signal_variance(epochs, chans=None):
    """
    Calculate signal variance - reflects the fluctuation degree of the signal
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
def signal_skewness(epochs, chans=None):
    """
    Calculate signal skewness - reflects the asymmetry of signal distribution
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
def signal_kurtosis(epochs, chans=None):
    """
    Calculate signal kurtosis - reflects the sharpness of signal distribution
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
def peak_to_peak_amplitude(epochs, chans=None):
    """
    Calculate peak-to-peak amplitude - difference between maximum and minimum values
    Reflects the dynamic range of the signal
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
def crest_factor(epochs, chans=None):
    """
    Calculate crest factor - ratio of peak value to RMS
    Reflects the peak characteristics of the signal
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
def shape_factor(epochs, chans=None):
    """
    Calculate shape factor - ratio of RMS to mean absolute value
    Reflects the shape characteristics of the signal
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
def impulse_factor(epochs, chans=None):
    """
    Calculate impulse factor - ratio of peak value to mean absolute value
    Reflects the impulse characteristics of the signal
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
def margin_factor(epochs, chans=None):
    """
    Calculate margin factor - ratio of peak value to root mean square
    Reflects the margin characteristics of the signal
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
def signal_entropy(epochs, chans=None, bins=50):
    """
    Calculate signal entropy - reflects the randomness and complexity of the signal
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
                # Calculate histogram
                hist, _ = np.histogram(epoch, bins=bins, density=True)
                # Remove zero probabilities
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
def signal_complexity(epochs, chans=None):
    """
    Calculate signal complexity - based on signal change rate and entropy
    Reflects the complexity degree of the signal
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
                # Calculate first-order difference of the signal
                diff = np.diff(epoch)
                
                # Calculate change rate
                change_rate = np.std(diff)
                
                # Calculate signal entropy
                hist, _ = np.histogram(epoch, bins=50, density=True)
                hist = hist[hist > 0]
                if len(hist) > 0:
                    signal_entropy_val = entropy(hist)
                else:
                    signal_entropy_val = 0
                
                # Complexity = change rate * entropy
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
def signal_regularity(epochs, chans=None):
    """
    Calculate signal regularity - based on autocorrelation decay rate
    Reflects the periodic characteristics of the signal
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
                # Calculate autocorrelation
                autocorr = np.correlate(epoch, epoch, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Normalize
                autocorr = autocorr / autocorr[0]
                
                # Calculate decay rate (average decay rate of first 10 points)
                if len(autocorr) > 10:
                    decay_rate = np.mean(np.abs(np.diff(autocorr[:10])))
                    regularity_vals.append(1.0 / (1.0 + decay_rate))  # Convert to regularity index
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
def signal_stability(epochs, chans=None):
    """
    Calculate signal stability - based on signal standard deviation and mean
    Reflects the stability degree of the signal
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
