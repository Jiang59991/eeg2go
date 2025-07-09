import numpy as np
from eeg2fx.feature.common import standardize_channel_name, wrap_structured_result
from mne.time_frequency import psd_array_welch
from scipy.stats import entropy as scipy_entropy
from eeg2fx.feature.common import auto_gc

@auto_gc
def bandpower(epochs, chans=None, band="alpha"):
    """
    Compute absolute band power using Welch PSD.
    """
    band_freqs = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45)
    }
    fmin, fmax = band_freqs.get(band, (8, 13))  # default to alpha

    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    sfreq = epochs.info["sfreq"]
    raw_ch_names = epochs.info["ch_names"]
    std_ch_names = [standardize_channel_name(ch) for ch in raw_ch_names]
    n_epochs = data.shape[0]

    if chans is None:
        chans = std_ch_names
    else:
        # 确保chans是列表格式
        if isinstance(chans, str):
            chans = [chans]
        # 标准化传入的通道名称
        chans = [standardize_channel_name(ch) for ch in chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in std_ch_names:
            idx = std_ch_names.index(ch)
            psds, freqs = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False)
            bp = np.sum(psds, axis=1)
            values.append(bp)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    
    result = wrap_structured_result(values, epochs, valid_chans)
    return result

@auto_gc
def relative_power(epochs, chans=None, band="alpha"):
    """
    Compute relative band power: band power / total power (1–45Hz)
    """
    band_freqs = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45)
    }
    fmin_band, fmax_band = band_freqs.get(band, (8, 13))
    fmin_total, fmax_total = 1, 45

    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    raw_ch_names = epochs.info["ch_names"]
    std_ch_names = [standardize_channel_name(ch) for ch in raw_ch_names]
    n_epochs = data.shape[0]

    if chans is None:
        chans = std_ch_names
    else:
        # 确保chans是列表格式
        if isinstance(chans, str):
            chans = [chans]
        # 标准化传入的通道名称
        chans = [standardize_channel_name(ch) for ch in chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in std_ch_names:
            idx = std_ch_names.index(ch)
            
            # 计算PSD
            psds, _ = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=fmin_band, fmax=fmax_band, verbose=False)
            
            # 计算归一化PSD
            psd_sum = np.sum(psds, axis=1, keepdims=True)
            psd_sum[psd_sum == 0] = 1e-12
            psds_norm = psds / psd_sum
            
            # 计算熵
            ent = scipy_entropy(psds_norm, base=2, axis=1)
            
            values.append(ent)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    
    result = wrap_structured_result(values, epochs, valid_chans)
    
    return result

@auto_gc
def spectral_entropy(epochs, chans=None):
    """
    Compute normalized spectral entropy over 1–45 Hz.
    """
    fmin, fmax = 1, 45
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    raw_ch_names = epochs.info["ch_names"]
    std_ch_names = [standardize_channel_name(ch) for ch in raw_ch_names]
    n_epochs = data.shape[0]

    if chans is None:
        chans = std_ch_names
    else:
        # 确保chans是列表格式
        if isinstance(chans, str):
            chans = [chans]
        # 标准化传入的通道名称
        chans = [standardize_channel_name(ch) for ch in chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in std_ch_names:
            idx = std_ch_names.index(ch)
            psds, _ = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False)
            
            # 计算归一化PSD
            psd_sum = np.sum(psds, axis=1, keepdims=True)
            psd_sum[psd_sum == 0] = 1e-12
            psds_norm = psds / psd_sum
            
            # 计算熵
            ent = scipy_entropy(psds_norm, base=2, axis=1)
            
            values.append(ent)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    
    result = wrap_structured_result(values, epochs, valid_chans)
    
    return result

@auto_gc
def alpha_peak_frequency(epochs, chans=None, fmin=7, fmax=13):
    """
    Compute alpha peak frequency - the frequency with maximum power in alpha band.
    This is one of the strongest age-related EEG features.
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    raw_ch_names = epochs.info["ch_names"]
    std_ch_names = [standardize_channel_name(ch) for ch in raw_ch_names]
    n_epochs = data.shape[0]
    if isinstance(chans, str):
        chans = [chans]

    if chans is None:
        chans = std_ch_names

    values = []
    valid_chans = []

    for ch in chans:
        if ch in std_ch_names:
            idx = std_ch_names.index(ch)
            psds, freqs = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False)
            
            # Find frequency with maximum power for each epoch
            peak_freqs = []
            for epoch_psd in psds:
                if np.any(epoch_psd > 0):
                    peak_idx = np.argmax(epoch_psd)
                    peak_freqs.append(freqs[peak_idx])
                else:
                    peak_freqs.append(np.nan)
            
            values.append(peak_freqs)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def theta_alpha_ratio(epochs, chans=None):
    """
    Compute theta/alpha power ratio.
    This ratio increases with age and is a strong age-related feature.
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    raw_ch_names = epochs.info["ch_names"]
    std_ch_names = [standardize_channel_name(ch) for ch in raw_ch_names]
    n_epochs = data.shape[0]

    if chans is None:
        chans = std_ch_names

    values = []
    valid_chans = []

    for ch in chans:
        if ch in std_ch_names:
            idx = std_ch_names.index(ch)
            
            # Compute theta power (4-8 Hz)
            psds_theta, _ = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=4, fmax=8, verbose=False)
            theta_power = np.sum(psds_theta, axis=1)
            
            # Compute alpha power (8-13 Hz)
            psds_alpha, _ = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=8, fmax=13, verbose=False)
            alpha_power = np.sum(psds_alpha, axis=1)
            
            # Compute ratio, avoid division by zero
            ratio = np.where(alpha_power > 0, theta_power / alpha_power, np.nan)
            values.append(ratio)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def spectral_edge_frequency(epochs, chans=None, percentile=95):
    """
    Compute spectral edge frequency - the frequency below which lies the specified percentile of total power.
    This decreases with age.
    """
    fmin, fmax = 1, 45
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    raw_ch_names = epochs.info["ch_names"]
    std_ch_names = [standardize_channel_name(ch) for ch in raw_ch_names]
    n_epochs = data.shape[0]

    if chans is None:
        chans = std_ch_names

    values = []
    valid_chans = []

    for ch in chans:
        if ch in std_ch_names:
            idx = std_ch_names.index(ch)
            psds, freqs = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=fmin, fmax=fmax, verbose=False)
            
            edge_freqs = []
            for epoch_psd in psds:
                if np.any(epoch_psd > 0):
                    # Cumulative sum of power
                    cumsum = np.cumsum(epoch_psd)
                    total_power = cumsum[-1]
                    threshold = total_power * percentile / 100
                    
                    # Find frequency where cumulative power reaches threshold
                    edge_idx = np.where(cumsum >= threshold)[0]
                    if len(edge_idx) > 0:
                        edge_freqs.append(freqs[edge_idx[0]])
                    else:
                        edge_freqs.append(freqs[-1])
                else:
                    edge_freqs.append(np.nan)
            
            values.append(edge_freqs)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def alpha_asymmetry(epochs, chans=None):
    """
    Compute alpha power asymmetry between C4 and C3: (C4 - C3) / (C4 + C3).
    This changes with age and is related to cognitive function.
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    raw_ch_names = epochs.info["ch_names"]
    std_ch_names = [standardize_channel_name(ch) for ch in raw_ch_names]
    n_epochs = data.shape[0]

    # Default to C3 and C4 if not specified
    if chans is None:
        chans = ["C3", "C4"]

    if len(chans) != 2:
        raise ValueError("alpha_asymmetry requires exactly 2 channels")

    values = []
    valid_chans = []

    # Get alpha power for both channels
    alpha_powers = []
    for ch in chans:
        if ch in std_ch_names:
            idx = std_ch_names.index(ch)
            psds, _ = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=8, fmax=13, verbose=False)
            alpha_power = np.sum(psds, axis=1)
            alpha_powers.append(alpha_power)
            valid_chans.append(ch)
        else:
            alpha_powers.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    # Compute asymmetry: (C4 - C3) / (C4 + C3)
    c3_power = alpha_powers[0]
    c4_power = alpha_powers[1]
    
    # Avoid division by zero
    denominator = c3_power + c4_power
    asymmetry = np.where(denominator > 0, (c4_power - c3_power) / denominator, np.nan)
    
    values.append(asymmetry)
    
    # Return single value per epoch (asymmetry is a scalar feature)
    values = np.array(values).T
    return wrap_structured_result(values, epochs, ["C4-C3_asymmetry"])
