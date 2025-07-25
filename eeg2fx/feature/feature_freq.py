from logging_config import logger
import numpy as np
from eeg2fx.feature.common import wrap_structured_result
from mne.time_frequency import psd_array_welch
from scipy.stats import entropy as scipy_entropy
from eeg2fx.feature.common import auto_gc
import mne
mne.set_log_level('WARNING')

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
    n_epochs = data.shape[0]
    raw_ch_names = epochs.info["ch_names"]

    if chans is None:
        chans = raw_ch_names
    else:
        # Ensure chans is in list format
        if isinstance(chans, str):
            chans = [chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            psds, freqs = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=fmin, fmax=fmax, verbose='ERROR')
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
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    else:
        # Ensure chans is in list format
        if isinstance(chans, str):
            chans = [chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            
            # Calculate PSD
            psds, _ = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=fmin_band, fmax=fmax_band, verbose='ERROR')
            
            # Calculate normalized PSD
            psd_sum = np.sum(psds, axis=1, keepdims=True)
            psd_sum[psd_sum == 0] = 1e-12
            psds_norm = psds / psd_sum
            
            # Calculate entropy
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
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    else:
        # Ensure chans is in list format
        if isinstance(chans, str):
            chans = [chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            psds, _ = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=fmin, fmax=fmax, verbose='ERROR')
            
            # Calculate normalized PSD
            psd_sum = np.sum(psds, axis=1, keepdims=True)
            psd_sum[psd_sum == 0] = 1e-12
            psds_norm = psds / psd_sum
            
            # Calculate entropy
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
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]
    
    # Standardize input channel names
    chans = [ch for ch in chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            psds, freqs = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=fmin, fmax=fmax, verbose='ERROR')
            
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
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]
    
    # Standardize input channel names
    chans = [ch for ch in chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            
            # Compute theta power (4-8 Hz)
            psds_theta, _ = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=4, fmax=8, verbose='ERROR')
            theta_power = np.sum(psds_theta, axis=1)
            
            # Compute alpha power (8-13 Hz)
            psds_alpha, _ = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=8, fmax=13, verbose='ERROR')
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
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]
    
    # Standardize input channel names
    chans = [ch for ch in chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            psds, freqs = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=fmin, fmax=fmax, verbose='ERROR')
            
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
    
    Args:
        epochs: MNE epochs object
        chans: Channel specification, must be exactly 2 channels
               - String format: "C3-C4" (recommended)
               - List format: ["C3", "C4"]
               - None: will raise ValueError
    
    Returns:
        Dictionary with asymmetry values for each epoch
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    # Process channel parameters
    if chans is None:
        raise ValueError("alpha_asymmetry requires exactly 2 channels to be specified")
    
    # Standardize channel names
    if isinstance(chans, str):
        if "-" in chans:
            # Hyphen format: "C3-C4"
            ch1, ch2 = chans.split("-", 1)
            ch1 = ch1.strip()
            ch2 = ch2.strip()
            chans = [ch1, ch2]
        else:
            raise ValueError(f"Invalid channel format: {chans}. Use 'C3-C4' format for channel pairs")
    else:
        raise ValueError(f"Invalid channel type: {type(chans)}. Expected string or list")

    # Validate channel existence
    ch1, ch2 = chans
    if ch1 not in raw_ch_names:
        raise ValueError(f"Channel {ch1} not found in available channels: {raw_ch_names}")
    if ch2 not in raw_ch_names:
        raise ValueError(f"Channel {ch2} not found in available channels: {raw_ch_names}")

    # Get channel indices
    idx1 = raw_ch_names.index(ch1)
    idx2 = raw_ch_names.index(ch2)

    # Calculate alpha power for both channels
    alpha_powers = []
    for idx in [idx1, idx2]:
        psds, _ = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=8, fmax=13, verbose='ERROR')
        alpha_power = np.sum(psds, axis=1)
        alpha_powers.append(alpha_power)

    # Calculate asymmetry: (C4 - C3) / (C4 + C3)
    c3_power = alpha_powers[0]
    c4_power = alpha_powers[1]
    
    # Avoid division by zero
    denominator = c3_power + c4_power
    asymmetry = np.where(denominator > 0, (c4_power - c3_power) / denominator, np.nan)
    
    values = np.array(asymmetry).reshape(-1, 1)  # shape (n_epochs, 1)
    return wrap_structured_result(values, epochs, [f"{ch1}-{ch2}_asymmetry"])

@auto_gc
def spectral_centroid(epochs, chans=None, fmin=1, fmax=45):
    """
    Calculate spectral centroid - weighted average frequency of power spectrum
    Reflects the main frequency components of the signal
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]
    
    chans = [ch for ch in chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            psds, freqs = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=fmin, fmax=fmax, verbose='ERROR')
            
            centroids = []
            for epoch_psd in psds:
                if np.sum(epoch_psd) > 0:
                    centroid = np.sum(freqs * epoch_psd) / np.sum(epoch_psd)
                    centroids.append(centroid)
                else:
                    centroids.append(np.nan)
            
            values.append(centroids)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def spectral_bandwidth(epochs, chans=None, fmin=1, fmax=45):
    """
    Calculate spectral bandwidth - standard deviation of power spectrum
    Reflects the dispersion degree of frequency distribution
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]
    
    chans = [ch for ch in chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            psds, freqs = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=fmin, fmax=fmax, verbose='ERROR')
            
            bandwidths = []
            for epoch_psd in psds:
                if np.sum(epoch_psd) > 0:
                    centroid = np.sum(freqs * epoch_psd) / np.sum(epoch_psd)
                    bandwidth = np.sqrt(np.sum(epoch_psd * (freqs - centroid)**2) / np.sum(epoch_psd))
                    bandwidths.append(bandwidth)
                else:
                    bandwidths.append(np.nan)
            
            values.append(bandwidths)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def spectral_rolloff(epochs, chans=None, percentile=85, fmin=1, fmax=45):
    """
    Calculate spectral rolloff frequency - cumulative frequency of specified percentage of power
    Reflects the distribution of high-frequency components
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]
    
    chans = [ch for ch in chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            psds, freqs = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=fmin, fmax=fmax, verbose='ERROR')
            
            rolloffs = []
            for epoch_psd in psds:
                if np.sum(epoch_psd) > 0:
                    cumsum = np.cumsum(epoch_psd)
                    threshold = cumsum[-1] * percentile / 100
                    rolloff_idx = np.where(cumsum >= threshold)[0]
                    if len(rolloff_idx) > 0:
                        rolloffs.append(freqs[rolloff_idx[0]])
                    else:
                        rolloffs.append(freqs[-1])
                else:
                    rolloffs.append(np.nan)
            
            values.append(rolloffs)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def spectral_flatness(epochs, chans=None, fmin=1, fmax=45):
    """
    Calculate spectral flatness - ratio of geometric mean to arithmetic mean
    Reflects the uniformity of the spectrum
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]
    
    chans = [ch for ch in chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            psds, freqs = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=fmin, fmax=fmax, verbose='ERROR')
            
            flatnesses = []
            for epoch_psd in psds:
                if np.all(epoch_psd > 0):
                    geometric_mean = np.exp(np.mean(np.log(epoch_psd)))
                    arithmetic_mean = np.mean(epoch_psd)
                    flatness = geometric_mean / arithmetic_mean
                    flatnesses.append(flatness)
                else:
                    flatnesses.append(np.nan)
            
            values.append(flatnesses)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def spectral_skewness(epochs, chans=None, fmin=1, fmax=45):
    """
    Calculate spectral skewness - asymmetry of power spectrum distribution
    Reflects the distribution characteristics of frequency components
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]
    
    chans = [ch for ch in chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            psds, freqs = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=fmin, fmax=fmax, verbose='ERROR')
            
            skewnesses = []
            for epoch_psd in psds:
                if np.sum(epoch_psd) > 0:
                    # Use power spectrum as weights to calculate weighted skewness
                    mean_freq = np.sum(freqs * epoch_psd) / np.sum(epoch_psd)
                    variance = np.sum(epoch_psd * (freqs - mean_freq)**2) / np.sum(epoch_psd)
                    if variance > 0:
                        std_dev = np.sqrt(variance)
                        skewness = np.sum(epoch_psd * ((freqs - mean_freq) / std_dev)**3) / np.sum(epoch_psd)
                        skewnesses.append(skewness)
                    else:
                        skewnesses.append(np.nan)
                else:
                    skewnesses.append(np.nan)
            
            values.append(skewnesses)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def spectral_kurtosis(epochs, chans=None, fmin=1, fmax=45):
    """
    Calculate spectral kurtosis - sharpness of power spectrum distribution
    Reflects the concentration degree of frequency components
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]
    
    chans = [ch for ch in chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            psds, freqs = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=fmin, fmax=fmax, verbose='ERROR')
            
            kurtoses = []
            for epoch_psd in psds:
                if np.sum(epoch_psd) > 0:
                    # Use power spectrum as weights to calculate weighted kurtosis
                    mean_freq = np.sum(freqs * epoch_psd) / np.sum(epoch_psd)
                    variance = np.sum(epoch_psd * (freqs - mean_freq)**2) / np.sum(epoch_psd)
                    if variance > 0:
                        std_dev = np.sqrt(variance)
                        kurtosis_val = np.sum(epoch_psd * ((freqs - mean_freq) / std_dev)**4) / np.sum(epoch_psd)
                        kurtoses.append(kurtosis_val)
                    else:
                        kurtoses.append(np.nan)
                else:
                    kurtoses.append(np.nan)
            
            values.append(kurtoses)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def band_energy_ratio(epochs, chans=None, band1="theta", band2="alpha"):
    """
    Calculate energy ratio between two frequency bands
    Common band combinations: theta/alpha, beta/alpha, gamma/beta, etc.
    """
    band_freqs = {
        "delta": (1, 4),
        "theta": (4, 8),
        "alpha": (8, 13),
        "beta": (13, 30),
        "gamma": (30, 45),
        "low_gamma": (30, 40),
        "high_gamma": (40, 100)
    }
    
    fmin1, fmax1 = band_freqs.get(band1, (4, 8))
    fmin2, fmax2 = band_freqs.get(band2, (8, 13))
    
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]
    
    chans = [ch for ch in chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            
            # Calculate power of first frequency band
            psds1, _ = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=fmin1, fmax=fmax1, verbose='ERROR')
            power1 = np.sum(psds1, axis=1)
            
            # Calculate power of second frequency band
            psds2, _ = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=fmin2, fmax=fmax2, verbose='ERROR')
            power2 = np.sum(psds2, axis=1)
            
            # Calculate ratio, avoid division by zero
            ratio = np.where(power2 > 0, power1 / power2, np.nan)
            values.append(ratio)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def spectral_complexity(epochs, chans=None, fmin=1, fmax=45):
    """
    Calculate spectral complexity - based on entropy and kurtosis of power spectrum
    Reflects the complexity of the signal
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]
    
    chans = [ch for ch in chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            psds, freqs = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=fmin, fmax=fmax, verbose='ERROR')
            
            complexities = []
            for epoch_psd in psds:
                if np.sum(epoch_psd) > 0:
                    # Normalize power spectrum
                    psd_norm = epoch_psd / np.sum(epoch_psd)
                    
                    # Calculate entropy
                    entropy = -np.sum(psd_norm * np.log2(psd_norm + 1e-12))
                    
                    # Calculate kurtosis
                    mean_freq = np.sum(freqs * psd_norm)
                    variance = np.sum(psd_norm * (freqs - mean_freq)**2)
                    if variance > 0:
                        std_dev = np.sqrt(variance)
                        kurtosis_val = np.sum(psd_norm * ((freqs - mean_freq) / std_dev)**4)
                    else:
                        kurtosis_val = 0
                    
                    # Complexity = entropy * kurtosis
                    complexity = entropy * kurtosis_val
                    complexities.append(complexity)
                else:
                    complexities.append(np.nan)
            
            values.append(complexities)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)

@auto_gc
def frequency_modulation_index(epochs, chans=None, fmin=1, fmax=45):
    """
    Calculate frequency modulation index - reflects the modulation degree of frequency components
    Based on variance and mean of power spectrum
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    raw_ch_names = epochs.info["ch_names"]
    n_epochs = data.shape[0]

    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]
    
    chans = [ch for ch in chans]

    values = []
    valid_chans = []

    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            psds, freqs = psd_array_welch(data[:, idx, :], sfreq=sfreq, fmin=fmin, fmax=fmax, verbose='ERROR')
            
            modulation_indices = []
            for epoch_psd in psds:
                if np.sum(epoch_psd) > 0:
                    # Normalize power spectrum
                    psd_norm = epoch_psd / np.sum(epoch_psd)
                    
                    # Calculate weighted mean and variance
                    mean_freq = np.sum(freqs * psd_norm)
                    variance = np.sum(psd_norm * (freqs - mean_freq)**2)
                    
                    # Modulation index = standard deviation / mean
                    if mean_freq > 0:
                        modulation_index = np.sqrt(variance) / mean_freq
                        modulation_indices.append(modulation_index)
                    else:
                        modulation_indices.append(np.nan)
                else:
                    modulation_indices.append(np.nan)
            
            values.append(modulation_indices)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)