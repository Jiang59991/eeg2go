import numpy as np
from eeg2fx.feature.common import wrap_structured_result, auto_gc
from scipy.signal import coherence, hilbert
from logging_config import logger
import mne
from scipy.stats import entropy
from scipy.signal import correlate
from sklearn.metrics import mutual_info_score
import typing
mne.set_log_level('WARNING')

@auto_gc
def coherence_band(
    epochs, 
    chans: typing.Optional[typing.Union[str, typing.List[str]]] = None, 
    band: typing.Tuple[float, float] = (8, 13)
) -> dict:
    """
    Compute coherence in a given frequency band between two channels.

    Args:
        epochs: MNE epochs object containing EEG data.
        chans: Channel specification, must be exactly 2 channels.
               - String format: "C3-C4" (recommended)
               - None: will raise ValueError
        band: Frequency band tuple (fmin, fmax), default (8, 13) Hz.

    Returns:
        dict: Dictionary with coherence values for each epoch.
    """
    data = epochs.get_data()
    sfreq = epochs.info["sfreq"]
    ch_names = list(epochs.info["ch_names"])
    n_epochs = data.shape[0]

    if chans is None:
        raise ValueError("coherence_band requires exactly 2 channels to be specified")
    if isinstance(chans, str):
        if "-" in chans:
            ch1, ch2 = chans.split("-", 1)
            ch1 = ch1.strip()
            ch2 = ch2.strip()
            chans = [ch1, ch2]
        else:
            raise ValueError(f"Invalid channel format: {chans}. Use 'C3-C4' format for channel pairs")
    else:
        raise ValueError(f"Invalid channel type: {type(chans)}. Expected string or list")

    ch1, ch2 = chans
    if ch1 not in ch_names:
        raise ValueError(f"Channel {ch1} not found in available channels: {ch_names}")
    if ch2 not in ch_names:
        raise ValueError(f"Channel {ch2} not found in available channels: {ch_names}")

    idx1 = ch_names.index(ch1)
    idx2 = ch_names.index(ch2)

    values = []
    for i in range(n_epochs):
        f, Cxy = coherence(data[i, idx1], data[i, idx2], fs=sfreq)
        band_mask = (f >= band[0]) & (f <= band[1])
        mean_coh = np.mean(Cxy[band_mask])
        values.append([mean_coh])

    values = np.array(values)
    return wrap_structured_result(values, epochs, [f"{ch1}-{ch2}_coherence"])

@auto_gc
def plv(
    epochs, 
    chans: typing.Optional[typing.Union[str, typing.List[str]]] = None
) -> dict:
    """
    Compute Phase-Locking Value (PLV) between two channels across time.

    Args:
        epochs: MNE epochs object containing EEG data.
        chans: Channel specification, must be exactly 2 channels.
               - String format: "C3-C4" (recommended)
               - None: will raise ValueError

    Returns:
        dict: Dictionary with PLV values for each epoch.
    """
    data = epochs.get_data()
    n_epochs = data.shape[0]
    sfreq = epochs.info["sfreq"]
    ch_names = list(epochs.info["ch_names"])

    if chans is None:
        raise ValueError("plv requires exactly 2 channels to be specified")
    if isinstance(chans, str):
        if "-" in chans:
            ch1, ch2 = chans.split("-", 1)
            ch1 = ch1.strip()
            ch2 = ch2.strip()
            chans = [ch1, ch2]
        else:
            raise ValueError(f"Invalid channel format: {chans}. Use 'C3-C4' format for channel pairs")
    else:
        raise ValueError(f"Invalid channel type: {type(chans)}. Expected string or list")

    ch1, ch2 = chans
    if ch1 not in ch_names:
        raise ValueError(f"Channel {ch1} not found in available channels: {ch_names}")
    if ch2 not in ch_names:
        raise ValueError(f"Channel {ch2} not found in available channels: {ch_names}")

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

    values = np.array(values)
    return wrap_structured_result(values, epochs, [f"{ch1}-{ch2}_plv"])

@auto_gc
def mutual_information(
    epochs, 
    chans: typing.Optional[typing.Union[str, typing.List[str]]] = None, 
    bins: int = 20
) -> dict:
    """
    Calculate mutual information between two channels, reflecting nonlinear dependency.

    Args:
        epochs: MNE epochs object containing EEG data.
        chans: Channel specification, must be exactly 2 channels.
        bins: Number of bins for histogram discretization.

    Returns:
        dict: Dictionary with mutual information values for each epoch.
    """
    data = epochs.get_data()
    n_epochs = data.shape[0]
    ch_names = list(epochs.info["ch_names"])

    if chans is None:
        raise ValueError("mutual_information requires exactly 2 channels to be specified")
    if isinstance(chans, str):
        if "-" in chans:
            ch1, ch2 = chans.split("-", 1)
            ch1 = ch1.strip()
            ch2 = ch2.strip()
            chans = [ch1, ch2]
        else:
            raise ValueError(f"Invalid channel format: {chans}. Use 'C3-C4' format for channel pairs")
    else:
        raise ValueError(f"Invalid channel type: {type(chans)}. Expected string or list")

    ch1, ch2 = chans
    if ch1 not in ch_names:
        raise ValueError(f"Channel {ch1} not found in available channels: {ch_names}")
    if ch2 not in ch_names:
        raise ValueError(f"Channel {ch2} not found in available channels: {ch_names}")

    idx1 = ch_names.index(ch1)
    idx2 = ch_names.index(ch2)

    values = []
    for i in range(n_epochs):
        sig1 = data[i, idx1, :]
        sig2 = data[i, idx2, :]
        hist1, _ = np.histogram(sig1, bins=bins)
        hist2, _ = np.histogram(sig2, bins=bins)
        mi_val = mutual_info_score(hist1, hist2)
        values.append([mi_val])

    values = np.array(values)
    return wrap_structured_result(values, epochs, [f"{ch1}-{ch2}_mutual_info"])

@auto_gc
def cross_correlation(
    epochs, 
    chans: typing.Optional[typing.Union[str, typing.List[str]]] = None, 
    max_lag: typing.Optional[int] = None
) -> dict:
    """
    Calculate cross-correlation between two channels, reflecting linear dependency.

    Args:
        epochs: MNE epochs object containing EEG data.
        chans: Channel specification, must be exactly 2 channels.
        max_lag: Maximum lag to consider for cross-correlation.

    Returns:
        dict: Dictionary with cross-correlation values for each epoch.
    """
    data = epochs.get_data()
    n_epochs = data.shape[0]
    ch_names = list(epochs.info["ch_names"])

    if chans is None:
        raise ValueError("cross_correlation requires exactly 2 channels to be specified")
    if isinstance(chans, str):
        if "-" in chans:
            ch1, ch2 = chans.split("-", 1)
            ch1 = ch1.strip()
            ch2 = ch2.strip()
            chans = [ch1, ch2]
        else:
            raise ValueError(f"Invalid channel format: {chans}. Use 'C3-C4' format for channel pairs")
    else:
        raise ValueError(f"Invalid channel type: {type(chans)}. Expected string or list")

    ch1, ch2 = chans
    if ch1 not in ch_names:
        raise ValueError(f"Channel {ch1} not found in available channels: {ch_names}")
    if ch2 not in ch_names:
        raise ValueError(f"Channel {ch2} not found in available channels: {ch_names}")

    idx1 = ch_names.index(ch1)
    idx2 = ch_names.index(ch2)

    if max_lag is None:
        max_lag = data.shape[2] // 4

    values = []
    for i in range(n_epochs):
        sig1 = data[i, idx1, :]
        sig2 = data[i, idx2, :]
        corr = correlate(sig1, sig2, mode='full')
        max_corr = np.max(np.abs(corr))
        values.append([max_corr])

    values = np.array(values)
    return wrap_structured_result(values, epochs, [f"{ch1}-{ch2}_cross_corr"])

@auto_gc
def phase_synchronization(
    epochs, 
    chans: typing.Optional[typing.Union[str, typing.List[str]]] = None
) -> dict:
    """
    Calculate phase synchronization between two channels, reflecting the degree of phase coupling.

    Args:
        epochs: MNE epochs object containing EEG data.
        chans: Channel specification, must be exactly 2 channels.

    Returns:
        dict: Dictionary with phase synchronization values for each epoch.
    """
    data = epochs.get_data()
    n_epochs = data.shape[0]
    ch_names = list(epochs.info["ch_names"])

    if chans is None:
        raise ValueError("phase_synchronization requires exactly 2 channels to be specified")
    if isinstance(chans, str):
        if "-" in chans:
            ch1, ch2 = chans.split("-", 1)
            ch1 = ch1.strip()
            ch2 = ch2.strip()
            chans = [ch1, ch2]
        else:
            raise ValueError(f"Invalid channel format: {chans}. Use 'C3-C4' format for channel pairs")
    else:
        raise ValueError(f"Invalid channel type: {type(chans)}. Expected string or list")

    ch1, ch2 = chans
    if ch1 not in ch_names:
        raise ValueError(f"Channel {ch1} not found in available channels: {ch_names}")
    if ch2 not in ch_names:
        raise ValueError(f"Channel {ch2} not found in available channels: {ch_names}")

    idx1 = ch_names.index(ch1)
    idx2 = ch_names.index(ch2)

    values = []
    for i in range(n_epochs):
        sig1 = data[i, idx1, :]
        sig2 = data[i, idx2, :]
        analytic1 = hilbert(sig1)
        analytic2 = hilbert(sig2)
        phase1 = np.angle(analytic1)
        phase2 = np.angle(analytic2)
        phase_diff = phase1 - phase2
        sync_index = np.abs(np.mean(np.exp(1j * phase_diff)))
        values.append([sync_index])

    values = np.array(values)
    return wrap_structured_result(values, epochs, [f"{ch1}-{ch2}_phase_sync"])

@auto_gc
def amplitude_correlation(
    epochs, 
    chans: typing.Optional[typing.Union[str, typing.List[str]]] = None
) -> dict:
    """
    Calculate amplitude correlation between two channels, reflecting amplitude coupling.

    Args:
        epochs: MNE epochs object containing EEG data.
        chans: Channel specification, must be exactly 2 channels.

    Returns:
        dict: Dictionary with amplitude correlation values for each epoch.
    """
    data = epochs.get_data()
    n_epochs = data.shape[0]
    ch_names = list(epochs.info["ch_names"])

    if chans is None:
        raise ValueError("amplitude_correlation requires exactly 2 channels to be specified")
    if isinstance(chans, str):
        if "-" in chans:
            ch1, ch2 = chans.split("-", 1)
            ch1 = ch1.strip()
            ch2 = ch2.strip()
            chans = [ch1, ch2]
        else:
            raise ValueError(f"Invalid channel format: {chans}. Use 'C3-C4' format for channel pairs")
    else:
        raise ValueError(f"Invalid channel type: {type(chans)}. Expected string or list")

    ch1, ch2 = chans
    if ch1 not in ch_names:
        raise ValueError(f"Channel {ch1} not found in available channels: {ch_names}")
    if ch2 not in ch_names:
        raise ValueError(f"Channel {ch2} not found in available channels: {ch_names}")

    idx1 = ch_names.index(ch1)
    idx2 = ch_names.index(ch2)

    values = []
    for i in range(n_epochs):
        sig1 = data[i, idx1, :]
        sig2 = data[i, idx2, :]
        analytic1 = hilbert(sig1)
        analytic2 = hilbert(sig2)
        amp1 = np.abs(analytic1)
        amp2 = np.abs(analytic2)
        corr = np.corrcoef(amp1, amp2)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        values.append([corr])

    values = np.array(values)
    return wrap_structured_result(values, epochs, [f"{ch1}-{ch2}_amp_corr"])

@auto_gc
def granger_causality(
    epochs, 
    chans: typing.Optional[typing.Union[str, typing.List[str]]] = None, 
    order: int = 5
) -> dict:
    """
    Calculate Granger causality between two channels, reflecting causal influence.

    Args:
        epochs: MNE epochs object containing EEG data.
        chans: Channel specification, must be exactly 2 channels.
        order: Model order for Granger causality.

    Returns:
        dict: Dictionary with Granger causality values for each epoch.
    """
    data = epochs.get_data()
    n_epochs = data.shape[0]
    ch_names = list(epochs.info["ch_names"])

    if chans is None:
        raise ValueError("granger_causality requires exactly 2 channels to be specified")
    if isinstance(chans, str):
        if "-" in chans:
            ch1, ch2 = chans.split("-", 1)
            ch1 = ch1.strip()
            ch2 = ch2.strip()
            chans = [ch1, ch2]
        else:
            raise ValueError(f"Invalid channel format: {chans}. Use 'C3-C4' format for channel pairs")
    else:
        raise ValueError(f"Invalid channel type: {type(chans)}. Expected string or list")

    ch1, ch2 = chans
    if ch1 not in ch_names:
        raise ValueError(f"Channel {ch1} not found in available channels: {ch_names}")
    if ch2 not in ch_names:
        raise ValueError(f"Channel {ch2} not found in available channels: {ch_names}")

    idx1 = ch_names.index(ch1)
    idx2 = ch_names.index(ch2)

    values = []
    for i in range(n_epochs):
        sig1 = data[i, idx1, :]
        sig2 = data[i, idx2, :]
        try:
            n_samples = len(sig1)
            X = np.zeros((n_samples - order, order * 2))
            for j in range(order):
                X[:, j] = sig1[j:n_samples - order + j]
                X[:, order + j] = sig2[j:n_samples - order + j]
            y = sig1[order:]
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            y_pred = X @ beta
            mse_full = np.mean((y - y_pred) ** 2)
            X_reduced = X[:, :order]
            beta_reduced = np.linalg.lstsq(X_reduced, y, rcond=None)[0]
            y_pred_reduced = X_reduced @ beta_reduced
            mse_reduced = np.mean((y - y_pred_reduced) ** 2)
            if mse_reduced > 0:
                gc_index = np.log(mse_reduced / mse_full)
            else:
                gc_index = 0.0
        except:
            gc_index = 0.0
        values.append([gc_index])

    values = np.array(values)
    return wrap_structured_result(values, epochs, [f"{ch1}-{ch2}_granger"])

@auto_gc
def directed_transfer_function(
    epochs, 
    chans: typing.Optional[typing.Union[str, typing.List[str]]] = None, 
    order: int = 5
) -> dict:
    """
    Calculate directed transfer function between two channels, reflecting information flow direction.

    Args:
        epochs: MNE epochs object containing EEG data.
        chans: Channel specification, must be exactly 2 channels.
        order: Model order (not used in this simplified implementation).

    Returns:
        dict: Dictionary with directed transfer function values for each epoch.
    """
    data = epochs.get_data()
    n_epochs = data.shape[0]
    ch_names = list(epochs.info["ch_names"])

    if chans is None:
        raise ValueError("directed_transfer_function requires exactly 2 channels to be specified")
    if isinstance(chans, str):
        if "-" in chans:
            ch1, ch2 = chans.split("-", 1)
            ch1 = ch1.strip()
            ch2 = ch2.strip()
            chans = [ch1, ch2]
        else:
            raise ValueError(f"Invalid channel format: {chans}. Use 'C3-C4' format for channel pairs")
    else:
        raise ValueError(f"Invalid channel type: {type(chans)}. Expected string or list")

    ch1, ch2 = chans
    if ch1 not in ch_names:
        raise ValueError(f"Channel {ch1} not found in available channels: {ch_names}")
    if ch2 not in ch_names:
        raise ValueError(f"Channel {ch2} not found in available channels: {ch_names}")

    idx1 = ch_names.index(ch1)
    idx2 = ch_names.index(ch2)

    values = []
    for i in range(n_epochs):
        sig1 = data[i, idx1, :]
        sig2 = data[i, idx2, :]
        try:
            corr_12 = correlate(sig1, sig2, mode='full')
            corr_21 = correlate(sig2, sig1, mode='full')
            asymmetry = np.sum(np.abs(corr_12 - corr_21)) / np.sum(np.abs(corr_12 + corr_21))
        except:
            asymmetry = 0.0
        values.append([asymmetry])

    values = np.array(values)
    return wrap_structured_result(values, epochs, [f"{ch1}-{ch2}_dtf"])

@auto_gc
def synchronization_likelihood(
    epochs, 
    chans: typing.Optional[typing.Union[str, typing.List[str]]] = None, 
    threshold: float = 0.1
) -> dict:
    """
    Calculate synchronization likelihood between two channels, reflecting nonlinear synchronization.

    Args:
        epochs: MNE epochs object containing EEG data.
        chans: Channel specification, must be exactly 2 channels.
        threshold: Threshold for synchronization likelihood (not used in this simplified implementation).

    Returns:
        dict: Dictionary with synchronization likelihood values for each epoch.
    """
    data = epochs.get_data()
    n_epochs = data.shape[0]
    ch_names = list(epochs.info["ch_names"])

    if chans is None:
        raise ValueError("synchronization_likelihood requires exactly 2 channels to be specified")
    if isinstance(chans, str):
        if "-" in chans:
            ch1, ch2 = chans.split("-", 1)
            ch1 = ch1.strip()
            ch2 = ch2.strip()
            chans = [ch1, ch2]
        else:
            raise ValueError(f"Invalid channel format: {chans}. Use 'C3-C4' format for channel pairs")
    else:
        raise ValueError(f"Invalid channel type: {type(chans)}. Expected string or list")

    ch1, ch2 = chans
    if ch1 not in ch_names:
        raise ValueError(f"Channel {ch1} not found in available channels: {ch_names}")
    if ch2 not in ch_names:
        raise ValueError(f"Channel {ch2} not found in available channels: {ch_names}")

    idx1 = ch_names.index(ch1)
    idx2 = ch_names.index(ch2)

    values = []
    for i in range(n_epochs):
        sig1 = data[i, idx1, :]
        sig2 = data[i, idx2, :]
        try:
            sig1_norm = (sig1 - np.mean(sig1)) / np.std(sig1)
            sig2_norm = (sig2 - np.mean(sig2)) / np.std(sig2)
            similarity = np.corrcoef(sig1_norm, sig2_norm)[0, 1]
            if np.isnan(similarity):
                similarity = 0.0
            sync_likelihood = max(0, similarity)
        except:
            sync_likelihood = 0.0
        values.append([sync_likelihood])

    values = np.array(values)
    return wrap_structured_result(values, epochs, [f"{ch1}-{ch2}_sync_likelihood"])
