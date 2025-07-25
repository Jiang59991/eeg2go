"""
TODO: 
Most connectivity features are "channel-pair" features, therefore exactly two channels must be specified.
Develop a solution that allows two channels to be generated for this case
"""
import numpy as np
from eeg2fx.feature.common import wrap_structured_result, auto_gc
from scipy.signal import coherence, hilbert
from logging_config import logger
import mne
from scipy.stats import entropy
from scipy.signal import correlate
from sklearn.metrics import mutual_info_score
mne.set_log_level('WARNING')

@auto_gc
def coherence_band(epochs, chans=None, band=(8, 13)):
    """
    Compute coherence in a given band between two channels.
    
    Args:
        epochs: MNE epochs object
        chans: Channel specification, must be exactly 2 channels
               - String format: "C3-C4" (recommended)
               - None: will raise ValueError
        band: Frequency band tuple (fmin, fmax), default (8, 13) Hz
    
    Returns:
        Dictionary with coherence values for each epoch
    """
    data = epochs.get_data()  # shape: (n_epochs, n_channels, n_times)
    sfreq = epochs.info["sfreq"]
    ch_names = list(epochs.info["ch_names"])
    n_epochs = data.shape[0]

    # Process channel parameters
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

    values = np.array(values)  # shape (n_epochs, 1)
    return wrap_structured_result(values, epochs, [f"{ch1}-{ch2}_coherence"])

@auto_gc
def plv(epochs, chans=None):
    """
    Compute Phase-Locking Value (PLV) between two channels across time.
    
    Args:
        epochs: MNE epochs object
        chans: Channel specification, must be exactly 2 channels
               - String format: "C3-C4" (recommended)
               - None: will raise ValueError
    
    Returns:
        Dictionary with PLV values for each epoch
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
def mutual_information(epochs, chans=None, bins=20):
    """
    Calculate mutual information between two channels
    Reflects nonlinear dependency between channels
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

    # Calculate mutual information
    values = []
    for i in range(n_epochs):
        sig1 = data[i, idx1, :]
        sig2 = data[i, idx2, :]
        
        # Discretize continuous signals into histograms
        hist1, _ = np.histogram(sig1, bins=bins)
        hist2, _ = np.histogram(sig2, bins=bins)
        
        # Calculate mutual information
        mi_val = mutual_info_score(hist1, hist2)
        values.append([mi_val])

    values = np.array(values)
    return wrap_structured_result(values, epochs, [f"{ch1}-{ch2}_mutual_info"])

@auto_gc
def cross_correlation(epochs, chans=None, max_lag=None):
    """
    Calculate cross-correlation between two channels
    Reflects linear dependency between channels
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

    # Set maximum lag
    if max_lag is None:
        max_lag = data.shape[2] // 4

    # Calculate cross-correlation
    values = []
    for i in range(n_epochs):
        sig1 = data[i, idx1, :]
        sig2 = data[i, idx2, :]
        
        # Calculate cross-correlation
        corr = correlate(sig1, sig2, mode='full')
        
        # Find maximum correlation value
        max_corr = np.max(np.abs(corr))
        values.append([max_corr])

    values = np.array(values)
    return wrap_structured_result(values, epochs, [f"{ch1}-{ch2}_cross_corr"])

@auto_gc
def phase_synchronization(epochs, chans=None):
    """
    Calculate phase synchronization between two channels
    Reflects the degree of phase coupling between channels
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

    # Validate channel existence
    ch1, ch2 = chans
    if ch1 not in ch_names:
        raise ValueError(f"Channel {ch1} not found in available channels: {ch_names}")
    if ch2 not in ch_names:
        raise ValueError(f"Channel {ch2} not found in available channels: {ch_names}")

    # Get channel indices
    idx1 = ch_names.index(ch1)
    idx2 = ch_names.index(ch2)

    # Calculate phase synchronization
    values = []
    for i in range(n_epochs):
        sig1 = data[i, idx1, :]
        sig2 = data[i, idx2, :]
        
        # Calculate analytic signals
        analytic1 = hilbert(sig1)
        analytic2 = hilbert(sig2)
        
        # Calculate instantaneous phases
        phase1 = np.angle(analytic1)
        phase2 = np.angle(analytic2)
        
        # Calculate phase difference
        phase_diff = phase1 - phase2
        
        # Calculate phase synchronization index
        sync_index = np.abs(np.mean(np.exp(1j * phase_diff)))
        values.append([sync_index])

    values = np.array(values)
    return wrap_structured_result(values, epochs, [f"{ch1}-{ch2}_phase_sync"])

@auto_gc
def amplitude_correlation(epochs, chans=None):
    """
    Calculate amplitude correlation between two channels
    Reflects amplitude coupling between channels
    """
    data = epochs.get_data()
    n_epochs = data.shape[0]
    ch_names = list(epochs.info["ch_names"])

    # Process channel parameters
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

    # Calculate amplitude correlation
    values = []
    for i in range(n_epochs):
        sig1 = data[i, idx1, :]
        sig2 = data[i, idx2, :]
        
        # Calculate analytic signals
        analytic1 = hilbert(sig1)
        analytic2 = hilbert(sig2)
        
        # Calculate instantaneous amplitudes
        amp1 = np.abs(analytic1)
        amp2 = np.abs(analytic2)
        
        # Calculate amplitude correlation
        corr = np.corrcoef(amp1, amp2)[0, 1]
        if np.isnan(corr):
            corr = 0.0
        values.append([corr])

    values = np.array(values)
    return wrap_structured_result(values, epochs, [f"{ch1}-{ch2}_amp_corr"])

@auto_gc
def granger_causality(epochs, chans=None, order=5):
    """
    Calculate Granger causality between two channels
    Reflects causal influence relationship between channels
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

    # Calculate Granger causality
    values = []
    for i in range(n_epochs):
        sig1 = data[i, idx1, :]
        sig2 = data[i, idx2, :]
        
        # Simplified Granger causality calculation
        # Use linear regression residual variance ratio as approximation
        try:
            # Build lag matrix
            n_samples = len(sig1)
            X = np.zeros((n_samples - order, order * 2))
            
            for j in range(order):
                X[:, j] = sig1[j:n_samples - order + j]
                X[:, order + j] = sig2[j:n_samples - order + j]
            
            y = sig1[order:]
            
            # Calculate regression coefficients
            beta = np.linalg.lstsq(X, y, rcond=None)[0]
            
            # Calculate prediction error
            y_pred = X @ beta
            mse_full = np.mean((y - y_pred) ** 2)
            
            # Use only sig1 lags
            X_reduced = X[:, :order]
            beta_reduced = np.linalg.lstsq(X_reduced, y, rcond=None)[0]
            y_pred_reduced = X_reduced @ beta_reduced
            mse_reduced = np.mean((y - y_pred_reduced) ** 2)
            
            # Granger causality index
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
def directed_transfer_function(epochs, chans=None, order=5):
    """
    Calculate directed transfer function between two channels
    Reflects information flow direction between channels
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

    # Calculate directed transfer function
    values = []
    for i in range(n_epochs):
        sig1 = data[i, idx1, :]
        sig2 = data[i, idx2, :]
        
        try:
            # Simplified DTF calculation
            # Use cross-correlation asymmetry as approximation
            corr_12 = correlate(sig1, sig2, mode='full')
            corr_21 = correlate(sig2, sig1, mode='full')
            
            # Calculate asymmetry
            asymmetry = np.sum(np.abs(corr_12 - corr_21)) / np.sum(np.abs(corr_12 + corr_21))
            
        except:
            asymmetry = 0.0
        
        values.append([asymmetry])

    values = np.array(values)
    return wrap_structured_result(values, epochs, [f"{ch1}-{ch2}_dtf"])

@auto_gc
def synchronization_likelihood(epochs, chans=None, threshold=0.1):
    """
    Calculate synchronization likelihood between two channels
    Reflects the degree of nonlinear synchronization between channels
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

    # Calculate synchronization likelihood
    values = []
    for i in range(n_epochs):
        sig1 = data[i, idx1, :]
        sig2 = data[i, idx2, :]
        
        try:
            # Simplified synchronization likelihood calculation
            # Use signal similarity as approximation
            # Normalize signals
            sig1_norm = (sig1 - np.mean(sig1)) / np.std(sig1)
            sig2_norm = (sig2 - np.mean(sig2)) / np.std(sig2)
            
            # Calculate similarity
            similarity = np.corrcoef(sig1_norm, sig2_norm)[0, 1]
            if np.isnan(similarity):
                similarity = 0.0
            
            # Convert to synchronization likelihood
            sync_likelihood = max(0, similarity)
            
        except:
            sync_likelihood = 0.0
        
        values.append([sync_likelihood])

    values = np.array(values)
    return wrap_structured_result(values, epochs, [f"{ch1}-{ch2}_sync_likelihood"])
