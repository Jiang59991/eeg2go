import numpy as np
from eeg2fx.feature.common import wrap_structured_result, auto_gc
from antropy import perm_entropy, app_entropy
from scipy.stats import skew, kurtosis
from logging_config import logger
import mne
from typing import Any, Dict, List, Optional
mne.set_log_level('WARNING')

@auto_gc
def permutation_entropy(
    epochs, 
    chans: Optional[List[str]] = None, 
    order: int = 3, 
    normalize: bool = True
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute permutation entropy for each epoch and channel.

    Args:
        epochs: MNE Epochs object.
        chans: List of channel names or None for all channels.
        order: Embedding dimension for permutation entropy.
        normalize: Whether to normalize the entropy value.

    Returns:
        Dictionary mapping channel names to a list of permutation entropy results per epoch.
    """
    data = epochs.get_data()
    n_epochs = data.shape[0]
    raw_ch_names = epochs.info["ch_names"]
    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]

    values = []
    valid_chans = []
    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            # Compute permutation entropy for each epoch of the channel
            pe_vals = [perm_entropy(epoch, order=order, normalize=normalize) for epoch in data[:, idx, :]]
            values.append(pe_vals)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)


@auto_gc
def approximate_entropy(
    epochs, 
    chans: Optional[List[str]] = None, 
    order: int = 2, 
    metric: str = 'chebyshev'
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute approximate entropy for each epoch and channel.

    Args:
        epochs: MNE Epochs object.
        chans: List of channel names or None for all channels.
        order: Embedding dimension for approximate entropy.
        metric: Distance metric to use.

    Returns:
        Dictionary mapping channel names to a list of approximate entropy results per epoch.
    """
    data = epochs.get_data()
    n_epochs = data.shape[0]
    raw_ch_names = epochs.info["ch_names"]
    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]

    values = []
    valid_chans = []
    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            # Compute approximate entropy for each epoch of the channel
            ae_vals = [app_entropy(epoch, order=order, metric=metric) for epoch in data[:, idx, :]]
            values.append(ae_vals)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)


@auto_gc
def signal_skewness(
    epochs, 
    chans: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute skewness of signal distribution per epoch and channel.

    Args:
        epochs: MNE Epochs object.
        chans: List of channel names or None for all channels.

    Returns:
        Dictionary mapping channel names to a list of skewness values per epoch.
    """
    data = epochs.get_data()
    n_epochs = data.shape[0]
    raw_ch_names = epochs.info["ch_names"]
    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]

    values = []
    valid_chans = []
    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            # Compute skewness for each epoch of the channel
            sk_vals = [skew(epoch) for epoch in data[:, idx, :]]
            values.append(sk_vals)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)


@auto_gc
def signal_kurtosis(
    epochs, 
    chans: Optional[List[str]] = None
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute kurtosis for each epoch and channel.

    Args:
        epochs: MNE Epochs object.
        chans: List of channel names or None for all channels.

    Returns:
        Dictionary mapping channel names to a list of kurtosis values per epoch.
    """
    data = epochs.get_data()
    n_epochs = data.shape[0]
    raw_ch_names = epochs.info["ch_names"]
    if chans is None:
        chans = raw_ch_names
    elif isinstance(chans, str):
        chans = [chans]

    values = []
    valid_chans = []
    for ch in chans:
        if ch in raw_ch_names:
            idx = raw_ch_names.index(ch)
            # Compute kurtosis for each epoch of the channel
            ku_vals = [kurtosis(epoch) for epoch in data[:, idx, :]]
            values.append(ku_vals)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)
