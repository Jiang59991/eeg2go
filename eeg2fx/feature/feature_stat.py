import numpy as np
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurt
from eeg2fx.feature.common import wrap_structured_result, auto_gc
from logging_config import logger
import mne
from typing import Any, Dict, List, Optional
mne.set_log_level('WARNING')


@auto_gc
def mean(epochs, chans: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute the mean value for each epoch and channel.

    Args:
        epochs: MNE Epochs object.
        chans: List of channel names or None for all channels.

    Returns:
        Dictionary mapping channel names to a list of mean values per epoch.
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
            val = np.mean(data[:, idx, :], axis=1)
        else:
            val = np.full(n_epochs, np.nan)
        values.append(val)
        valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)


@auto_gc
def std(epochs, chans: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute the standard deviation for each epoch and channel.

    Args:
        epochs: MNE Epochs object.
        chans: List of channel names or None for all channels.

    Returns:
        Dictionary mapping channel names to a list of standard deviation values per epoch.
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
            val = np.std(data[:, idx, :], axis=1)
        else:
            val = np.full(n_epochs, np.nan)
        values.append(val)
        valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)


@auto_gc
def skew(epochs, chans: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute the skewness for each epoch and channel.

    Args:
        epochs: MNE Epochs object.
        chans: List of channel names or None for all channels.

    Returns:
        Dictionary mapping channel names to a list of skewness values per epoch.
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
            val = np.array([scipy_skew(epoch) for epoch in data[:, idx, :]])
        else:
            val = np.full(n_epochs, np.nan)
        values.append(val)
        valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)


@auto_gc
def kurt(epochs, chans: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute the kurtosis for each epoch and channel.

    Args:
        epochs: MNE Epochs object.
        chans: List of channel names or None for all channels.

    Returns:
        Dictionary mapping channel names to a list of kurtosis values per epoch.
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
            val = np.array([scipy_kurt(epoch, fisher=True, bias=False) for epoch in data[:, idx, :]])
        else:
            val = np.full(n_epochs, np.nan)
        values.append(val)
        valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)


@auto_gc
def zscore_stddev(epochs, chans: Optional[List[str]] = None) -> Dict[str, List[Dict[str, Any]]]:
    """
    Compute the mean of the standard deviation (zscore stddev) for each channel across all epochs.

    Args:
        epochs: MNE Epochs object.
        chans: List of channel names or None for all channels.

    Returns:
        Dictionary mapping channel names to a single mean stddev value.
    """
    data = epochs.get_data()
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
            std_vals = np.std(data[:, idx, :], axis=1)
            mean_std = np.mean(std_vals)
            values.append([mean_std])  # scalar
            valid_chans.append(ch)
        else:
            values.append([np.nan])
            valid_chans.append(ch)

    values = np.array(values).T  # shape: (1, n_chans)
    return wrap_structured_result(values, epochs, valid_chans)