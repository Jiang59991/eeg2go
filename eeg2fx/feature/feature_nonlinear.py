import numpy as np
from eeg2fx.feature.common import wrap_structured_result, auto_gc
from antropy import perm_entropy, app_entropy
from scipy.stats import skew, kurtosis
from logging_config import logger
import mne
mne.set_log_level('WARNING')

@auto_gc
def permutation_entropy(epochs, chans=None, order=3, normalize=True):
    """
    Compute permutation entropy for each epoch and channel.
    """
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
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
            pe_vals = [perm_entropy(epoch, order=order, normalize=normalize) for epoch in data[:, idx, :]]
            values.append(pe_vals)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)


@auto_gc
def approximate_entropy(epochs, chans=None, order=2, metric='chebyshev'):
    """
    Compute approximate entropy.
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
            ae_vals = [app_entropy(epoch, order=order, metric=metric) for epoch in data[:, idx, :]]
            values.append(ae_vals)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)


@auto_gc
def signal_skewness(epochs, chans=None):
    """
    Compute skewness of signal distribution per epoch.
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
            sk_vals = [skew(epoch) for epoch in data[:, idx, :]]
            values.append(sk_vals)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)


@auto_gc
def signal_kurtosis(epochs, chans=None):
    """
    Compute kurtosis per epoch and channel.
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
            ku_vals = [kurtosis(epoch) for epoch in data[:, idx, :]]
            values.append(ku_vals)
            valid_chans.append(ch)
        else:
            values.append(np.full(n_epochs, np.nan))
            valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)
