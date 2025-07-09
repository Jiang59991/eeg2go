import numpy as np
from scipy.stats import skew as scipy_skew, kurtosis as scipy_kurt
from eeg2fx.feature.common import standardize_channel_name, wrap_structured_result, auto_gc


@auto_gc
def mean(epochs, chans=None):
    data = epochs.get_data()
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
            val = np.mean(data[:, idx, :], axis=1)
        else:
            val = np.full(n_epochs, np.nan)
        values.append(val)
        valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)


@auto_gc
def std(epochs, chans=None):
    data = epochs.get_data()
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
            val = np.std(data[:, idx, :], axis=1)
        else:
            val = np.full(n_epochs, np.nan)
        values.append(val)
        valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)


@auto_gc
def skew(epochs, chans=None):
    data = epochs.get_data()
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
            val = np.array([scipy_skew(epoch) for epoch in data[:, idx, :]])
        else:
            val = np.full(n_epochs, np.nan)
        values.append(val)
        valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)


@auto_gc
def kurt(epochs, chans=None):
    data = epochs.get_data()
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
            val = np.array([scipy_kurt(epoch, fisher=True, bias=False) for epoch in data[:, idx, :]])
        else:
            val = np.full(n_epochs, np.nan)
        values.append(val)
        valid_chans.append(ch)

    values = np.stack(values, axis=1)
    return wrap_structured_result(values, epochs, valid_chans)


@auto_gc
def zscore_stddev(epochs, chans=None):
    data = epochs.get_data()  # (n_epochs, n_channels, n_times)
    ch_names = [standardize_channel_name(ch) for ch in epochs.info["ch_names"]]

    if chans is None:
        chans = ch_names

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