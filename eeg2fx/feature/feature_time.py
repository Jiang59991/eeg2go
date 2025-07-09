# EEG feature module: Time-domain features

import numpy as np
import gc
from eeg2fx.feature.common import standardize_channel_name, wrap_structured_result, auto_gc


@auto_gc
def mean_amplitude(epochs, chans=None):
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
    std_ch_names = [standardize_channel_name(ch) for ch in raw_ch_names]
    n_epochs = data.shape[0]

    if chans is None:
        chans = std_ch_names

    values = []
    valid_chans = []

    for ch in chans:
        if ch in std_ch_names:
            idx = std_ch_names.index(ch)
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
    std_ch_names = [standardize_channel_name(ch) for ch in raw_ch_names]
    n_epochs = data.shape[0]

    if chans is None:
        chans = std_ch_names

    def count_zero_crossings(signal):
        return np.sum(np.diff(np.signbit(signal)))

    values = []
    valid_chans = []

    for ch in chans:
        if ch in std_ch_names:
            idx = std_ch_names.index(ch)
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
