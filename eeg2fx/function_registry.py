from eeg2fx.steps import load_recording, filter, reref, zscore, epoch, reref, notch_filter, resample, ica
from eeg2fx.feature import (
    feature_time,
    feature_freq,
    feature_nonlinear,
    feature_connect,
    feature_tf,
    feature_stat
)

def split_channel(result_dict, chan):
    if isinstance(result_dict, dict) and chan in result_dict:
        return result_dict[chan]
    return []

PREPROCESSING_FUNCS = {
    "raw": load_recording,
    "filter": filter,
    "notch_filter": notch_filter,
    "reref": reref,
    "resample": resample,
    "ica": ica,
    "zscore": zscore,
    "epoch": epoch,
}

FEATURE_FUNCS = {
    # time features
    "mean_amplitude": feature_time.mean_amplitude,
    "rms": feature_time.rms,
    "zero_crossings": feature_time.zero_crossings,

    # freq features
    "bandpower": feature_freq.bandpower,
    "relative_power": feature_freq.relative_power,
    "spectral_entropy": feature_freq.spectral_entropy,
    "alpha_peak_frequency": feature_freq.alpha_peak_frequency,
    "theta_alpha_ratio": feature_freq.theta_alpha_ratio,
    "spectral_edge_frequency": feature_freq.spectral_edge_frequency,
    "alpha_asymmetry": feature_freq.alpha_asymmetry,

    # nonlinear features
    "permutation_entropy": feature_nonlinear.permutation_entropy,
    "approximate_entropy": feature_nonlinear.approximate_entropy,
    "signal_skewness": feature_nonlinear.signal_skewness,
    "signal_kurtosis": feature_nonlinear.signal_kurtosis,

    # tf features
    "wavelet_entropy": feature_tf.wavelet_entropy,
    "stft_power": feature_tf.stft_power,

    # connect features
    "coherence_band": feature_connect.coherence_band,
    "plv": feature_connect.plv,

    # stat features
    "mean": feature_stat.mean,
    "zscore_stddev": feature_stat.zscore_stddev,
}

UTILITY_FUNCS = {
    "split_channel": split_channel,
}
