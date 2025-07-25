from eeg2fx.steps import load_recording, filter, reref, zscore, epoch, reref, notch_filter, resample, ica
from eeg2fx.feature import (
    feature_time,
    feature_freq,
    feature_nonlinear,
    feature_connect,
    feature_tf,
    feature_stat
)
from logging_config import logger

def split_channel(result_dict, chan):
    if isinstance(result_dict, dict) and chan in result_dict:
        return result_dict[chan]
    return []

# Feature metadata definition
FEATURE_METADATA = {
    # Single-channel features
    "mean_amplitude": {"type": "single_channel", "description": "Mean amplitude per epoch"},
    "rms": {"type": "single_channel", "description": "Root mean square per epoch"},
    "zero_crossings": {"type": "single_channel", "description": "Zero crossing count per epoch"},
    "bandpower": {"type": "single_channel", "description": "Band power per epoch"},
    "relative_power": {"type": "single_channel", "description": "Relative power per epoch"},
    "spectral_entropy": {"type": "single_channel", "description": "Spectral entropy per epoch"},
    "alpha_peak_frequency": {"type": "single_channel", "description": "Alpha peak frequency per epoch"},
    "theta_alpha_ratio": {"type": "single_channel", "description": "Theta/alpha ratio per epoch"},
    "spectral_edge_frequency": {"type": "single_channel", "description": "Spectral edge frequency per epoch"},
    "permutation_entropy": {"type": "single_channel", "description": "Permutation entropy per epoch"},
    "approximate_entropy": {"type": "single_channel", "description": "Approximate entropy per epoch"},
    "signal_skewness": {"type": "single_channel", "description": "Signal skewness per epoch"},
    "signal_kurtosis": {"type": "single_channel", "description": "Signal kurtosis per epoch"},
    "wavelet_entropy": {"type": "single_channel", "description": "Wavelet entropy per epoch"},
    "stft_power": {"type": "single_channel", "description": "STFT power per epoch"},
    "mean": {"type": "single_channel", "description": "Mean value per epoch"},
    "std": {"type": "single_channel", "description": "Standard deviation per epoch"},
    "skew": {"type": "single_channel", "description": "Skewness per epoch"},
    "kurt": {"type": "single_channel", "description": "Kurtosis per epoch"},
    
    # 新增高级时域特征
    "signal_variance": {"type": "single_channel", "description": "Signal variance per epoch"},
    "peak_to_peak_amplitude": {"type": "single_channel", "description": "Peak-to-peak amplitude per epoch"},
    "crest_factor": {"type": "single_channel", "description": "Crest factor per epoch"},
    "shape_factor": {"type": "single_channel", "description": "Shape factor per epoch"},
    "impulse_factor": {"type": "single_channel", "description": "Impulse factor per epoch"},
    "margin_factor": {"type": "single_channel", "description": "Margin factor per epoch"},
    "signal_entropy": {"type": "single_channel", "description": "Signal entropy per epoch"},
    "signal_complexity": {"type": "single_channel", "description": "Signal complexity per epoch"},
    "signal_regularity": {"type": "single_channel", "description": "Signal regularity per epoch"},
    "signal_stability": {"type": "single_channel", "description": "Signal stability per epoch"},
    
    # 新增高级频域特征
    "spectral_centroid": {"type": "single_channel", "description": "Spectral centroid per epoch"},
    "spectral_bandwidth": {"type": "single_channel", "description": "Spectral bandwidth per epoch"},
    "spectral_rolloff": {"type": "single_channel", "description": "Spectral rolloff frequency per epoch"},
    "spectral_flatness": {"type": "single_channel", "description": "Spectral flatness per epoch"},
    "spectral_skewness": {"type": "single_channel", "description": "Spectral skewness per epoch"},
    "spectral_kurtosis": {"type": "single_channel", "description": "Spectral kurtosis per epoch"},
    "band_energy_ratio": {"type": "single_channel", "description": "Band energy ratio per epoch"},
    "spectral_complexity": {"type": "single_channel", "description": "Spectral complexity per epoch"},
    "frequency_modulation_index": {"type": "single_channel", "description": "Frequency modulation index per epoch"},
    
    # Channel-pair features
    "coherence_band": {"type": "channel_pair", "description": "Coherence between two channels", "default_pairs": [["C3", "C4"], ["F3", "F4"], ["P3", "P4"]]},
    "plv": {"type": "channel_pair", "description": "Phase-locking value between two channels", "default_pairs": [["C3", "C4"], ["F3", "F4"], ["P3", "P4"]]},
    "alpha_asymmetry": {"type": "channel_pair", "description": "Alpha power asymmetry between two channels", "default_pairs": [["C3", "C4"]]},
    
    # 新增高级连接性特征
    "mutual_information": {"type": "channel_pair", "description": "Mutual information between two channels", "default_pairs": [["C3", "C4"], ["F3", "F4"], ["P3", "P4"]]},
    "cross_correlation": {"type": "channel_pair", "description": "Cross correlation between two channels", "default_pairs": [["C3", "C4"], ["F3", "F4"], ["P3", "P4"]]},
    "phase_synchronization": {"type": "channel_pair", "description": "Phase synchronization between two channels", "default_pairs": [["C3", "C4"], ["F3", "F4"], ["P3", "P4"]]},
    "amplitude_correlation": {"type": "channel_pair", "description": "Amplitude correlation between two channels", "default_pairs": [["C3", "C4"], ["F3", "F4"], ["P3", "P4"]]},
    "granger_causality": {"type": "channel_pair", "description": "Granger causality between two channels", "default_pairs": [["C3", "C4"], ["F3", "F4"], ["P3", "P4"]]},
    "directed_transfer_function": {"type": "channel_pair", "description": "Directed transfer function between two channels", "default_pairs": [["C3", "C4"], ["F3", "F4"], ["P3", "P4"]]},
    "synchronization_likelihood": {"type": "channel_pair", "description": "Synchronization likelihood between two channels", "default_pairs": [["C3", "C4"], ["F3", "F4"], ["P3", "P4"]]},
    
    # Scalar features
    "zscore_stddev": {"type": "scalar", "description": "Z-score standard deviation (scalar value)"},
}

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
    
    # 新增高级时域特征
    "signal_variance": feature_time.signal_variance,
    "signal_skewness": feature_time.signal_skewness,
    "signal_kurtosis": feature_time.signal_kurtosis,
    "peak_to_peak_amplitude": feature_time.peak_to_peak_amplitude,
    "crest_factor": feature_time.crest_factor,
    "shape_factor": feature_time.shape_factor,
    "impulse_factor": feature_time.impulse_factor,
    "margin_factor": feature_time.margin_factor,
    "signal_entropy": feature_time.signal_entropy,
    "signal_complexity": feature_time.signal_complexity,
    "signal_regularity": feature_time.signal_regularity,
    "signal_stability": feature_time.signal_stability,

    # freq features
    "bandpower": feature_freq.bandpower,
    "relative_power": feature_freq.relative_power,
    "spectral_entropy": feature_freq.spectral_entropy,
    "alpha_peak_frequency": feature_freq.alpha_peak_frequency,
    "theta_alpha_ratio": feature_freq.theta_alpha_ratio,
    "spectral_edge_frequency": feature_freq.spectral_edge_frequency,
    "alpha_asymmetry": feature_freq.alpha_asymmetry,
    
    # 新增高级频域特征
    "spectral_centroid": feature_freq.spectral_centroid,
    "spectral_bandwidth": feature_freq.spectral_bandwidth,
    "spectral_rolloff": feature_freq.spectral_rolloff,
    "spectral_flatness": feature_freq.spectral_flatness,
    "spectral_skewness": feature_freq.spectral_skewness,
    "spectral_kurtosis": feature_freq.spectral_kurtosis,
    "band_energy_ratio": feature_freq.band_energy_ratio,
    "spectral_complexity": feature_freq.spectral_complexity,
    "frequency_modulation_index": feature_freq.frequency_modulation_index,

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
    
    # 新增高级连接性特征
    "mutual_information": feature_connect.mutual_information,
    "cross_correlation": feature_connect.cross_correlation,
    "phase_synchronization": feature_connect.phase_synchronization,
    "amplitude_correlation": feature_connect.amplitude_correlation,
    "granger_causality": feature_connect.granger_causality,
    "directed_transfer_function": feature_connect.directed_transfer_function,
    "synchronization_likelihood": feature_connect.synchronization_likelihood,

    # stat features
    "mean": feature_stat.mean,
    "std": feature_stat.std,
    "skew": feature_stat.skew,
    "kurt": feature_stat.kurt,
    "zscore_stddev": feature_stat.zscore_stddev,
}

UTILITY_FUNCS = {
    "split_channel": split_channel,
}

def resolve_function(func_name):
    """解析函数名称到实际函数"""
    if func_name in PREPROCESSING_FUNCS:
        return PREPROCESSING_FUNCS[func_name]
    if func_name in FEATURE_FUNCS:
        return FEATURE_FUNCS[func_name]
    if func_name in UTILITY_FUNCS:
        return UTILITY_FUNCS[func_name]
    raise ValueError(f"Function '{func_name}' is not registered in function_registry.")
