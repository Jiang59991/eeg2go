import numpy as np
import gc
import functools
import inspect
import traceback
from logging_config import logger

def auto_gc(fn):
    """
    Decorator for EEG feature functions to automatically trigger garbage collection
    after execution and optionally clean up large intermediate variables.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            result = fn(*args, **kwargs)
        finally:
            gc.collect()
        return result
    return wrapper

def wrap_structured_result(values, epochs, chans):
    """
    Wrap a multi-channel (n_epochs, n_chans) result matrix into structured format per channel.
    Output: Dict[channel_name, List[{epoch, start, end, value}]]
    """
    sfreq = epochs.info["sfreq"]
    tmin = epochs.tmin
    duration = epochs.times[-1] - epochs.times[0]
    n_epochs, n_chans = values.shape

    structured_by_channel = {}

    for ch_idx, ch_name in enumerate(chans):
        structured = []
        for i in range(n_epochs):
            start = tmin + i * (duration + 1.0 / sfreq)
            end = start + duration
            val = values[i, ch_idx]
            structured.append({
                "epoch": i,
                "start": float(start),
                "end": float(end),
                "value": float(val) if np.isscalar(val) else val
            })
        structured_by_channel[ch_name] = structured

    return structured_by_channel
