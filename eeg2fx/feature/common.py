import numpy as np
import gc
import functools
import inspect
import traceback
from logging_config import logger
from typing import Callable, Any, Dict, List

def auto_gc(fn: Callable) -> Callable:
    """
    Decorator for EEG feature functions.
    Automatically triggers garbage collection after the function execution.
    
    Args:
        fn (Callable): The function to be decorated.
    
    Returns:
        Callable: The wrapped function with automatic garbage collection.
    """
    @functools.wraps(fn)
    def wrapper(*args, **kwargs):
        try:
            result = fn(*args, **kwargs)
        finally:
            gc.collect()
        return result
    return wrapper

def wrap_structured_result(
    values: np.ndarray, 
    epochs: Any, 
    chans: List[str]
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Wrap a multi-channel (n_epochs, n_chans) result matrix into a structured format per channel.
    
    Args:
        values (np.ndarray): The result matrix of shape (n_epochs, n_chans).
        epochs (Any): The epochs object containing info, tmin, and times.
        chans (List[str]): List of channel names.
    
    Returns:
        Dict[str, List[Dict[str, Any]]]: 
            Dictionary mapping channel names to a list of dictionaries, 
            each containing epoch index, start time, end time, and value.
    """
    sfreq = epochs.info["sfreq"]
    tmin = epochs.tmin
    duration = epochs.times[-1] - epochs.times[0]
    n_epochs, n_chans = values.shape

    structured_by_channel: Dict[str, List[Dict[str, Any]]] = {}

    for ch_idx, ch_name in enumerate(chans):
        structured = []
        for i in range(n_epochs):
            # Calculate start and end time for each epoch
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
