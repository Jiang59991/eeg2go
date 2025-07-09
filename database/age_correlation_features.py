#!/usr/bin/env python3
"""
Age correlation features for EEG2Go

This module defines EEG features that are known to have strong correlations with age.
Based on research literature, these features include:

1. Alpha peak frequency - decreases with age
2. Alpha power - decreases with age  
3. Theta/Alpha ratio - increases with age
4. Beta power - decreases with age
5. Spectral edge frequency - decreases with age
6. Spectral entropy - changes with age
7. Alpha asymmetry - changes with age

These features are designed to test the correlation analysis functionality.
"""

from database.add_fxdef import add_fxdefs
from logging_config import logger

def create_age_correlation_features():
    """Create feature definitions for age correlation testing"""
    
    features = []
    
    # 1. Alpha peak frequency - one of the strongest age-related features
    features.append({
        "func": "alpha_peak_frequency",
        "pipeid": 5,  # Using the entropy pipeline
        "shortname": "alpha_peak_freq",
        "channels": ["C3", "C4", "Pz", "O1", "O2"],
        "params": {"fmin": 7, "fmax": 13},
        "dim": "scalar",
        "ver": "v1",
        "notes": "Alpha peak frequency - decreases with age"
    })
    
    # 2. Alpha power - decreases with age
    features.append({
        "func": "bandpower",
        "pipeid": 5,
        "shortname": "alpha_power",
        "channels": ["C3", "C4", "Pz", "O1", "O2"],
        "params": {"band": "alpha"},
        "dim": "1d",
        "ver": "v1", 
        "notes": "Alpha band power - decreases with age"
    })
    
    # 3. Theta/Alpha ratio - increases with age
    features.append({
        "func": "theta_alpha_ratio",
        "pipeid": 5,
        "shortname": "theta_alpha_ratio",
        "channels": ["C3", "C4", "Pz"],
        "params": {},
        "dim": "scalar",
        "ver": "v1",
        "notes": "Theta/Alpha power ratio - increases with age"
    })
    
    # 4. Beta power - decreases with age
    features.append({
        "func": "bandpower",
        "pipeid": 5,
        "shortname": "beta_power",
        "channels": ["C3", "C4", "F3", "F4"],
        "params": {"band": "beta"},
        "dim": "1d",
        "ver": "v1",
        "notes": "Beta band power - decreases with age"
    })
    
    # 5. Spectral edge frequency - decreases with age
    features.append({
        "func": "spectral_edge_frequency",
        "pipeid": 5,
        "shortname": "spectral_edge",
        "channels": ["C3", "C4", "Pz"],
        "params": {"percentile": 95},
        "dim": "scalar",
        "ver": "v1",
        "notes": "95th percentile spectral edge frequency - decreases with age"
    })
    
    # 6. Spectral entropy - changes with age
    features.append({
        "func": "spectral_entropy",
        "pipeid": 5,
        "shortname": "spectral_entropy",
        "channels": ["C3", "C4", "Pz"],
        "params": {},
        "dim": "1d",
        "ver": "v1",
        "notes": "Spectral entropy - changes with age"
    })
    
    # 7. Alpha asymmetry (C4-C3) - changes with age (使用连字符格式)
    features.append({
        "func": "alpha_asymmetry",
        "pipeid": 5,
        "shortname": "alpha_asymmetry",
        "channels": ["C3-C4"],  # 修改：使用连字符格式
        "params": {},
        "dim": "scalar",
        "ver": "v1",
        "notes": "Alpha power asymmetry (C4-C3) - changes with age"
    })
    
    # 8. Delta power - increases with age
    features.append({
        "func": "bandpower",
        "pipeid": 5,
        "shortname": "delta_power",
        "channels": ["C3", "C4", "F3", "F4"],
        "params": {"band": "delta"},
        "dim": "1d",
        "ver": "v1",
        "notes": "Delta band power - increases with age"
    })
    
    # 9. Gamma power - decreases with age
    features.append({
        "func": "bandpower",
        "pipeid": 5,
        "shortname": "gamma_power",
        "channels": ["C3", "C4", "F3", "F4"],
        "params": {"band": "gamma"},
        "dim": "1d",
        "ver": "v1",
        "notes": "Gamma band power - decreases with age"
    })
    
    # 10. Relative alpha power - decreases with age
    features.append({
        "func": "relative_power",
        "pipeid": 5,
        "shortname": "relative_alpha_power",
        "channels": ["C3", "C4", "Pz", "O1", "O2"],
        "params": {"band": "alpha"},
        "dim": "1d",
        "ver": "v1",
        "notes": "Relative alpha power - decreases with age"
    })
    
    # 11. Coherence features (双通道特征 - 使用连字符格式)
    features.append({
        "func": "coherence_band",
        "pipeid": 5,
        "shortname": "alpha_coherence",
        "channels": ["C3-C4", "F3-F4"],  # 修改：使用连字符格式
        "params": {"band": (8, 13)},
        "dim": "scalar",
        "ver": "v1",
        "notes": "Alpha band coherence between channels"
    })
    
    # 12. PLV features (双通道特征 - 使用连字符格式)
    features.append({
        "func": "plv",
        "pipeid": 5,
        "shortname": "alpha_plv",
        "channels": ["C3-C4", "F3-F4"],  # 修改：使用连字符格式
        "params": {},
        "dim": "scalar",
        "ver": "v1",
        "notes": "Phase-locking value between channels"
    })
    
    return features

def add_age_correlation_features():
    """Add age correlation features to the database"""
    features = create_age_correlation_features()
    logger.info("Adding age correlation features to database...")
    for i, fxdef_spec in enumerate(features, 1):
        logger.info(f"{i}. Adding feature: {fxdef_spec['shortname']} ({fxdef_spec['func']}, {fxdef_spec['channels']})")
        try:
            add_fxdefs(fxdef_spec)
            logger.info("   ✓ Successfully added")
        except Exception as e:
            logger.error(f"   ✗ Failed to add: {e}")
    logger.info(f"Completed adding {len(features)} age correlation features.")

if __name__ == "__main__":
    add_age_correlation_features() 