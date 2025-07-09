import os
import sqlite3
import subprocess
from database.add_fxdef import add_fxdefs
from database.add_featureset import add_featureset

DB_PATH = os.path.join(os.path.dirname(__file__), "database", "eeg2go.db")

def register_age_correlation_features():
    print("register age correlation features...")
    # 常用通道
    channels = ["C3", "C4", "Pz", "O1", "O2"]
    fxids = []

    # 1. 频谱熵 (单通道特征)
    fxids += add_fxdefs({
        "func": "spectral_entropy",
        "pipeid": 1,  # 假设1号pipeline是标准预处理
        "shortname": "entropy",
        "channels": channels,
        "params": {},
        "dim": "1d",
        "ver": "v1",
        "notes": "Spectral entropy for {chan}"
    })

    # 2. alpha波段功率 (单通道特征)
    fxids += add_fxdefs({
        "func": "bandpower",
        "pipeid": 1,
        "shortname": "bp_alpha",
        "channels": channels,
        "params": {"band": "alpha"},
        "dim": "1d",
        "ver": "v1",
        "notes": "Alpha band power for {chan}"
    })

    # 3. alpha peak frequency (单通道特征)
    fxids += add_fxdefs({
        "func": "alpha_peak_frequency",
        "pipeid": 1,
        "shortname": "alpha_peak",
        "channels": channels,
        "params": {},
        "dim": "1d",
        "ver": "v1",
        "notes": "Alpha peak frequency for {chan}"
    })

    # 4. alpha asymmetry (通道对特征 - 使用连字符格式)
    fxids += add_fxdefs({
        "func": "alpha_asymmetry",
        "pipeid": 1,
        "shortname": "alpha_asym",
        "channels": ["C3-C4"],  # 修改：使用连字符格式
        "params": {},
        "dim": "scalar",
        "ver": "v1",
        "notes": "Alpha power asymmetry (C4-C3) - changes with age"
    })

    # 5. theta/alpha ratio (单通道特征)
    fxids += add_fxdefs({
        "func": "theta_alpha_ratio",
        "pipeid": 1,
        "shortname": "theta_alpha_ratio",
        "channels": ["C3", "C4", "Pz"],
        "params": {},
        "dim": "scalar",
        "ver": "v1",
        "notes": "Theta/Alpha power ratio - increases with age"
    })

    # 6. spectral edge frequency (单通道特征)
    fxids += add_fxdefs({
        "func": "spectral_edge_frequency",
        "pipeid": 1,
        "shortname": "spectral_edge",
        "channels": ["C3", "C4", "Pz"],
        "params": {"percentile": 95},
        "dim": "scalar",
        "ver": "v1",
        "notes": "95th percentile spectral edge frequency - decreases with age"
    })

    # 7. beta power (单通道特征)
    fxids += add_fxdefs({
        "func": "bandpower",
        "pipeid": 1,
        "shortname": "bp_beta",
        "channels": ["C3", "C4", "F3", "F4"],
        "params": {"band": "beta"},
        "dim": "1d",
        "ver": "v1",
        "notes": "Beta band power - decreases with age"
    })

    # 8. delta power (单通道特征)
    fxids += add_fxdefs({
        "func": "bandpower",
        "pipeid": 1,
        "shortname": "bp_delta",
        "channels": ["C3", "C4", "F3", "F4"],
        "params": {"band": "delta"},
        "dim": "1d",
        "ver": "v1",
        "notes": "Delta band power - increases with age"
    })

    # 9. gamma power (单通道特征)
    fxids += add_fxdefs({
        "func": "bandpower",
        "pipeid": 1,
        "shortname": "bp_gamma",
        "channels": ["C3", "C4", "F3", "F4"],
        "params": {"band": "gamma"},
        "dim": "1d",
        "ver": "v1",
        "notes": "Gamma band power - decreases with age"
    })

    # 10. relative alpha power (单通道特征)
    fxids += add_fxdefs({
        "func": "relative_power",
        "pipeid": 1,
        "shortname": "rel_alpha_power",
        "channels": ["C3", "C4", "Pz", "O1", "O2"],
        "params": {"band": "alpha"},
        "dim": "1d",
        "ver": "v1",
        "notes": "Relative alpha power - decreases with age"
    })

    # 注册特征集
    set_id = add_featureset({
        "name": "age_correlation_features",
        "description": "Features for age correlation analysis - updated for channel_pair support",
        "fxdef_ids": fxids
    })
    print(f"feature set id: {set_id}")
    print(f"Total features added: {len(fxids)}")

def main():
    register_age_correlation_features()
    print("database setup completed!")

if __name__ == "__main__":
    main()