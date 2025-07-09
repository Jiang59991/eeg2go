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

    # 1. 频谱熵
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

    # 2. alpha波段功率
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

    # 3. alpha peak frequency
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

    # 4. alpha asymmetry (只用C3/C4)
    fxids += add_fxdefs({
        "func": "alpha_asymmetry",
        "pipeid": 1,
        "shortname": "alpha_asym",
        "channels": ["C3", "C4"],
        "params": {},
        "dim": "1d",
        "ver": "v1",
        "notes": "Alpha asymmetry for {chan}"
    })

    # 注册特征集
    set_id = add_featureset({
        "name": "age_correlation_features",
        "description": "Features for age correlation analysis",
        "fxdef_ids": fxids
    })
    print(f"feature set id: {set_id}")

def main():
    register_age_correlation_features()
    print("database setup completed!")

if __name__ == "__main__":
    main()