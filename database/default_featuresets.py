from database.add_pipeline import add_pipeline
from database.add_fxdef import add_fxdefs
from database.add_featureset import add_featureset
from eeg2fx.function_registry import FEATURE_FUNCS
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(__file__), "eeg2go.db")

def get_latest_fxdef_ids(n):
    """Return last n fxdef IDs in ascending order (used immediately after batch insert)."""
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM fxdef ORDER BY id DESC LIMIT ?", (n,))
    rows = c.fetchall()
    conn.close()
    return [row[0] for row in reversed(rows)]

def create_all_features_featureset():
    """
    创建一个包含eeg2fx/feature目录下所有可用特征的特征集。
    这个特征集将包含所有在function_registry.py中注册的特征函数。
    """
    print("正在创建包含所有可用特征的特征集...")
    
    # 获取所有可用的特征函数
    available_features = list(FEATURE_FUNCS.keys())
    
    # 过滤掉 alpha_asymmetry 特征（暂时不测试）
    available_features = [f for f in available_features if f != "alpha_asymmetry"]
    
    print(f"发现 {len(available_features)} 个可用特征函数:")
    for i, feature in enumerate(available_features, 1):
        print(f"  {i:2d}. {feature}")
    
    # 常用通道配置
    common_channels = ["C3", "C4", "Pz", "O1", "O2"]
    
    # 获取可用的pipeline ID（使用第一个可用的pipeline）
    conn = sqlite3.connect(DB_PATH)
    c = conn.cursor()
    c.execute("SELECT id FROM pipedef ORDER BY id LIMIT 1")
    pipeline_id = c.fetchone()[0]
    conn.close()
    
    print(f"\n使用pipeline ID: {pipeline_id}")
    print(f"使用通道: {common_channels}")
    
    all_fxdef_ids = []
    
    # 为每个特征函数创建特征定义
    for feature_func in available_features:
        print(f"\n处理特征: {feature_func}")
        
        # 注释掉 alpha_asymmetry 的特殊处理
        # # 特殊处理alpha_asymmetry（需要两个通道）
        # if feature_func == "alpha_asymmetry":
        #     try:
        #         fxdef_ids = add_fxdefs({
        #             "func": feature_func,
        #             "pipeid": pipeline_id,
        #             "shortname": feature_func,
        #             "channels": ["C3", "C4"],  # alpha_asymmetry需要两个通道
        #             "params": {},
        #             "dim": "1d",
        #             "ver": "v1",
        #             "notes": f"{feature_func} for C3-C4"
        #         })
        #         all_fxdef_ids.extend(fxdef_ids)
        #         print(f"  ✓ 添加了 {len(fxdef_ids)} 个alpha_asymmetry特征定义")
        #     except Exception as e:
        #         print(f"  ✗ 添加alpha_asymmetry失败: {e}")
        # else:
        try:
            fxdef_ids = add_fxdefs({
                "func": feature_func,
                "pipeid": pipeline_id,
                "shortname": feature_func,
                "channels": common_channels,
                "params": {},
                "dim": "1d",
                "ver": "v1",
                "notes": f"{feature_func} for {{chan}}"
            })
            all_fxdef_ids.extend(fxdef_ids)
            print(f"  ✓ 添加了 {len(fxdef_ids)} 个{feature_func}特征定义")
        except Exception as e:
            print(f"  ✗ 添加{feature_func}失败: {e}")
    
    # 创建特征集
    try:
        set_id = add_featureset({
            "name": "all_available_features",
            "description": f"包含eeg2fx/feature目录下所有可用特征的特征集（除alpha_asymmetry），共{len(all_fxdef_ids)}个特征定义",
            "fxdef_ids": all_fxdef_ids
        })
        print(f"\n✓ 成功创建特征集 'all_available_features' (ID: {set_id})")
        print(f"  包含 {len(all_fxdef_ids)} 个特征定义")
        print(f"  注意：alpha_asymmetry 特征已暂时排除")
        return set_id
    except Exception as e:
        print(f"\n✗ 创建特征集失败: {e}")
        return None

def register_comparison_feature_sets():
    all_fxids = []

    # 1. 同 pipeline，多通道
    entropy_multi_ch = []
    for ch in ["C3", "C4", "Pz"]:
        entropy_multi_ch.append({
            "func": "spectral_entropy",
            "pipeid": 5,  # pipeline: entropy_eval_base
            "shortname": "entropy",
            "channels": [ch],
            "params": {},
            "dim": "1d",
            "ver": "v1",
            "notes": "Spectral entropy for {chan}"
        })

    print("\nCreating feature set: set_entropy_multi_ch")
    for fxdef in entropy_multi_ch:
        all_fxids += add_fxdefs(fxdef)

    add_featureset({
        "name": "Entropy across multiple channels",
        "description": "Same pipeline & function, C3/C4/Pz",
        "fxdef_ids": all_fxids
    })

    # 2. 同通道、同函数，不同 pipeline（差一个节点）
    print("\nAdding comparison pipelines...")

    base_steps = [
        ["flt", "filter", ["raw"], {"hp": 1.0, "lp": 35.0}],
        ["epoch", "epoch", ["flt"], {"duration": 2.0}]
    ]

    zscore_steps = [
        ["flt", "filter", ["raw"], {"hp": 1.0, "lp": 35.0}],
        ["notch", "notch_filter", ["flt"], {"freq": 50}],
        ["epoch", "epoch", ["notch"], {"duration": 2.0}]
    ]

    pipe1 = {
        "shortname": "entropy_base",
        "description": "Base pipeline without zscore",
        "source": "comparison",
        "chanset": "10/20",
        "fs": 250,
        "hp": 1.0,
        "lp": 35.0,
        "epoch": 2.0,
        "steps": base_steps
    }

    pipe2 = {
        "shortname": "entropy_notch",
        "description": "Adds notch filter before epoch",
        "source": "comparison",
        "chanset": "10/20",
        "fs": 250,
        "hp": 1.0,
        "lp": 35.0,
        "epoch": 2.0,
        "steps": zscore_steps
    }

    id1 = add_pipeline(pipe1)
    id2 = add_pipeline(pipe2)

    entropy_multi_pipe = [
        {
            "func": "spectral_entropy",
            "pipeid": id1,
            "shortname": "entropy_nz",
            "channels": ["C3"],
            "params": {},
            "dim": "1d",
            "ver": "v1",
            "notes": "Entropy C3 no zscore"
        },
        {
            "func": "spectral_entropy",
            "pipeid": id2,
            "shortname": "entropy_notch",
            "channels": ["C3"],
            "params": {},
            "dim": "1d",
            "ver": "v1",
            "notes": "Entropy C3 with notch"
        }
    ]

    all_fxids = []
    print("\nCreating feature set: set_entropy_multi_pipe")
    for fxdef in entropy_multi_pipe:
        all_fxids += add_fxdefs(fxdef)

    add_featureset({
        "name": "Entropy from different pipelines",
        "description": "Same function & channel, but pipeline differs (zscore)",
        "fxdef_ids": all_fxids
    })

if __name__ == "__main__":
    # 创建包含所有特征的特征集
    create_all_features_featureset()
    
    # 创建比较特征集（可选）
    register_comparison_feature_sets()
