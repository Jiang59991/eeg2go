from database.add_pipeline import add_pipeline
from logging_config import logger

def create_default_pipelines():
    pipelines = [

        # 1. 标准 EEG 预处理流程：滤波 + 参考 + 分段
        {
            "shortname": "classic_clean_5s",
            "description": "Standard EEG preprocessing: bandpass filter → reref → epoch",
            "source": "default",
            "chanset": "10/20",
            "fs": 250,
            "hp": 1.0,
            "lp": 40.0,
            "epoch": 5.0,
            "sample_rating": 8.0,
            "steps": [
                ["flt", "filter", ["raw"], {"hp": 1.0, "lp": 40.0}],
                ["reref", "reref", ["flt"], {"method": "average"}],
                ["epoch", "epoch", ["reref"], {"duration": 5.0}]
            ]
        },

        # 2. 功能覆盖验证：包含所有 preprocessing steps（用于代码测试）
        {
            "shortname": "full_preprocessing_test",
            "description": "Covers all core preprocessing steps for functional verification",
            "source": "test_suite",
            "chanset": "10/20",
            "fs": 256,
            "hp": 1.0,
            "lp": 30.0,
            "epoch": 4.0,
            "sample_rating": 9.0,
            "steps": [
                ["flt", "filter", ["raw"], {"hp": 1.0, "lp": 30.0}],
                ["notch", "notch_filter", ["flt"], {"freq": 50.0}],
                ["resample", "resample", ["notch"], {"sfreq": 128.0}],
                ["reref", "reref", ["resample"], {"method": "average"}],
                ["ica", "ica", ["reref"], {"n_components": 20, "detect_artifacts": "none"}],
                ["epoch", "epoch", ["ica"], {"duration": 4.0}],
                ["rej", "reject_high_amplitude", ["epoch"], {"threshold_uv": 150}],
                ["z", "zscore", ["rej"], {"mode": "per_epoch"}]
            ]
        },

        # 3. 最小滤波路径（用于实验对照）
        {
            "shortname": "filter_only_epoch",
            "description": "Minimal preprocessing: bandpass filter then epoch",
            "source": "default",
            "chanset": "10/20",
            "fs": 200,
            "hp": 1.0,
            "lp": 30.0,
            "epoch": 3.0,
            "sample_rating": 6.0,
            "steps": [
                ["flt", "filter", ["raw"], {"hp": 1.0, "lp": 30.0}],
                ["epoch", "epoch", ["flt"], {"duration": 3.0}]
            ]
        },

        # 4. 最小参考路径（用于实验对照）
        {
            "shortname": "reref_only_epoch",
            "description": "Minimal preprocessing: rereferencing then epoch",
            "source": "default",
            "chanset": "10/20",
            "fs": 200,
            "hp": 1.0,
            "lp": 30.0,
            "epoch": 3.0,
            "sample_rating": 5.0,
            "steps": [
                ["reref", "reref", ["raw"], {"method": "average"}],
                ["epoch", "epoch", ["reref"], {"duration": 3.0}]
            ]
        },

        # 5. 用于 entropy 提取的最小 pipeline（仅至 epoch）
        {
            "shortname": "entropy_eval_base",
            "description": "Minimal pipeline ending with epochs (used for entropy fxdef)",
            "source": "test",
            "chanset": "minimal",
            "fs": 250,
            "hp": 1.0,
            "lp": 35.0,
            "epoch": 2.0,
            "sample_rating": 7.0,
            "steps": [
                ["flt", "filter", ["raw"], {"hp": 1.0, "lp": 35.0}],
                ["epoch", "epoch", ["flt"], {"duration": 2.0}]
            ]
        }
    ]

    for p in pipelines:
        logger.info(f"Adding pipeline: {p['shortname']}")
        add_pipeline(p)

if __name__ == "__main__":
    create_default_pipelines()
