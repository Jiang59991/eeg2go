from feature_mill.experiment_engine import run_experiment

result = run_experiment(
    experiment_type='correlation',
    dataset_id=1,
    feature_set_id=5,
    output_dir="data/experiments/correlation_all_available_new",
    extra_args={
        "target_vars": ["age"],  # 可选，默认已包含
        "method": "pearson",            # 可选，相关性方法
        "min_corr": 0.3,                # 可选，最小相关系数
        "top_n": 20                     # 可选，输出top N
    }
)
print(result)
