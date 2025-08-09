#!/usr/bin/env python3
import os
import json
import argparse
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline as SkPipeline
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, balanced_accuracy_score
from scipy import stats

from logging_config import logger
from feature_mill.experiment_engine import (
    extract_feature_matrix_direct,
    get_relevant_metadata,
    DEFAULT_DB_PATH,
)

# 实验常量
REQUIRED_CHANNELS = ["F3", "F4", "C3", "C4", "O1", "O2"]
N_SPLITS = 5


def infer_binary_label_from_meta(df_meta: pd.DataFrame) -> pd.Series:
    """
    将 recording_metadata 中的 normal / abnormal / status 字段解析为二分类标签：
    - y=1: abnormal
    - y=0: normal
    优先级：abnormal > normal > status（包含 'abnormal'/'normal' 字样）
    解析失败的行返回 NaN（后续会被丢弃）。
    """
    y = pd.Series(index=df_meta.index, dtype=float)

    def truthy(v) -> bool:
        if v is None:
            return False
        if isinstance(v, (int, float)):
            return v != 0
        s = str(v).strip().lower()
        return s in ("1", "y", "yes", "true", "t") or s == "abnormal" or s == "normal"

    # 先 abnormal
    if "abnormal" in df_meta.columns:
        mask = df_meta["abnormal"].apply(truthy)
        y.loc[mask] = 1.0

    # 再 normal（不覆盖已确定标签）
    if "normal" in df_meta.columns:
        mask = df_meta["normal"].apply(truthy)
        y.loc[mask & y.isna()] = 0.0

    # 最后 status（字符串包含 abnormal/normal）
    if "status" in df_meta.columns:
        s = df_meta["status"].astype(str).str.lower()
        y.loc[s.str.contains("abnormal", na=False) & y.isna()] = 1.0
        y.loc[s.str.contains("normal", na=False) & y.isna()] = 0.0

    return y


def select_feature_columns(df_feat: pd.DataFrame) -> List[str]:
    """
    只选 bandpower relative（bp_rel）+ 目标通道 + 录波级聚合的中位数：*_median
    """
    cols = []
    for col in df_feat.columns:
        if not col.endswith("_median"):
            continue
        # 仅限相对功率（短名包含 bp_rel_）
        if "_bp_rel_" not in col:
            continue
        # 必须包含目标通道（末尾下划线前的 token 通常是通道名）
        # 例：fx123_bp_rel_alpha_C3_median
        parts = col.split("_")
        if len(parts) < 3:
            continue
        # 从倒数第二个 token 抽出通道名（median 前一个）
        ch = parts[-2].upper()
        if ch in [c.upper() for c in REQUIRED_CHANNELS]:
            cols.append(col)
    return sorted(cols)


def build_cv_splits(df_all: pd.DataFrame, n_splits: int = N_SPLITS) -> List[Tuple[np.ndarray, np.ndarray]]:
    """
    基于 subject_id 做 GroupKFold，保证同 subject 不跨折。
    为“公平性”，固定排序（按 subject_id, recording_id）后的顺序来生成分割。
    """
    df_sorted = df_all.sort_values(["subject_id", "recording_id"]).reset_index(drop=True)
    groups = df_sorted["subject_id"].astype(str).values
    gkf = GroupKFold(n_splits=n_splits)
    splits = []
    for train_idx, test_idx in gkf.split(df_sorted.index.values, groups=groups, groups=groups):
        splits.append((train_idx, test_idx))
    return splits, df_sorted


def train_eval_one_pipeline(
    df_feat: pd.DataFrame,
    df_meta: pd.DataFrame,
    splits: List[Tuple[np.ndarray, np.ndarray]],
    df_ordered: pd.DataFrame,
    model_name: str = "logreg",
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    在给定的固定 splits（同折）下，训练并评估单条 pipeline。
    返回：
      - 每折指标 dataframe
      - 各指标 mean/std 汇总 dict（键名如 auc_mean, auc_std ...）
    """
    # 对齐到固定顺序（与 splits 一一对应）
    df_feat_idxed = df_feat.set_index("recording_id")
    X_rows = []
    for rid in df_ordered["recording_id"].tolist():
        if rid not in df_feat_idxed.index:
            # 若该 pipeline 缺此录波，则无法公平比较（通常已在上游做了交集）
            raise ValueError(f"Missing recording_id={rid} in feature matrix.")
        X_rows.append(df_feat_idxed.loc[rid])
    df_X = pd.DataFrame(X_rows).reset_index()

    # 选择特征列
    feat_cols = select_feature_columns(df_X)
    if not feat_cols:
        raise ValueError("No valid bp_rel *_median feature columns found.")
    X_all = df_X[feat_cols].values

    # 标签（与 df_ordered 对齐）
    df_meta_idxed = df_meta.set_index("recording_id").loc[df_ordered["recording_id"].values]
    y_all = df_meta_idxed["label"].values.astype(int)

    # 模型
    if model_name == "logreg":
        estimator = LogisticRegression(
            penalty="l2",
            C=1.0,
            class_weight="balanced",
            max_iter=2000,
            n_jobs=None,
            solver="liblinear",
        )
    elif model_name == "linsvm":
        estimator = LinearSVC(C=1.0, class_weight="balanced", max_iter=20000)
    else:
        raise ValueError(f"Unknown model_name: {model_name}")

    clf = SkPipeline(steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler(with_mean=True, with_std=True)),
        ("est", estimator),
    ])

    rows = []
    for fold_id, (tr, te) in enumerate(splits, start=1):
        X_tr, X_te = X_all[tr], X_all[te]
        y_tr, y_te = y_all[tr], y_all[te]

        clf.fit(X_tr, y_tr)

        # 预测与指标
        if model_name == "logreg":
            proba = clf.predict_proba(X_te)[:, 1]
            y_pred = (proba >= 0.5).astype(int)
            auc = roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else np.nan
        else:
            # LinearSVC 无 predict_proba，用 decision_function 近似 AUC
            dec = clf.decision_function(X_te)
            y_pred = (dec >= 0.0).astype(int)
            auc = roc_auc_score(y_te, dec) if len(np.unique(y_te)) > 1 else np.nan

        acc = accuracy_score(y_te, y_pred)
        f1 = f1_score(y_te, y_pred, zero_division=0)
        bacc = balanced_accuracy_score(y_te, y_pred)

        rows.append({
            "fold": fold_id,
            "auc": auc,
            "accuracy": acc,
            "f1": f1,
            "balanced_accuracy": bacc,
        })

    df_fold = pd.DataFrame(rows)

    summary = {}
    for metric in ["auc", "accuracy", "f1", "balanced_accuracy"]:
        summary[f"{metric}_mean"] = float(np.nanmean(df_fold[metric].values))
        summary[f"{metric}_std"] = float(np.nanstd(df_fold[metric].values, ddof=1))

    return df_fold, summary


def paired_tests_by_auc(results: Dict[str, pd.DataFrame], k: int = N_SPLITS) -> pd.DataFrame:
    """
    对每对 pipeline 的按折 AUC 进行配对比较：
      - 常规配对 t 检验
      - Nadeau–Bengio 校正的重复K折 t 检验（近似校正因子 1/k + test/train ≈ 1/k + 1/(k-1)）
    返回比较表 DataFrame。
    """
    names = list(results.keys())
    records = []

    def nb_correction(var, k):
        # test/train ≈ 1/(k-1)
        return (1.0 / k) + (1.0 / (k - 1))

    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            a = names[i]
            b = names[j]
            auc_a = results[a]["auc"].values
            auc_b = results[b]["auc"].values
            if len(auc_a) != len(auc_b):
                raise ValueError("AUC folds length mismatch.")
            d = auc_a - auc_b

            # 常规配对 t
            t_stat, p_val = stats.ttest_rel(auc_a, auc_b, nan_policy="omit")

            # Nadeau–Bengio 校正
            d_clean = d[~np.isnan(d)]
            n = len(d_clean)
            if n > 1:
                mean_d = np.mean(d_clean)
                var_d = np.var(d_clean, ddof=1)
                c = nb_correction(var_d, k)
                # 校正后的标准误
                se = np.sqrt(var_d * c)
                t_nb = mean_d / se if se > 0 else np.inf
                # 自由度近似 n-1（常见近似）
                p_nb = 2 * (1 - stats.t.cdf(abs(t_nb), df=n - 1)) if np.isfinite(t_nb) else 0.0
            else:
                t_nb, p_nb = np.nan, np.nan

            records.append({
                "A_vs_B": f"{a}_vs_{b}",
                "t_paired": float(t_stat) if t_stat is not None else np.nan,
                "p_paired": float(p_val) if p_val is not None else np.nan,
                "t_nb": float(t_nb) if t_nb is not None else np.nan,
                "p_nb": float(p_nb) if p_nb is not None else np.nan,
                "mean_diff_auc": float(np.nanmean(d)),
            })

    return pd.DataFrame(records)


def run_compare(
    dataset_id: int,
    pipeline_to_featureset: Dict[str, int],
    output_dir: str,
    db_path: str = DEFAULT_DB_PATH,
    use_linsvm: bool = False,
):
    """
    主入口：同一批 recording，比较四条 pipeline。
    pipeline_to_featureset: 映射 { 'P0_minimal_hp': fs_id0, 'P1_hp_avg_reref': fs_id1, ... }
    """
    os.makedirs(output_dir, exist_ok=True)

    # 1) 公共元数据（含 subject_id / normal / abnormal 等）
    df_meta = get_relevant_metadata(dataset_id, db_path, target_vars=["sex", "age", "status", "normal", "abnormal"])
    df_meta["label"] = infer_binary_label_from_meta(df_meta)
    df_meta = df_meta.dropna(subset=["label"]).copy()
    df_meta["label"] = df_meta["label"].astype(int)

    # 2) 载入每条 pipeline 的特征矩阵
    name_to_df = {}
    name_to_rids = {}
    for pname, fsid in pipeline_to_featureset.items():
        logger.info(f"Loading feature matrix for {pname} (featureset_id={fsid}) ...")
        df_feat = extract_feature_matrix_direct(dataset_id, fsid, db_path)
        if "recording_id" not in df_feat.columns:
            raise ValueError("Feature matrix missing 'recording_id' column.")
        name_to_df[pname] = df_feat
        name_to_rids[pname] = set(df_feat["recording_id"].tolist())

    # 3) 统一录波集合（四条 pipeline 交集）并构建固定折
    common_rids = set.intersection(*name_to_rids.values())
    logger.info(f"Common recordings across pipelines: {len(common_rids)}")
    df_meta_common = df_meta[df_meta["recording_id"].isin(common_rids)].copy()
    if df_meta_common.empty:
        raise ValueError("No common recordings across pipelines after label filtering.")
    splits, df_ordered = build_cv_splits(df_meta_common[["recording_id", "subject_id"]], n_splits=N_SPLITS)

    # 4) 分别训练与评估
    model_name = "linsvm" if use_linsvm else "logreg"
    fold_metrics: Dict[str, pd.DataFrame] = {}
    summaries: Dict[str, Dict[str, float]] = {}
    for pname, df_feat in name_to_df.items():
        # 对应 pipeline 也要裁剪到共同录波集合
        df_feat_use = df_feat[df_feat["recording_id"].isin(common_rids)].copy()
        df_fold, summary = train_eval_one_pipeline(
            df_feat=df_feat_use,
            df_meta=df_meta_common,
            splits=splits,
            df_ordered=df_ordered,
            model_name=model_name,
        )
        df_fold.to_csv(os.path.join(output_dir, f"fold_metrics_{pname}.csv"), index=False)
        fold_metrics[pname] = df_fold
        summaries[pname] = summary

    # 5) 汇总均值±标准差
    rows = []
    for pname, sm in summaries.items():
        rows.append({
            "pipeline": pname,
            "auc_mean": sm["auc_mean"], "auc_std": sm["auc_std"],
            "acc_mean": sm["accuracy_mean"], "acc_std": sm["accuracy_std"],
            "f1_mean": sm["f1_mean"], "f1_std": sm["f1_std"],
            "bacc_mean": sm["balanced_accuracy_mean"], "bacc_std": sm["balanced_accuracy_std"],
        })
    df_summary = pd.DataFrame(rows)
    df_summary.to_csv(os.path.join(output_dir, "summary_metrics.csv"), index=False)

    # 6) 统计比较（按折 AUC 配对）
    df_tests = paired_tests_by_auc(fold_metrics, k=N_SPLITS)
    df_tests.to_csv(os.path.join(output_dir, "paired_tests_auc.csv"), index=False)

    # 7) 简要文本总结
    def fmt(mean, std): return f"{mean:.4f} ± {std:.4f}"
    lines = []
    lines.append("Experiment 1: TUAB normal vs abnormal (relative bandpower; 4 pipelines)")
    lines.append(f"Dataset: {dataset_id}")
    lines.append(f"Common recordings: {len(common_rids)}")
    lines.append(f"Model: {'LinearSVM' if use_linsvm else 'LogisticRegression(L2, balanced)'}")
    lines.append(f"CV: GroupKFold(n_splits={N_SPLITS}) by subject_id")
    lines.append("")
    for _, r in df_summary.iterrows():
        lines.append(
            f"{r['pipeline']}: AUC {fmt(r['auc_mean'], r['auc_std'])}, "
            f"ACC {fmt(r['acc_mean'], r['acc_std'])}, "
            f"F1 {fmt(r['f1_mean'], r['f1_std'])}, "
            f"BACC {fmt(r['bacc_mean'], r['bacc_std'])}"
        )
    lines.append("")
    lines.append("Paired tests (AUC): see paired_tests_auc.csv (paired t and Nadeau–Bengio corrected t)")
    with open(os.path.join(output_dir, "summary.txt"), "w", encoding="utf-8") as f:
        f.write("\n".join(lines))

    logger.info("Done. Results saved to: %s", output_dir)


def main():
    parser = argparse.ArgumentParser(description="Experiment 1: TUAB normal vs abnormal comparison across 4 pipelines")
    parser.add_argument("--dataset", type=int, required=True)
    parser.add_argument("--featureset-P0", type=int, required=True, help="featureset id for P0_minimal_hp")
    parser.add_argument("--featureset-P1", type=int, required=True, help="featureset id for P1_hp_avg_reref")
    parser.add_argument("--featureset-P2", type=int, required=True, help="featureset id for P2_hp_notch50")
    parser.add_argument("--featureset-P3", type=int, required=True, help="featureset id for P3_bp_ica_auto")
    parser.add_argument("--output", type=str, required=True)
    parser.add_argument("--db", type=str, default=DEFAULT_DB_PATH)
    parser.add_argument("--use-linsvm", action="store_true", help="Use LinearSVM instead of LogisticRegression")
    args = parser.parse_args()

    pipeline_to_featureset = {
        "P0_minimal_hp": args.featureset_P0,
        "P1_hp_avg_reref": args.featureset_P1,
        "P2_hp_notch50": args.featureset_P2,
        "P3_ica_auto": args.featureset_P3,
    }

    run_compare(
        dataset_id=args.dataset,
        pipeline_to_featureset=pipeline_to_featureset,
        output_dir=args.output,
        db_path=args.db,
        use_linsvm=args.use_linsvm,
    )


if __name__ == "__main__":
    main()