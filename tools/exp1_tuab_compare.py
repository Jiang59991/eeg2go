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

REQUIRED_CHANNELS = ["F3", "F4", "C3", "C4", "O1", "O2"]
N_SPLITS = 5


def infer_binary_label_from_meta(df_meta: pd.DataFrame) -> pd.Series:
    """
    Infer binary label from metadata DataFrame.
    - y=1: abnormal
    - y=0: normal
    Priority: abnormal > normal > status (contains 'abnormal'/'normal')
    Rows that cannot be parsed will be NaN.
    Args:
        df_meta (pd.DataFrame): Metadata DataFrame.
    Returns:
        pd.Series: Series of binary labels (1 for abnormal, 0 for normal, NaN for unknown).
    """
    y = pd.Series(index=df_meta.index, dtype=float)

    def truthy(v) -> bool:
        if v is None:
            return False
        if isinstance(v, (int, float)):
            return v != 0
        s = str(v).strip().lower()
        return s in ("1", "y", "yes", "true", "t") or s == "abnormal" or s == "normal"

    if "abnormal" in df_meta.columns:
        mask = df_meta["abnormal"].apply(truthy)
        y.loc[mask] = 1.0

    if "normal" in df_meta.columns:
        mask = df_meta["normal"].apply(truthy)
        y.loc[mask & y.isna()] = 0.0

    if "status" in df_meta.columns:
        s = df_meta["status"].astype(str).str.lower()
        y.loc[s.str.contains("abnormal", na=False) & y.isna()] = 1.0
        y.loc[s.str.contains("normal", na=False) & y.isna()] = 0.0

    return y


def select_feature_columns(df_feat: pd.DataFrame) -> List[str]:
    """
    Select feature columns: only relative bandpower (bp_rel) for required channels and median aggregation.
    Args:
        df_feat (pd.DataFrame): Feature DataFrame.
    Returns:
        List[str]: List of selected feature column names.
    """
    cols = []
    for col in df_feat.columns:
        if not col.endswith("_median"):
            continue
        if "_bp_rel_" not in col:
            continue
        parts = col.split("_")
        if len(parts) < 3:
            continue
        ch = parts[-2].upper()
        if ch in [c.upper() for c in REQUIRED_CHANNELS]:
            cols.append(col)
    return sorted(cols)


def build_cv_splits(
    df_all: pd.DataFrame, n_splits: int = N_SPLITS
) -> Tuple[List[Tuple[np.ndarray, np.ndarray]], pd.DataFrame]:
    """
    Build GroupKFold splits by subject_id, with fixed order.
    Args:
        df_all (pd.DataFrame): DataFrame with at least 'subject_id' and 'recording_id'.
        n_splits (int): Number of splits.
    Returns:
        Tuple[List[Tuple[np.ndarray, np.ndarray]], pd.DataFrame]: List of (train_idx, test_idx) and ordered DataFrame.
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
    Train and evaluate a single pipeline on fixed splits.
    Args:
        df_feat (pd.DataFrame): Feature DataFrame.
        df_meta (pd.DataFrame): Metadata DataFrame with labels.
        splits (List[Tuple[np.ndarray, np.ndarray]]): List of (train_idx, test_idx) splits.
        df_ordered (pd.DataFrame): Ordered DataFrame for alignment.
        model_name (str): Model name, "logreg" or "linsvm".
    Returns:
        Tuple[pd.DataFrame, Dict[str, float]]: Fold metrics DataFrame and summary dict.
    """
    df_feat_idxed = df_feat.set_index("recording_id")
    X_rows = []
    for rid in df_ordered["recording_id"].tolist():
        if rid not in df_feat_idxed.index:
            raise ValueError(f"Missing recording_id={rid} in feature matrix.")
        X_rows.append(df_feat_idxed.loc[rid])
    df_X = pd.DataFrame(X_rows).reset_index()

    feat_cols = select_feature_columns(df_X)
    if not feat_cols:
        raise ValueError("No valid bp_rel *_median feature columns found.")
    X_all = df_X[feat_cols].values

    df_meta_idxed = df_meta.set_index("recording_id").loc[df_ordered["recording_id"].values]
    y_all = df_meta_idxed["label"].values.astype(int)

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

        if model_name == "logreg":
            proba = clf.predict_proba(X_te)[:, 1]
            y_pred = (proba >= 0.5).astype(int)
            auc = roc_auc_score(y_te, proba) if len(np.unique(y_te)) > 1 else np.nan
        else:
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


def paired_tests_by_auc(
    results: Dict[str, pd.DataFrame], k: int = N_SPLITS
) -> pd.DataFrame:
    """
    Perform paired t-tests and Nadeau–Bengio corrected t-tests on AUCs between all pipeline pairs.
    Args:
        results (Dict[str, pd.DataFrame]): Dict of pipeline name to fold metrics DataFrame.
        k (int): Number of folds.
    Returns:
        pd.DataFrame: DataFrame of paired test results.
    """
    names = list(results.keys())
    records = []

    def nb_correction(var, k):
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

            t_stat, p_val = stats.ttest_rel(auc_a, auc_b, nan_policy="omit")

            d_clean = d[~np.isnan(d)]
            n = len(d_clean)
            if n > 1:
                mean_d = np.mean(d_clean)
                var_d = np.var(d_clean, ddof=1)
                c = nb_correction(var_d, k)
                se = np.sqrt(var_d * c)
                t_nb = mean_d / se if se > 0 else np.inf
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
) -> None:
    """
    Main entry: Compare four pipelines on the same set of recordings.
    Args:
        dataset_id (int): Dataset ID.
        pipeline_to_featureset (Dict[str, int]): Mapping from pipeline name to featureset id.
        output_dir (str): Output directory.
        db_path (str): Path to database.
        use_linsvm (bool): Whether to use LinearSVM instead of LogisticRegression.
    Returns:
        None
    """
    os.makedirs(output_dir, exist_ok=True)

    df_meta = get_relevant_metadata(dataset_id, db_path, target_vars=["sex", "age", "status", "normal", "abnormal"])
    df_meta["label"] = infer_binary_label_from_meta(df_meta)
    df_meta = df_meta.dropna(subset=["label"]).copy()
    df_meta["label"] = df_meta["label"].astype(int)

    name_to_df = {}
    name_to_rids = {}
    for pname, fsid in pipeline_to_featureset.items():
        logger.info(f"Loading feature matrix for {pname} (featureset_id={fsid}) ...")
        df_feat = extract_feature_matrix_direct(dataset_id, fsid, db_path)
        if "recording_id" not in df_feat.columns:
            raise ValueError("Feature matrix missing 'recording_id' column.")
        name_to_df[pname] = df_feat
        name_to_rids[pname] = set(df_feat["recording_id"].tolist())

    common_rids = set.intersection(*name_to_rids.values())
    logger.info(f"Common recordings across pipelines: {len(common_rids)}")
    df_meta_common = df_meta[df_meta["recording_id"].isin(common_rids)].copy()
    if df_meta_common.empty:
        raise ValueError("No common recordings across pipelines after label filtering.")
    splits, df_ordered = build_cv_splits(df_meta_common[["recording_id", "subject_id"]], n_splits=N_SPLITS)

    model_name = "linsvm" if use_linsvm else "logreg"
    fold_metrics: Dict[str, pd.DataFrame] = {}
    summaries: Dict[str, Dict[str, float]] = {}
    for pname, df_feat in name_to_df.items():
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

    df_tests = paired_tests_by_auc(fold_metrics, k=N_SPLITS)
    df_tests.to_csv(os.path.join(output_dir, "paired_tests_auc.csv"), index=False)

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


def main() -> None:
    """
    Main function to parse arguments and run the comparison experiment.
    Returns:
        None
    """
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