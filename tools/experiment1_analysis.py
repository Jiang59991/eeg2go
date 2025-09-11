#!/usr/bin/env python3
import os
import sqlite3
import numpy as np
import pandas as pd
import pickle
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple, Optional
from sklearn.model_selection import GroupKFold
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, accuracy_score, f1_score, classification_report
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from logging_config import logger

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "experiment1")

class Experiment1Analyzer:
    """
    Experiment 1 Analyzer: feature aggregation, cross-validation, visualization.
    """

    def __init__(self):
        self.db_path = DB_PATH
        self.output_dir = OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        self.featuresets = [
            "exp1_bp_rel__P0_minimal_hp",
            "exp1_bp_rel__P1_hp_avg_reref", 
            "exp1_bp_rel__P2_hp_notch50",
            "exp1_bp_rel__P3_hp_ica_auto"
        ]
        self.bands = ["delta", "theta", "alpha", "beta"]
        self.channels = ["F3", "F4", "C3", "C4", "O1", "O2"]

    def get_tuab_subset_recordings(self) -> List[Dict]:
        """
        Get the list of TUAB subset recordings, one per subject, prioritizing abnormal and longest duration.
        Returns:
            List[Dict]: List of recording info dicts.
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id FROM datasets WHERE name = ?", ("TUAB_v3.0.1",))
        dataset_id = c.fetchone()[0]
        c.execute("""
            SELECT DISTINCT r.subject_id 
            FROM recordings r
            WHERE r.dataset_id = ? 
            ORDER BY r.subject_id
        """, (dataset_id,))
        subjects = [row[0] for row in c.fetchall()]
        recordings = []
        for subject_id in subjects:
            c.execute("""
                SELECT r.id, r.subject_id, r.filename, r.path, r.duration, rm.abnormal
                FROM recordings r
                JOIN recording_metadata rm ON r.id = rm.recording_id
                JOIN feature_values fv ON r.id = fv.recording_id
                JOIN fxdef fd ON fv.fxdef_id = fd.id
                JOIN feature_set_items fsi ON fd.id = fsi.fxdef_id
                JOIN feature_sets fs ON fsi.feature_set_id = fs.id
                WHERE r.dataset_id = ? AND r.subject_id = ? AND fs.name LIKE '%exp1_bp_rel%'
                ORDER BY rm.abnormal DESC, r.duration DESC
                LIMIT 1
            """, (dataset_id, subject_id))
            row = c.fetchone()
            if row:
                recordings.append({
                    'recording_id': row[0],
                    'subject_id': row[1], 
                    'filename': row[2],
                    'path': row[3],
                    'duration': row[4],
                    'is_abnormal': row[5] == '1' if isinstance(row[5], str) else bool(row[5])
                })
        conn.close()
        return recordings

    def extract_features_for_pipeline(self, featureset_name: str, recordings: List[Dict]) -> pd.DataFrame:
        """
        Extract features for a given pipeline/featureset.
        Args:
            featureset_name (str): Name of the featureset.
            recordings (List[Dict]): List of recording dicts.
        Returns:
            pd.DataFrame: Feature matrix (recording_id as index).
        """
        logger.info(f"Extracting features for {featureset_name}")
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id FROM feature_sets WHERE name = ?", (featureset_name,))
        featureset_id = c.fetchone()[0]
        conn.close()
        recording_ids = [r['recording_id'] for r in recordings]
        if not recording_ids:
            return pd.DataFrame()
        placeholders = ','.join(['?' for _ in recording_ids])
        query = f"""
            SELECT fv.recording_id, fv.value, fd.shortname, fd.chans
            FROM feature_values fv
            JOIN fxdef fd ON fv.fxdef_id = fd.id
            JOIN feature_set_items fsi ON fd.id = fsi.fxdef_id
            WHERE fsi.feature_set_id = ? AND fv.recording_id IN ({placeholders})
        """
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute(query, [featureset_id] + recording_ids)
        rows = c.fetchall()
        conn.close()
        if not rows:
            logger.error(f"No features found for {featureset_name}")
            return pd.DataFrame()
        feature_data = {}
        for recording_id, value, shortname, chans in rows:
            if recording_id not in feature_data:
                feature_data[recording_id] = {}
            try:
                import json
                if isinstance(value, str):
                    parsed_value = json.loads(value)
                else:
                    parsed_value = value
                if isinstance(parsed_value, list) and len(parsed_value) > 0:
                    if isinstance(parsed_value[0], dict) and 'value' in parsed_value[0]:
                        epoch_values = [epoch['value'] for epoch in parsed_value if 'value' in epoch]
                        if epoch_values:
                            final_value = sum(epoch_values) / len(epoch_values)
                        else:
                            continue
                    else:
                        final_value = sum(parsed_value) / len(parsed_value)
                elif isinstance(parsed_value, (int, float)):
                    final_value = parsed_value
                else:
                    logger.warning(f"Unknown feature value format for {recording_id}: {type(parsed_value)}")
                    continue
                feature_name = f"{shortname}_{chans}"
                feature_data[recording_id][feature_name] = final_value
            except Exception as e:
                logger.warning(f"Could not parse feature value for {recording_id}: {e}")
                continue
        feature_matrix = pd.DataFrame.from_dict(feature_data, orient='index')
        logger.info(f"Extracted {feature_matrix.shape[1]} features for {len(recordings)} recordings")
        if not feature_matrix.empty:
            logger.info(f"Feature columns: {list(feature_matrix.columns[:5])}...")
        return feature_matrix

    def aggregate_features(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        Aggregate features: median across channels and epochs for each band/channel.
        Args:
            feature_matrix (pd.DataFrame): Raw feature matrix.
        Returns:
            pd.DataFrame: Aggregated feature matrix.
        """
        if feature_matrix.empty:
            return pd.DataFrame()
        logger.info("Aggregating features...")
        logger.info(f"Available bands: {self.bands}")
        logger.info(f"Available channels: {self.channels}")
        logger.info(f"Feature matrix columns: {list(feature_matrix.columns[:10])}...")
        aggregated_features = {}
        for band in self.bands:
            for channel in self.channels:
                pattern = f"bp_rel_{band}_{channel}_{channel}"
                matching_cols = [col for col in feature_matrix.columns if pattern in col]
                if matching_cols:
                    logger.info(f"Found {len(matching_cols)} features for {pattern}")
                    median_val = feature_matrix[matching_cols].median(axis=1)
                    aggregated_features[f"bp_rel_{band}_{channel}_median"] = median_val
                else:
                    logger.warning(f"No features found for pattern: {pattern}")
        if not aggregated_features:
            logger.warning("No matching features found for aggregation, returning original features")
            return feature_matrix
        aggregated_df = pd.DataFrame(aggregated_features, index=feature_matrix.index)
        logger.info(f"Aggregated features shape: {aggregated_df.shape}")
        return aggregated_df

    def create_target_variable(self, recordings: List[Dict]) -> pd.Series:
        """
        Create target variable y (1=abnormal, 0=normal) using true labels.
        Args:
            recordings (List[Dict]): List of recording dicts.
        Returns:
            pd.Series: Target variable (index=recording_id).
        """
        recording_ids = [r['recording_id'] for r in recordings]
        labels = [1 if r['is_abnormal'] else 0 for r in recordings]
        y = pd.Series(labels, index=recording_ids, name='target')
        logger.info(f"Created target variable: {y.value_counts().to_dict()}")
        logger.info(f"Label mapping: 0=Normal, 1=Abnormal")
        return y

    def run_cross_validation(
        self, 
        X: pd.DataFrame, 
        y: pd.Series, 
        groups: pd.Series, 
        n_splits: int = 5, 
        model_name: str = 'rf'
    ) -> Dict:
        """
        Run cross-validation for the given model and data.
        Args:
            X (pd.DataFrame): Feature matrix.
            y (pd.Series): Target variable.
            groups (pd.Series): Group labels for GroupKFold.
            n_splits (int): Number of folds.
            model_name (str): Model type ('rf' or 'lr').
        Returns:
            Dict: Cross-validation results and metrics.
        """
        logger.info(f"Running {n_splits}-fold CV with {model_name} model")
        if model_name == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == 'lr':
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        cv = GroupKFold(n_splits=n_splits)
        fold_results = []
        oof_pred = np.zeros(len(X))
        oof_prob = np.zeros(len(X))
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
            logger.info(f"Fold {fold + 1}/{n_splits}")
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            model.fit(X_train_scaled, y_train)
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            auc = roc_auc_score(y_test, y_prob)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            oof_pred[test_idx] = y_pred
            oof_prob[test_idx] = y_prob
            fold_results.append({
                'fold': fold + 1,
                'auc': auc,
                'accuracy': acc,
                'f1': f1
            })
            logger.info(f"  Fold {fold + 1}: AUC={auc:.3f}, ACC={acc:.3f}, F1={f1:.3f}")
        oof_auc = roc_auc_score(y, oof_prob)
        oof_acc = accuracy_score(y, oof_pred)
        oof_f1 = f1_score(y, oof_pred)
        return {
            'fold_results': fold_results,
            'oof_predictions': oof_pred,
            'oof_probabilities': oof_prob,
            'oof_scores': {
                'auc': oof_auc,
                'accuracy': oof_acc,
                'f1': oof_f1
            },
            'average_scores': {
                'auc': np.mean([f['auc'] for f in fold_results]),
                'accuracy': np.mean([f['accuracy'] for f in fold_results]),
                'f1': np.mean([f['f1'] for f in fold_results])
            }
        }

    def compare_pipelines(
        self, 
        pipeline_features: Dict[str, pd.DataFrame], 
        y: pd.Series, 
        groups: pd.Series
    ) -> Dict:
        """
        Compare the performance of different pipelines.
        Args:
            pipeline_features (Dict[str, pd.DataFrame]): Feature matrices for each pipeline.
            y (pd.Series): Target variable.
            groups (pd.Series): Group labels.
        Returns:
            Dict: Results for each pipeline.
        """
        results = {}
        for pipeline_name, X in pipeline_features.items():
            logger.info(f"\nEvaluating {pipeline_name}")
            common_idx = X.index.intersection(y.index)
            logger.info(f"Feature matrix shape: {X.shape}")
            logger.info(f"Target variable shape: {y.shape}")
            logger.info(f"Common indices: {len(common_idx)}")
            if len(common_idx) < len(X.index):
                logger.warning(f"Removing {len(X.index) - len(common_idx)} samples due to missing targets")
            if len(common_idx) == 0:
                logger.error(f"No common indices found between features and targets for {pipeline_name}")
                continue
            X_aligned = X.loc[common_idx]
            y_aligned = y.loc[common_idx]
            groups_aligned = groups.loc[common_idx]
            cv_results = self.run_cross_validation(X_aligned, y_aligned, groups_aligned)
            results[pipeline_name] = cv_results
        return results

    def create_visualizations(self, results: Dict, save_plots: bool = True) -> pd.DataFrame:
        """
        Create visualizations for pipeline performance and differences.
        Args:
            results (Dict): Results from compare_pipelines.
            save_plots (bool): Whether to save plots to disk.
        Returns:
            pd.DataFrame: Summary results table.
        """
        logger.info("Creating visualizations...")
        if not results:
            logger.error("No results to visualize")
            return pd.DataFrame()
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        metrics = ['auc', 'accuracy', 'f1']
        metric_names = ['AUC', 'Accuracy', 'F1 Score']
        available_pipelines = [p for p in self.featuresets if p in results]
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            values = [results[pipeline]['oof_scores'][metric] for pipeline in available_pipelines]
            pipeline_mapping = {
                'P0_minimal_hp': 'P0',
                'P1_hp_avg_reref': 'P1', 
                'P2_hp_notch50': 'P2',
                'P3_hp_ica_auto': 'P3'
            }
            pipeline_names = [pipeline_mapping.get(p.split('__')[-1], p.split('__')[-1]) for p in available_pipelines]
            axes[i].bar(pipeline_names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            axes[i].set_title(f'{name} Comparison')
            axes[i].set_ylabel(name)
            axes[i].set_ylim(0, 1)
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(self.output_dir, 'pipeline_performance_comparison.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        pipeline_mapping = {
            'P0_minimal_hp': 'P0',
            'P1_hp_avg_reref': 'P1', 
            'P2_hp_notch50': 'P2',
            'P3_hp_ica_auto': 'P3'
        }
        pipeline_names = [pipeline_mapping.get(p.split('__')[-1], p.split('__')[-1]) for p in available_pipelines]
        auc_scores = [results[pipeline]['oof_scores']['auc'] for pipeline in available_pipelines]
        diff_matrix = np.zeros((len(pipeline_names), len(pipeline_names)))
        for i, name1 in enumerate(pipeline_names):
            for j, name2 in enumerate(pipeline_names):
                if i != j:
                    diff_matrix[i, j] = auc_scores[i] - auc_scores[j]
        plt.figure(figsize=(8, 6))
        sns.heatmap(diff_matrix, annot=True, fmt='.3f', cmap='RdBu_r', 
                   xticklabels=pipeline_names, yticklabels=pipeline_names,
                   center=0, cbar_kws={'label': 'ΔAUC'})
        plt.title('Pipeline Performance Differences (ΔAUC)')
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(self.output_dir, 'pipeline_differences_heatmap.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        pipeline_names = [pipeline_mapping.get(p.split('__')[-1], p.split('__')[-1]) for p in self.featuresets]
        results_df = pd.DataFrame({
            'Pipeline': pipeline_names,
            'AUC': [results[p]['oof_scores']['auc'] for p in self.featuresets],
            'Accuracy': [results[p]['oof_scores']['accuracy'] for p in self.featuresets],
            'F1 Score': [results[p]['oof_scores']['f1'] for p in self.featuresets],
            'Avg AUC (CV)': [results[p]['average_scores']['auc'] for p in self.featuresets],
            'Avg Accuracy (CV)': [results[p]['average_scores']['accuracy'] for p in self.featuresets],
            'Avg F1 (CV)': [results[p]['average_scores']['f1'] for p in self.featuresets]
        })
        print("\n=== Experiment 1 Results Summary ===")
        print(results_df.round(3))
        if save_plots:
            results_df.to_csv(os.path.join(self.output_dir, 'experiment1_results.csv'), index=False)
        return results_df

    def print_dataset_info(
        self, 
        recordings: List[Dict], 
        y: pd.Series, 
        all_features: Dict[str, pd.DataFrame]
    ) -> List[str]:
        """
        Print dataset information and statistics.
        Args:
            recordings (List[Dict]): List of recording dicts.
            y (pd.Series): Target variable.
            all_features (Dict[str, pd.DataFrame]): All pipeline features.
        Returns:
            List[str]: Output lines for summary.
        """
        output_lines = []
        total_recordings = len(recordings)
        total_subjects = len(set(r['subject_id'] for r in recordings))
        normal_count = (y == 0).sum()
        abnormal_count = (y == 1).sum()
        normal_ratio = normal_count / total_recordings * 100
        abnormal_ratio = abnormal_count / total_recordings * 100
        if normal_count > 0:
            ratio_abnormal_to_normal = abnormal_count / normal_count
            ratio_str = f"{abnormal_count}:{normal_count}"
            percentage_str = f"{abnormal_ratio:.1f}% abnormal"
        else:
            ratio_str = "N/A"
            percentage_str = "N/A"
        channel_coverage = self.calculate_channel_coverage(all_features)
        output_lines.extend([
            "="*60,
            "DATASET INFORMATION",
            "="*60,
            f"Total Recordings: {total_recordings}",
            f"Total Subjects: {total_subjects}",
            f"Recordings per Subject: {total_recordings / total_subjects:.2f}",
            "",
            "Label Distribution:",
            f"  Normal (0): {normal_count} ({normal_ratio:.1f}%)",
            f"  Abnormal (1): {abnormal_count} ({abnormal_ratio:.1f}%)",
            f"  Class Ratio: {ratio_str} ({percentage_str})",
            "",
            "Feature Information:"
        ])
        for pipeline_name, features in all_features.items():
            short_name = pipeline_name.split('__')[-1]
            feature_count = features.shape[1]
            sample_count = features.shape[0]
            output_lines.append(f"  {short_name}: {feature_count} features, {sample_count} samples")
        output_lines.extend([
            "",
            "Data Completeness:"
        ])
        for pipeline_name, features in all_features.items():
            short_name = pipeline_name.split('__')[-1]
            missing_ratio = features.isnull().sum().sum() / (features.shape[0] * features.shape[1]) * 100
            output_lines.append(f"  {short_name}: {missing_ratio:.2f}% missing values")
        output_lines.extend([
            "",
            "Data Quality Statistics:"
        ])
        for pipeline_name, features in all_features.items():
            short_name = pipeline_name.split('__')[-1]
            if not features.empty:
                feature_stds = features.std()
                constant_features = (feature_stds == 0).sum()
                output_lines.append(f"  {short_name}: {constant_features} constant features out of {features.shape[1]}")
        output_lines.extend([
            "",
            "Channel Coverage (P0 baseline pipeline):"
        ])
        for channel in self.channels:
            if channel in channel_coverage:
                coverage_info = channel_coverage[channel]
                output_lines.append(f"  {channel}: {coverage_info['median']:.1f}% [{coverage_info['iqr']:.1f}]")
        output_lines.extend([
            "="*60,
            ""
        ])
        for line in output_lines:
            print(line)
        output_file = os.path.join(self.output_dir, "data_summary.txt")
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(output_lines))
        logger.info(f"Dataset summary saved to {output_file}")
        return output_lines

    def calculate_channel_coverage(
        self, 
        all_features: Dict[str, pd.DataFrame]
    ) -> Dict[str, Dict]:
        """
        Calculate channel coverage (based on P0 baseline pipeline).
        For each channel, compute the non-missing ratio for each band, then median and IQR.
        Args:
            all_features (Dict[str, pd.DataFrame]): All pipeline features.
        Returns:
            Dict[str, Dict]: Channel coverage statistics.
        """
        p0_pipeline = None
        for pipeline_name in all_features.keys():
            if 'P0_minimal_hp' in pipeline_name:
                p0_pipeline = all_features[pipeline_name]
                break
        if p0_pipeline is None or p0_pipeline.empty:
            logger.warning("P0 baseline pipeline not found for channel coverage calculation")
            return {}
        channel_coverage = {}
        for channel in self.channels:
            band_coverages = []
            for band in self.bands:
                pattern = f"bp_rel_{band}_{channel}_median"
                matching_cols = [col for col in p0_pipeline.columns if pattern in col]
                if matching_cols:
                    non_missing_count = p0_pipeline[matching_cols].notna().sum().sum()
                    total_count = len(matching_cols) * len(p0_pipeline)
                    coverage = (non_missing_count / total_count) * 100
                    band_coverages.append(coverage)
                    logger.info(f"Channel {channel}, Band {band}: {coverage:.1f}% coverage")
                else:
                    band_coverages.append(0.0)
                    logger.warning(f"No features found for channel {channel}, band {band}")
            if band_coverages:
                median_coverage = np.median(band_coverages)
                q1 = np.percentile(band_coverages, 25)
                q3 = np.percentile(band_coverages, 75)
                iqr = q3 - q1
                channel_coverage[channel] = {
                    'median': median_coverage,
                    'iqr': iqr,
                    'band_coverages': band_coverages
                }
        return channel_coverage

    def calculate_fold_statistics(
        self, 
        y: pd.Series, 
        groups: pd.Series, 
        n_splits: int = 5
    ) -> Dict:
        """
        Calculate statistics for each cross-validation fold.
        Args:
            y (pd.Series): Target variable.
            groups (pd.Series): Group labels.
            n_splits (int): Number of folds.
        Returns:
            Dict: Fold statistics summary.
        """
        from sklearn.model_selection import GroupKFold
        cv = GroupKFold(n_splits=n_splits)
        fold_stats = []
        for fold, (train_idx, test_idx) in enumerate(cv.split(y, y, groups)):
            y_test = y.iloc[test_idx]
            groups_test = groups.iloc[test_idx]
            total_subjects = len(set(groups_test))
            abnormal_count = (y_test == 1).sum()
            normal_count = (y_test == 0).sum()
            fold_stats.append({
                'fold': fold + 1,
                'total_subjects': total_subjects,
                'abnormal_count': abnormal_count,
                'normal_count': normal_count
            })
        total_subjects_list = [f['total_subjects'] for f in fold_stats]
        abnormal_counts_list = [f['abnormal_count'] for f in fold_stats]
        normal_counts_list = [f['normal_count'] for f in fold_stats]
        stats = {
            'FoldTotMed': np.median(total_subjects_list),
            'FoldTotMin': np.min(total_subjects_list),
            'FoldTotMax': np.max(total_subjects_list),
            'FoldAbnMed': np.median(abnormal_counts_list),
            'FoldAbnMin': np.min(abnormal_counts_list),
            'FoldAbnMax': np.max(abnormal_counts_list),
            'FoldNorMed': np.median(normal_counts_list),
            'FoldNorMin': np.min(normal_counts_list),
            'FoldNorMax': np.max(normal_counts_list)
        }
        return stats

    def print_fold_statistics(
        self, 
        y: pd.Series, 
        groups: pd.Series
    ) -> List[str]:
        """
        Print cross-validation fold statistics.
        Args:
            y (pd.Series): Target variable.
            groups (pd.Series): Group labels.
        Returns:
            List[str]: Output lines for fold statistics.
        """
        fold_stats = self.calculate_fold_statistics(y, groups)
        output_lines = [
            "",
            "="*60,
            "FOLD STATISTICS",
            "="*60,
            "Cross-validation fold statistics:",
            f"\\FoldTotMed{{}}: {fold_stats['FoldTotMed']:.0f}",
            f"\\FoldTotMin{{}} / \\FoldTotMax{{}}: {fold_stats['FoldTotMin']:.0f} / {fold_stats['FoldTotMax']:.0f}",
            f"\\FoldAbnMed{{}}: {fold_stats['FoldAbnMed']:.0f}",
            f"\\FoldAbnMin{{}} / \\FoldAbnMax{{}}: {fold_stats['FoldAbnMin']:.0f} / {fold_stats['FoldAbnMax']:.0f}",
            f"\\FoldNorMed{{}}: {fold_stats['FoldNorMed']:.0f}",
            f"\\FoldNorMin{{}} / \\FoldNorMax{{}}: {fold_stats['FoldNorMin']:.0f} / {fold_stats['FoldNorMax']:.0f}",
            "="*60,
            ""
        ]
        for line in output_lines:
            print(line)
        return output_lines

def main() -> Optional[Dict]:
    """
    Main function: run the full analysis for Experiment 1.
    Returns:
        Optional[Dict]: Results dictionary if successful, else None.
    """
    analyzer = Experiment1Analyzer()
    recordings = analyzer.get_tuab_subset_recordings()
    if not recordings:
        logger.error("No recordings found for Experiment 1")
        return
    all_features = {}
    for featureset_name in analyzer.featuresets:
        logger.info(f"Processing {featureset_name}")
        raw_features = analyzer.extract_features_for_pipeline(featureset_name, recordings)
        if not raw_features.empty:
            aggregated_features = analyzer.aggregate_features(raw_features)
            all_features[featureset_name] = aggregated_features
        else:
            logger.warning(f"No features extracted for {featureset_name}")
    y = analyzer.create_target_variable(recordings)
    recording_to_subject = {r['recording_id']: r['subject_id'] for r in recordings}
    groups = pd.Series([recording_to_subject.get(rid, rid) for rid in y.index], index=y.index)
    analyzer.print_dataset_info(recordings, y, all_features)
    fold_stats_lines = analyzer.print_fold_statistics(y, groups)
    results = analyzer.compare_pipelines(all_features, y, groups)
    results_df = analyzer.create_visualizations(results)
    output_file = os.path.join(OUTPUT_DIR, "experiment1_complete_results.pkl")
    with open(output_file, 'wb') as f:
        pickle.dump({
            'recordings': recordings,
            'features': all_features,
            'target': y,
            'groups': groups,
            'cv_results': results,
            'results_summary': results_df
        }, f)
    summary_file = os.path.join(OUTPUT_DIR, "data_summary.txt")
    with open(summary_file, 'a', encoding='utf-8') as f:
        f.write('\n'.join(fold_stats_lines))
    logger.info(f"Complete results saved to {output_file}")
    logger.info(f"Updated data summary with fold statistics to {summary_file}")
    return results

if __name__ == "__main__":
    print("Starting Experiment 1 analysis...")
    try:
        main()
        print("Experiment 1 analysis completed successfully!")
    except Exception as e:
        print(f"Error in Experiment 1 analysis: {e}")
        import traceback
        traceback.print_exc()
