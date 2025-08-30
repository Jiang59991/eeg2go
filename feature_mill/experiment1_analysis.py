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
# 移除FeatureMatrix导入，直接使用数据库查询

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "database", "eeg2go.db")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "..", "outputs", "experiment1")

class Experiment1Analyzer:
    """
    实验1分析器：聚合特征、执行交叉验证、生成可视化
    """
    
    def __init__(self):
        self.db_path = DB_PATH
        self.output_dir = OUTPUT_DIR
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 实验1的featureset名称
        self.featuresets = [
            "exp1_bp_rel__P0_minimal_hp",
            "exp1_bp_rel__P1_hp_avg_reref", 
            "exp1_bp_rel__P2_hp_notch50",
            "exp1_bp_rel__P3_hp_ica_auto"
        ]
        
        # 频段和通道
        self.bands = ["delta", "theta", "alpha", "beta"]
        self.channels = ["F3", "F4", "C3", "C4", "O1", "O2"]
        
    def get_tuab_subset_recordings(self) -> List[Dict]:
        """获取TUAB子集的recording列表"""
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        
        # 获取TUAB数据集ID
        c.execute("SELECT id FROM datasets WHERE name = ?", ("TUAB_v3.0.1",))
        dataset_id = c.fetchone()[0]
        
        # 为每个subject选择recording（优先选择abnormal，然后选择最长的）
        c.execute("""
            SELECT DISTINCT r.subject_id 
            FROM recordings r
            WHERE r.dataset_id = ? 
            ORDER BY r.subject_id
        """, (dataset_id,))
        
        subjects = [row[0] for row in c.fetchall()]
        
        # 为每个subject选择recording（只选择有特征数据的）
        recordings = []
        for subject_id in subjects:
            # 优先选择abnormal标签的recording，然后按duration排序，但只选择有特征数据的
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
        """为指定pipeline提取特征"""
        logger.info(f"Extracting features for {featureset_name}")
        
        # 获取featureset ID
        conn = sqlite3.connect(self.db_path)
        c = conn.cursor()
        c.execute("SELECT id FROM feature_sets WHERE name = ?", (featureset_name,))
        featureset_id = c.fetchone()[0]
        conn.close()
        
        # 直接查询特征值
        recording_ids = [r['recording_id'] for r in recordings]
        if not recording_ids:
            return pd.DataFrame()
        
        # 构建查询
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
        
        # 构建特征矩阵
        feature_data = {}
        for recording_id, value, shortname, chans in rows:
            if recording_id not in feature_data:
                feature_data[recording_id] = {}
            
            # 解析特征值（处理复杂的JSON格式）
            try:
                import json
                if isinstance(value, str):
                    parsed_value = json.loads(value)
                else:
                    parsed_value = value
                
                # 处理不同的特征值格式
                if isinstance(parsed_value, list) and len(parsed_value) > 0:
                    # 如果是epoch列表，计算平均值
                    if isinstance(parsed_value[0], dict) and 'value' in parsed_value[0]:
                        # 提取所有epoch的value并计算平均值
                        epoch_values = [epoch['value'] for epoch in parsed_value if 'value' in epoch]
                        if epoch_values:
                            final_value = sum(epoch_values) / len(epoch_values)
                        else:
                            continue
                    else:
                        # 如果是简单的数值列表，计算平均值
                        final_value = sum(parsed_value) / len(parsed_value)
                elif isinstance(parsed_value, (int, float)):
                    final_value = parsed_value
                else:
                    logger.warning(f"Unknown feature value format for {recording_id}: {type(parsed_value)}")
                    continue
                
                # 生成特征名称
                feature_name = f"{shortname}_{chans}"
                feature_data[recording_id][feature_name] = final_value
            except Exception as e:
                logger.warning(f"Could not parse feature value for {recording_id}: {e}")
                continue
        
        # 转换为DataFrame
        feature_matrix = pd.DataFrame.from_dict(feature_data, orient='index')
        
        logger.info(f"Extracted {feature_matrix.shape[1]} features for {len(recordings)} recordings")
        if not feature_matrix.empty:
            logger.info(f"Feature columns: {list(feature_matrix.columns[:5])}...")  # 显示前5个特征名
        return feature_matrix
    
    def aggregate_features(self, feature_matrix: pd.DataFrame) -> pd.DataFrame:
        """
        聚合特征：通道中位数 + epoch到recording聚合
        """
        if feature_matrix.empty:
            return pd.DataFrame()
        
        logger.info("Aggregating features...")
        logger.info(f"Available bands: {self.bands}")
        logger.info(f"Available channels: {self.channels}")
        logger.info(f"Feature matrix columns: {list(feature_matrix.columns[:10])}...")  # 显示前10个特征名
        
        # 1. 通道中位数聚合
        aggregated_features = {}
        
        for band in self.bands:
            for channel in self.channels:
                # 查找该频段和通道的所有特征（格式：bp_rel_{band}_{channel}_{channel}）
                pattern = f"bp_rel_{band}_{channel}_{channel}"
                matching_cols = [col for col in feature_matrix.columns if pattern in col]
                
                if matching_cols:
                    logger.info(f"Found {len(matching_cols)} features for {pattern}")
                    # 计算中位数
                    median_val = feature_matrix[matching_cols].median(axis=1)
                    aggregated_features[f"bp_rel_{band}_{channel}_median"] = median_val
                else:
                    logger.warning(f"No features found for pattern: {pattern}")
        
        # 如果没有找到匹配的特征，直接返回原始特征矩阵
        if not aggregated_features:
            logger.warning("No matching features found for aggregation, returning original features")
            return feature_matrix
        
        # 2. 创建聚合后的DataFrame
        aggregated_df = pd.DataFrame(aggregated_features, index=feature_matrix.index)
        
        logger.info(f"Aggregated features shape: {aggregated_df.shape}")
        return aggregated_df
    
    def create_target_variable(self, recordings: List[Dict]) -> pd.Series:
        """
        创建目标变量y（使用真实的normal/abnormal标签）
        """
        # 使用真实的normal/abnormal标签，使用recording_id作为索引
        recording_ids = [r['recording_id'] for r in recordings]
        labels = [1 if r['is_abnormal'] else 0 for r in recordings]  # 1=abnormal, 0=normal
        
        y = pd.Series(labels, index=recording_ids, name='target')
        
        logger.info(f"Created target variable: {y.value_counts().to_dict()}")
        logger.info(f"Label mapping: 0=Normal, 1=Abnormal")
        return y
    
    def run_cross_validation(self, X: pd.DataFrame, y: pd.Series, groups: pd.Series, 
                           n_splits: int = 5, model_name: str = 'rf') -> Dict:
        """
        执行交叉验证
        """
        logger.info(f"Running {n_splits}-fold CV with {model_name} model")
        
        # 选择模型
        if model_name == 'rf':
            model = RandomForestClassifier(n_estimators=100, random_state=42)
        elif model_name == 'lr':
            model = LogisticRegression(random_state=42, max_iter=1000)
        else:
            raise ValueError(f"Unknown model: {model_name}")
        
        # 使用GroupKFold
        cv = GroupKFold(n_splits=n_splits)
        
        # 存储结果
        fold_results = []
        oof_pred = np.zeros(len(X))
        oof_prob = np.zeros(len(X))
        
        for fold, (train_idx, test_idx) in enumerate(cv.split(X, y, groups)):
            logger.info(f"Fold {fold + 1}/{n_splits}")
            
            X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
            y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]
            
            # 特征标准化
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(X_train)
            X_test_scaled = scaler.transform(X_test)
            
            # 训练模型
            model.fit(X_train_scaled, y_train)
            
            # 预测
            y_pred = model.predict(X_test_scaled)
            y_prob = model.predict_proba(X_test_scaled)[:, 1]
            
            # 计算指标
            auc = roc_auc_score(y_test, y_prob)
            acc = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred)
            
            # 存储OOF预测
            oof_pred[test_idx] = y_pred
            oof_prob[test_idx] = y_prob
            
            fold_results.append({
                'fold': fold + 1,
                'auc': auc,
                'accuracy': acc,
                'f1': f1
            })
            
            logger.info(f"  Fold {fold + 1}: AUC={auc:.3f}, ACC={acc:.3f}, F1={f1:.3f}")
        
        # 计算整体OOF指标
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
    
    def compare_pipelines(self, pipeline_features: Dict[str, pd.DataFrame], 
                         y: pd.Series, groups: pd.Series) -> Dict:
        """
        比较不同pipeline的性能
        """
        results = {}
        
        for pipeline_name, X in pipeline_features.items():
            logger.info(f"\nEvaluating {pipeline_name}")
            
            # 确保X和y的索引对齐
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
            
            # 运行交叉验证
            cv_results = self.run_cross_validation(X_aligned, y_aligned, groups_aligned)
            results[pipeline_name] = cv_results
            
        return results
    
    def create_visualizations(self, results: Dict, save_plots: bool = True):
        """
        创建可视化图表
        """
        logger.info("Creating visualizations...")
        
        if not results:
            logger.error("No results to visualize")
            return pd.DataFrame()
        
        # 1. 性能比较图
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        metrics = ['auc', 'accuracy', 'f1']
        metric_names = ['AUC', 'Accuracy', 'F1 Score']
        
        # 只处理有结果的pipeline
        available_pipelines = [p for p in self.featuresets if p in results]
        
        for i, (metric, name) in enumerate(zip(metrics, metric_names)):
            values = [results[pipeline]['oof_scores'][metric] for pipeline in available_pipelines]
            pipeline_names = [p.split('__')[-1] for p in available_pipelines]  # 提取pipeline名称
            
            axes[i].bar(pipeline_names, values, color=['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728'])
            axes[i].set_title(f'{name} Comparison')
            axes[i].set_ylabel(name)
            axes[i].set_ylim(0, 1)
            
            # 添加数值标签
            for j, v in enumerate(values):
                axes[i].text(j, v + 0.01, f'{v:.3f}', ha='center', va='bottom')
        
        plt.tight_layout()
        if save_plots:
            plt.savefig(os.path.join(self.output_dir, 'pipeline_performance_comparison.png'), 
                       dpi=300, bbox_inches='tight')
        plt.show()
        
        # 2. 统计显著性热图
        pipeline_names = [p.split('__')[-1] for p in available_pipelines]
        auc_scores = [results[pipeline]['oof_scores']['auc'] for pipeline in available_pipelines]
        
        # 计算差异矩阵
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
        
        # 3. 详细结果表格
        results_df = pd.DataFrame({
            'Pipeline': [p.split('__')[-1] for p in self.featuresets],
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

def main():
    """
    主函数：执行实验1的完整分析
    """
    analyzer = Experiment1Analyzer()
    
    # 1. 获取TUAB子集
    recordings = analyzer.get_tuab_subset_recordings()
    
    if not recordings:
        logger.error("No recordings found for Experiment 1")
        return
    
    # 2. 提取所有pipeline的特征
    all_features = {}
    for featureset_name in analyzer.featuresets:
        logger.info(f"Processing {featureset_name}")
        
        # 提取原始特征
        raw_features = analyzer.extract_features_for_pipeline(featureset_name, recordings)
        
        if not raw_features.empty:
            # 聚合特征
            aggregated_features = analyzer.aggregate_features(raw_features)
            all_features[featureset_name] = aggregated_features
        else:
            logger.warning(f"No features extracted for {featureset_name}")
    
    # 3. 创建目标变量
    y = analyzer.create_target_variable(recordings)
    
    # 4. 创建groups（使用subject_id作为group，但索引是recording_id）
    recording_to_subject = {r['recording_id']: r['subject_id'] for r in recordings}
    groups = pd.Series([recording_to_subject.get(rid, rid) for rid in y.index], index=y.index)
    
    # 5. 比较所有pipeline
    results = analyzer.compare_pipelines(all_features, y, groups)
    
    # 6. 创建可视化
    results_df = analyzer.create_visualizations(results)
    
    # 7. 保存完整结果
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
    
    logger.info(f"Complete results saved to {output_file}")
    
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
