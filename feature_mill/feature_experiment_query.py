import pandas as pd
import json
from typing import Dict, List, Optional, Any
from feature_mill.experiment_result_manager import ExperimentResultManager
from logging_config import logger  # global logger


class FeatureExperimentQuery:
    """
    Feature experiment query tool.
    """

    def __init__(self, db_path: str = "database/eeg2go.db") -> None:
        """
        Initialize the FeatureExperimentQuery with a database path.

        Args:
            db_path (str): Path to the database.
        """
        self.result_manager = ExperimentResultManager(db_path)

    def get_feature_correlation_summary(
        self,
        fxdef_id: Optional[int] = None,
        feature_name: Optional[str] = None,
        target_variable: str = 'age'
    ) -> Dict[str, Any]:
        """
        Get summary statistics of feature correlation with a target variable.

        Args:
            fxdef_id (Optional[int]): Feature definition ID.
            feature_name (Optional[str]): Feature name.
            target_variable (str): Target variable.

        Returns:
            Dict[str, Any]: Correlation summary information.
        """
        try:
            history_df = self.result_manager.get_feature_correlation_history(
                fxdef_id=fxdef_id,
                feature_name=feature_name,
                target_variable=target_variable
            )

            if len(history_df) == 0:
                return {
                    'has_correlation_data': False,
                    'message': f'未找到特征 {feature_name or fxdef_id} 与 {target_variable} 的相关性数据'
                }

            summary = {
                'has_correlation_data': True,
                'target_variable': target_variable,
                'total_experiments': len(history_df),
                'avg_correlation': history_df['correlation_coefficient'].mean(),
                'max_correlation': history_df['correlation_coefficient'].max(),
                'min_correlation': history_df['correlation_coefficient'].min(),
                'significant_count': len(history_df[history_df['significance_level'] != 'ns']),
                'significant_ratio': len(history_df[history_df['significance_level'] != 'ns']) / len(history_df),
                'best_rank': history_df['rank_position'].min(),
                'avg_rank': history_df['rank_position'].mean(),
                'recent_results': history_df.head(5).to_dict('records')
            }

            # Correlation strength
            avg_abs_corr = abs(summary['avg_correlation'])
            if avg_abs_corr >= 0.5:
                summary['correlation_strength'] = '强相关'
            elif avg_abs_corr >= 0.3:
                summary['correlation_strength'] = '中等相关'
            elif avg_abs_corr >= 0.1:
                summary['correlation_strength'] = '弱相关'
            else:
                summary['correlation_strength'] = '无相关'

            # Correlation direction
            if summary['avg_correlation'] > 0:
                summary['correlation_direction'] = '正相关'
            elif summary['avg_correlation'] < 0:
                summary['correlation_direction'] = '负相关'
            else:
                summary['correlation_direction'] = '无相关'

            return summary

        except Exception as e:
            logger.error(f"获取特征相关性汇总失败: {e}")
            return {'has_correlation_data': False, 'error': str(e)}

    def get_feature_importance_summary(
        self,
        fxdef_id: Optional[int] = None,
        feature_name: Optional[str] = None,
        target_variable: str = 'age'
    ) -> Dict[str, Any]:
        """
        Get summary statistics of feature importance for a target variable.

        Args:
            fxdef_id (Optional[int]): Feature definition ID.
            feature_name (Optional[str]): Feature name.
            target_variable (str): Target variable.

        Returns:
            Dict[str, Any]: Importance summary information.
        """
        try:
            history_df = self.result_manager.get_feature_importance_history(
                fxdef_id=fxdef_id,
                feature_name=feature_name,
                target_variable=target_variable
            )

            if len(history_df) == 0:
                return {
                    'has_importance_data': False,
                    'message': f'未找到特征 {feature_name or fxdef_id} 与 {target_variable} 的重要性数据'
                }

            classification_importance = history_df[history_df['result_type'] == 'classification_importance']
            selection_importance = history_df[history_df['result_type'] == 'selection_score']

            summary = {
                'has_importance_data': True,
                'target_variable': target_variable,
                'total_experiments': len(history_df),
                'classification_experiments': len(classification_importance),
                'selection_experiments': len(selection_importance)
            }

            if len(classification_importance) > 0:
                summary['classification_importance'] = {
                    'avg_importance': classification_importance['importance_score'].mean(),
                    'max_importance': classification_importance['importance_score'].max(),
                    'best_rank': classification_importance['rank_position'].min(),
                    'avg_rank': classification_importance['rank_position'].mean()
                }

            if len(selection_importance) > 0:
                summary['selection_importance'] = {
                    'avg_importance': selection_importance['importance_score'].mean(),
                    'max_importance': selection_importance['importance_score'].max(),
                    'best_rank': selection_importance['rank_position'].min(),
                    'avg_rank': selection_importance['rank_position'].mean()
                }

            if len(history_df) > 0:
                avg_importance = history_df['importance_score'].mean()
                avg_rank = history_df['rank_position'].mean()

                if avg_importance >= 0.1 or avg_rank <= 10:
                    summary['overall_importance'] = '高重要性'
                elif avg_importance >= 0.05 or avg_rank <= 50:
                    summary['overall_importance'] = '中等重要性'
                else:
                    summary['overall_importance'] = '低重要性'

            summary['recent_results'] = history_df.head(5).to_dict('records')

            return summary

        except Exception as e:
            logger.error(f"获取特征重要性汇总失败: {e}")
            return {'has_importance_data': False, 'error': str(e)}

    def get_feature_experiment_report(
        self,
        fxdef_id: Optional[int] = None,
        feature_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Get a complete report of feature experiments.

        Args:
            fxdef_id (Optional[int]): Feature definition ID.
            feature_name (Optional[str]): Feature name.

        Returns:
            Dict[str, Any]: Complete feature experiment report.
        """
        try:
            summary = self.result_manager.get_feature_experiment_summary(
                fxdef_id=fxdef_id,
                feature_name=feature_name
            )

            correlation_summaries = {}
            for target_var in ['age', 'sex', 'age_days']:
                correlation_summaries[target_var] = self.get_feature_correlation_summary(
                    fxdef_id=fxdef_id,
                    feature_name=feature_name,
                    target_variable=target_var
                )

            importance_summaries = {}
            for target_var in ['age', 'sex', 'age_group']:
                importance_summaries[target_var] = self.get_feature_importance_summary(
                    fxdef_id=fxdef_id,
                    feature_name=feature_name,
                    target_variable=target_var
                )

            overall_assessment = self._generate_overall_assessment(
                correlation_summaries, importance_summaries
            )

            return {
                'feature_info': {
                    'fxdef_id': fxdef_id,
                    'feature_name': feature_name
                },
                'correlation_summaries': correlation_summaries,
                'importance_summaries': importance_summaries,
                'overall_assessment': overall_assessment,
                'experiment_history': summary.get('recent_history', [])
            }

        except Exception as e:
            logger.error(f"获取特征实验报告失败: {e}")
            return {'error': str(e)}

    def _generate_overall_assessment(
        self,
        correlation_summaries: Dict[str, Any],
        importance_summaries: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Generate an overall assessment based on correlation and importance summaries.

        Args:
            correlation_summaries (Dict[str, Any]): Correlation summary dictionary.
            importance_summaries (Dict[str, Any]): Importance summary dictionary.

        Returns:
            Dict[str, Any]: Overall assessment result.
        """
        assessment = {
            'has_age_correlation': False,
            'has_sex_correlation': False,
            'has_importance_data': False,
            'recommendations': []
        }

        if 'age' in correlation_summaries and correlation_summaries['age']['has_correlation_data']:
            assessment['has_age_correlation'] = True
            age_corr = correlation_summaries['age']

            if age_corr['significant_ratio'] >= 0.5:
                assessment['recommendations'].append(
                    f"该特征与年龄有稳定的相关性 ({age_corr['correlation_strength']}, {age_corr['correlation_direction']})"
                )
            elif age_corr['significant_ratio'] >= 0.2:
                assessment['recommendations'].append(
                    f"该特征与年龄有一定相关性，建议进一步验证"
                )

        if 'sex' in correlation_summaries and correlation_summaries['sex']['has_correlation_data']:
            assessment['has_sex_correlation'] = True
            sex_corr = correlation_summaries['sex']

            if sex_corr['significant_ratio'] >= 0.5:
                assessment['recommendations'].append(
                    f"该特征与性别有稳定的相关性 ({sex_corr['correlation_strength']})"
                )

        for target_var, importance_summary in importance_summaries.items():
            if importance_summary['has_importance_data']:
                assessment['has_importance_data'] = True
                if importance_summary.get('overall_importance') == '高重要性':
                    assessment['recommendations'].append(
                        f"该特征在{target_var}分类任务中具有高重要性"
                    )

        if len(assessment['recommendations']) == 0:
            assessment['recommendations'].append(
                "该特征在现有实验中未显示出显著的模式，建议进行更多实验验证"
            )

        return assessment

    def search_features_by_correlation(
        self,
        target_variable: str = 'age',
        min_correlation: float = 0.3,
        significant_only: bool = True
    ) -> pd.DataFrame:
        """
        Search features by correlation with a target variable.

        Args:
            target_variable (str): Target variable.
            min_correlation (float): Minimum correlation coefficient.
            significant_only (bool): Only return significant results.

        Returns:
            pd.DataFrame: List of features meeting the criteria.
        """
        try:
            history_df = self.result_manager.get_feature_correlation_history(
                target_variable=target_variable,
                min_correlation=min_correlation,
                significant_only=significant_only
            )

            if len(history_df) == 0:
                return pd.DataFrame()

            # Group by feature and calculate mean correlation
            feature_summary = history_df.groupby(['fxdef_id', 'feature_name']).agg({
                'correlation_coefficient': ['mean', 'max', 'count'],
                'rank_position': ['mean', 'min'],
                'significance_level': lambda x: (x != 'ns').sum()
            }).reset_index()

            feature_summary.columns = [
                'fxdef_id', 'feature_name', 'avg_correlation', 'max_correlation',
                'experiment_count', 'avg_rank', 'best_rank', 'significant_count'
            ]

            feature_summary['significant_ratio'] = feature_summary['significant_count'] / feature_summary['experiment_count']

            feature_summary = feature_summary.sort_values('avg_correlation', key=abs, ascending=False)

            return feature_summary

        except Exception as e:
            logger.error(f"搜索特征失败: {e}")
            return pd.DataFrame()

    def search_features_by_importance(
        self,
        target_variable: str = 'age',
        min_importance: float = 0.05
    ) -> pd.DataFrame:
        """
        Search features by importance score for a target variable.

        Args:
            target_variable (str): Target variable.
            min_importance (float): Minimum importance score.

        Returns:
            pd.DataFrame: List of features meeting the criteria.
        """
        try:
            history_df = self.result_manager.get_feature_importance_history(
                target_variable=target_variable
            )

            if len(history_df) == 0:
                return pd.DataFrame()

            history_df = history_df[history_df['importance_score'] >= min_importance]

            if len(history_df) == 0:
                return pd.DataFrame()

            # Group by feature and calculate mean importance
            feature_summary = history_df.groupby(['fxdef_id', 'feature_name', 'result_type']).agg({
                'importance_score': ['mean', 'max', 'count'],
                'rank_position': ['mean', 'min']
            }).reset_index()

            feature_summary.columns = [
                'fxdef_id', 'feature_name', 'result_type', 'avg_importance',
                'max_importance', 'experiment_count', 'avg_rank', 'best_rank'
            ]

            feature_summary = feature_summary.sort_values('avg_importance', ascending=False)

            return feature_summary

        except Exception as e:
            logger.error(f"搜索特征失败: {e}")
            return pd.DataFrame()


def print_feature_report(feature_name: str, db_path: str = "database/eeg2go.db") -> None:
    """
    Print the feature experiment report.

    Args:
        feature_name (str): Feature name.
        db_path (str): Database path.
    """
    query_tool = FeatureExperimentQuery(db_path)
    report = query_tool.get_feature_experiment_report(feature_name=feature_name)

    if 'error' in report:
        print(f"错误: {report['error']}")
        return

    print("=" * 60)
    print(f"特征实验报告: {feature_name}")
    print("=" * 60)

    print("\n📊 相关性分析结果:")
    for target_var, corr_summary in report['correlation_summaries'].items():
        if corr_summary['has_correlation_data']:
            print(f"  {target_var}:")
            print(f"    平均相关系数: {corr_summary['avg_correlation']:.3f}")
            print(f"    相关性强度: {corr_summary['correlation_strength']}")
            print(f"    相关性方向: {corr_summary['correlation_direction']}")
            print(f"    显著性比例: {corr_summary['significant_ratio']:.1%}")
            print(f"    最佳排名: {corr_summary['best_rank']}")
        else:
            print(f"  {target_var}: 无数据")

    print("\n🎯 重要性分析结果:")
    for target_var, imp_summary in report['importance_summaries'].items():
        if imp_summary['has_importance_data']:
            print(f"  {target_var}:")
            if 'classification_importance' in imp_summary:
                cls_imp = imp_summary['classification_importance']
                print(f"    分类重要性: {cls_imp['avg_importance']:.3f} (最佳排名: {cls_imp['best_rank']})")
            if 'selection_importance' in imp_summary:
                sel_imp = imp_summary['selection_importance']
                print(f"    选择重要性: {sel_imp['avg_importance']:.3f} (最佳排名: {sel_imp['best_rank']})")
            print(f"    综合重要性: {imp_summary.get('overall_importance', '未知')}")
        else:
            print(f"  {target_var}: 无数据")

    print("\n📋 综合评估:")
    assessment = report['overall_assessment']
    for recommendation in assessment['recommendations']:
        print(f"  • {recommendation}")

    if report['experiment_history']:
        print("\n🕒 最近实验历史:")
        for exp in report['experiment_history'][:3]:
            print(f"  {exp['run_time']}: {exp['experiment_type']} - {exp['target_variable']}")
            if 'metric_value' in exp:
                print(f"    结果: {exp['metric_value']:.3f} (排名: {exp['rank_position']})")


if __name__ == "__main__":
    query_tool = FeatureExperimentQuery()

    print("搜索与年龄相关的特征...")
    age_features = query_tool.search_features_by_correlation('age', min_correlation=0.2)
    if len(age_features) > 0:
        print(f"找到 {len(age_features)} 个与年龄相关的特征:")
        print(age_features.head())

    print("\n生成特征报告...")
    print_feature_report("fx20_bp_alpha_O1_mean") 