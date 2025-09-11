import os
import json
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from logging_config import logger  # Use global logger


class ExperimentResultManager:
    """Experiment Result Manager"""

    def __init__(self, db_path: str = "database/eeg2go.db") -> None:
        """
        Initialize the ExperimentResultManager.

        Args:
            db_path (str): Path to the SQLite database file.
        """
        self.db_path = db_path
        self._init_database()

    def _init_database(self) -> None:
        """
        Initialize the database tables and views if they do not exist.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='experiment_results'")
            if c.fetchone() is None:
                schema_path = "database/experiment_schema.sql"
                if os.path.exists(schema_path):
                    with open(schema_path, 'r', encoding='utf-8') as f:
                        schema_sql = f.read()
                    conn.executescript(schema_sql)
                    logger.info("Experiment management database tables initialized.")
                else:
                    logger.warning(f"Database schema file does not exist: {schema_path}")
            else:
                self._ensure_views_exist(conn)
                logger.info("Experiment management database tables exist, views checked.")
            conn.close()
        except Exception as e:
            logger.error(f"Failed to initialize database: {e}")

    def _ensure_views_exist(self, conn) -> None:
        """
        Ensure required views exist in the database, create if missing.

        Args:
            conn: SQLite connection object.
        """
        try:
            c = conn.cursor()
            c.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='feature_experiment_summary'")
            if c.fetchone() is None:
                # Create views for experiment summaries
                views_sql = """
                -- View: Feature Experiment Summary
                CREATE VIEW feature_experiment_summary AS
                SELECT 
                    efr.fxdef_id,
                    efr.feature_name,
                    efr.target_variable,
                    efr.result_type,
                    efr.metric_name,
                    efr.metric_value,
                    efr.significance_level,
                    efr.rank_position,
                    er.experiment_type,
                    er.dataset_id,
                    er.feature_set_id,
                    er.run_time,
                    er.parameters,
                    fd.shortname as feature_shortname,
                    fd.chans as feature_channels
                FROM experiment_feature_results efr
                JOIN experiment_results er ON efr.experiment_result_id = er.id
                LEFT JOIN fxdef fd ON efr.fxdef_id = fd.id;

                -- View: Feature Correlation History
                CREATE VIEW feature_correlation_history AS
                SELECT 
                    efr.fxdef_id,
                    efr.feature_name,
                    efr.target_variable,
                    efr.metric_value as correlation_coefficient,
                    efr.significance_level,
                    efr.rank_position,
                    er.dataset_id,
                    er.feature_set_id,
                    er.run_time,
                    er.parameters
                FROM experiment_feature_results efr
                JOIN experiment_results er ON efr.experiment_result_id = er.id
                WHERE efr.result_type = 'correlation' AND efr.metric_name = 'correlation_coefficient'
                ORDER BY er.run_time DESC;

                -- View: Feature Importance History
                CREATE VIEW feature_importance_history AS
                SELECT 
                    efr.fxdef_id,
                    efr.feature_name,
                    efr.target_variable,
                    efr.metric_value as importance_score,
                    efr.rank_position,
                    er.dataset_id,
                    er.feature_set_id,
                    er.run_time,
                    er.parameters
                FROM experiment_feature_results efr
                JOIN experiment_results er ON efr.experiment_result_id = er.id
                WHERE efr.result_type IN ('classification_importance', 'selection_score')
                ORDER BY er.run_time DESC;
                """
                conn.executescript(views_sql)
                logger.info("Experiment management views created.")
        except Exception as e:
            logger.error(f"Failed to create views: {e}")

    def save_experiment_result(
        self,
        experiment_type: str,
        dataset_id: int,
        feature_set_id: int,
        parameters: Dict,
        output_dir: str,
        summary: str,
        duration: float,
        experiment_name: str = None
    ) -> int:
        """
        Save an experiment run record.

        Args:
            experiment_type (str): Type of the experiment.
            dataset_id (int): Dataset identifier.
            feature_set_id (int): Feature set identifier.
            parameters (Dict): Experiment parameters.
            output_dir (str): Output directory.
            summary (str): Experiment summary.
            duration (float): Duration in seconds.
            experiment_name (str, optional): Name of the experiment.

        Returns:
            int: Experiment result ID.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            c.execute("""
                INSERT INTO experiment_results (
                    experiment_type, experiment_name, dataset_id, feature_set_id,
                    parameters, output_dir, summary, duration_seconds
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                experiment_type,
                experiment_name or experiment_type,
                dataset_id,
                feature_set_id,
                json.dumps(parameters, ensure_ascii=False),
                output_dir,
                summary,
                duration
            ))
            experiment_result_id = c.lastrowid
            conn.commit()
            conn.close()
            logger.info(f"Experiment record saved, ID: {experiment_result_id}")
            return experiment_result_id
        except Exception as e:
            logger.error(f"Failed to save experiment record: {e}")
            raise

    def save_correlation_results(
        self,
        experiment_result_id: int,
        correlation_results: Dict,
        target_vars: List[str]
    ) -> None:
        """
        Save correlation analysis results.

        Args:
            experiment_result_id (int): Experiment result ID.
            correlation_results (Dict): Correlation analysis results.
            target_vars (List[str]): List of target variables.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            for target_var in target_vars:
                if target_var not in correlation_results:
                    continue
                result = correlation_results[target_var]
                if 'top_results' not in result or len(result['top_results']) == 0:
                    continue
                for _, row in result['top_results'].iterrows():
                    feature_name = row['feature']
                    correlation = row['correlation']
                    p_value = row['p_value']
                    significance = row.get('significance_level', 'ns')
                    rank_pos = row.get('rank_position', 0)
                    fxdef_id = self._extract_fxdef_id(feature_name)
                    # Save correlation coefficient
                    c.execute("""
                        INSERT INTO experiment_feature_results (
                            experiment_result_id, fxdef_id, feature_name, target_variable,
                            result_type, metric_name, metric_value, metric_unit,
                            significance_level, rank_position
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        experiment_result_id, fxdef_id, feature_name, target_var,
                        'correlation', 'correlation_coefficient', correlation, 'correlation',
                        significance, rank_pos
                    ))
                    # Save p-value
                    c.execute("""
                        INSERT INTO experiment_feature_results (
                            experiment_result_id, fxdef_id, feature_name, target_variable,
                            result_type, metric_name, metric_value, metric_unit,
                            significance_level, rank_position
                        ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                    """, (
                        experiment_result_id, fxdef_id, feature_name, target_var,
                        'correlation', 'p_value', p_value, 'probability',
                        significance, rank_pos
                    ))
            conn.commit()
            conn.close()
            logger.info(f"Correlation analysis results saved, experiment ID: {experiment_result_id}")
        except Exception as e:
            logger.error(f"Failed to save correlation analysis results: {e}")
            raise

    def save_classification_results(
        self,
        experiment_result_id: int,
        classification_results: Dict,
        target_var: str,
        feature_names: List[str]
    ) -> None:
        """
        Save classification analysis results.

        Args:
            experiment_result_id (int): Experiment result ID.
            classification_results (Dict): Classification analysis results.
            target_var (str): Target variable.
            feature_names (List[str]): List of feature names.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            for model_name, result in classification_results.items():
                # Save overall performance metrics
                c.execute("""
                    INSERT INTO experiment_metadata (
                        experiment_result_id, key, value, value_type
                    ) VALUES (?, ?, ?, ?)
                """, (
                    experiment_result_id,
                    f"{model_name}_accuracy",
                    str(result['accuracy']),
                    'number'
                ))
                c.execute("""
                    INSERT INTO experiment_metadata (
                        experiment_result_id, key, value, value_type
                    ) VALUES (?, ?, ?, ?)
                """, (
                    experiment_result_id,
                    f"{model_name}_f1_score",
                    str(result['f1_score']),
                    'number'
                ))
                # Save feature importance
                if result.get('feature_importance') is not None:
                    importance_scores = result['feature_importance']
                    for i, (feature_name, importance) in enumerate(zip(feature_names, importance_scores)):
                        fxdef_id = self._extract_fxdef_id(feature_name)
                        c.execute("""
                            INSERT INTO experiment_feature_results (
                                experiment_result_id, fxdef_id, feature_name, target_variable,
                                result_type, metric_name, metric_value, metric_unit,
                                rank_position
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            experiment_result_id, fxdef_id, feature_name, target_var,
                            'classification_importance', 'importance_score', importance, 'score',
                            i + 1
                        ))
            conn.commit()
            conn.close()
            logger.info(f"Classification analysis results saved, experiment ID: {experiment_result_id}")
        except Exception as e:
            logger.error(f"Failed to save classification analysis results: {e}")
            raise

    def save_feature_selection_results(
        self,
        experiment_result_id: int,
        selection_results: Dict,
        target_var: str
    ) -> None:
        """
        Save feature selection results.

        Args:
            experiment_result_id (int): Experiment result ID.
            selection_results (Dict): Feature selection results.
            target_var (str): Target variable.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            for method_name, result in selection_results.items():
                if 'selected_features' in result and 'scores' in result:
                    selected_features = result['selected_features']
                    scores = result['scores']
                    for i, (feature_name, score) in enumerate(zip(selected_features, scores)):
                        fxdef_id = self._extract_fxdef_id(feature_name)
                        c.execute("""
                            INSERT INTO experiment_feature_results (
                                experiment_result_id, fxdef_id, feature_name, target_variable,
                                result_type, metric_name, metric_value, metric_unit,
                                rank_position, additional_data
                            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                        """, (
                            experiment_result_id, fxdef_id, feature_name, target_var,
                            'selection_score', f'{method_name}_score', score, 'score',
                            i + 1, json.dumps({'method': method_name})
                        ))
            conn.commit()
            conn.close()
            logger.info(f"Feature selection results saved, experiment ID: {experiment_result_id}")
        except Exception as e:
            logger.error(f"Failed to save feature selection results: {e}")
            raise

    def get_feature_correlation_history(
        self,
        fxdef_id: Optional[int] = None,
        feature_name: Optional[str] = None,
        target_variable: Optional[str] = None,
        min_correlation: Optional[float] = None,
        significant_only: bool = False
    ) -> pd.DataFrame:
        """
        Get feature correlation history.

        Args:
            fxdef_id (Optional[int]): Feature definition ID.
            feature_name (Optional[str]): Feature name.
            target_variable (Optional[str]): Target variable.
            min_correlation (Optional[float]): Minimum correlation coefficient.
            significant_only (bool): Only return significant results.

        Returns:
            pd.DataFrame: Correlation history records.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT 
                    efr.fxdef_id,
                    efr.feature_name,
                    efr.target_variable,
                    efr.metric_value as correlation_coefficient,
                    efr.significance_level,
                    efr.rank_position,
                    er.dataset_id,
                    er.feature_set_id,
                    er.run_time,
                    er.parameters
                FROM experiment_feature_results efr
                JOIN experiment_results er ON efr.experiment_result_id = er.id
                WHERE efr.result_type = 'correlation' 
                AND efr.metric_name = 'correlation_coefficient'
            """
            params = []
            if fxdef_id is not None:
                query += " AND efr.fxdef_id = ?"
                params.append(fxdef_id)
            if feature_name is not None:
                query += " AND efr.feature_name LIKE ?"
                params.append(f"%{feature_name}%")
            if target_variable is not None:
                query += " AND efr.target_variable = ?"
                params.append(target_variable)
            if min_correlation is not None:
                query += " AND ABS(efr.metric_value) >= ?"
                params.append(min_correlation)
            if significant_only:
                query += " AND efr.significance_level != 'ns'"
            query += " ORDER BY er.run_time DESC"
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Failed to get feature correlation history: {e}")
            return pd.DataFrame()

    def get_feature_importance_history(
        self,
        fxdef_id: Optional[int] = None,
        feature_name: Optional[str] = None,
        target_variable: Optional[str] = None,
        result_type: Optional[str] = None
    ) -> pd.DataFrame:
        """
        Get feature importance history.

        Args:
            fxdef_id (Optional[int]): Feature definition ID.
            feature_name (Optional[str]): Feature name.
            target_variable (Optional[str]): Target variable.
            result_type (Optional[str]): Result type ('classification_importance', 'selection_score').

        Returns:
            pd.DataFrame: Importance history records.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            query = """
                SELECT 
                    efr.fxdef_id,
                    efr.feature_name,
                    efr.target_variable,
                    efr.result_type,
                    efr.metric_value as importance_score,
                    efr.rank_position,
                    er.dataset_id,
                    er.feature_set_id,
                    er.run_time,
                    er.parameters
                FROM experiment_feature_results efr
                JOIN experiment_results er ON efr.experiment_result_id = er.id
                WHERE efr.result_type IN ('classification_importance', 'selection_score')
                AND (efr.metric_name LIKE '%importance%' OR efr.metric_name LIKE '%score%')
            """
            params = []
            if fxdef_id is not None:
                query += " AND efr.fxdef_id = ?"
                params.append(fxdef_id)
            if feature_name is not None:
                query += " AND efr.feature_name LIKE ?"
                params.append(f"%{feature_name}%")
            if target_variable is not None:
                query += " AND efr.target_variable = ?"
                params.append(target_variable)
            if result_type is not None:
                query += " AND efr.result_type = ?"
                params.append(result_type)
            query += " ORDER BY er.run_time DESC"
            df = pd.read_sql_query(query, conn, params=params)
            conn.close()
            return df
        except Exception as e:
            logger.error(f"Failed to get feature importance history: {e}")
            return pd.DataFrame()

    def get_feature_experiment_summary(
        self,
        fxdef_id: Optional[int] = None,
        feature_name: Optional[str] = None
    ) -> Dict:
        """
        Get feature experiment summary.

        Args:
            fxdef_id (Optional[int]): Feature definition ID.
            feature_name (Optional[str]): Feature name.

        Returns:
            Dict: Feature experiment summary.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            where_conditions = []
            params = []
            if fxdef_id is not None:
                where_conditions.append("fxdef_id = ?")
                params.append(fxdef_id)
            if feature_name is not None:
                where_conditions.append("feature_name LIKE ?")
                params.append(f"%{feature_name}%")
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            # Correlation summary
            correlation_query = f"""
                SELECT 
                    target_variable,
                    COUNT(*) as experiment_count,
                    AVG(ABS(metric_value)) as avg_correlation,
                    MAX(ABS(metric_value)) as max_correlation,
                    COUNT(CASE WHEN significance_level != 'ns' THEN 1 END) as significant_count
                FROM experiment_feature_results 
                WHERE result_type = 'correlation' 
                AND metric_name = 'correlation_coefficient'
                AND {where_clause}
                GROUP BY target_variable
            """
            correlation_df = pd.read_sql_query(correlation_query, conn, params=params)
            # Importance summary
            importance_query = f"""
                SELECT 
                    target_variable,
                    result_type,
                    COUNT(*) as experiment_count,
                    AVG(metric_value) as avg_importance,
                    MAX(metric_value) as max_importance,
                    AVG(rank_position) as avg_rank
                FROM experiment_feature_results 
                WHERE result_type IN ('classification_importance', 'selection_score')
                AND {where_clause}
                GROUP BY target_variable, result_type
            """
            importance_df = pd.read_sql_query(importance_query, conn, params=params)
            # Recent experiment history
            history_query = f"""
                SELECT 
                    experiment_type,
                    target_variable,
                    run_time,
                    metric_value,
                    significance_level,
                    rank_position
                FROM experiment_feature_results efr
                JOIN experiment_results er ON efr.experiment_result_id = er.id
                WHERE {where_clause}
                ORDER BY run_time DESC
                LIMIT 10
            """
            history_df = pd.read_sql_query(history_query, conn, params=params)
            conn.close()
            return {
                'correlation_summary': correlation_df.to_dict('records'),
                'importance_summary': importance_df.to_dict('records'),
                'recent_history': history_df.to_dict('records')
            }
        except Exception as e:
            logger.error(f"Failed to get feature experiment summary: {e}")
            return {}

    def _extract_fxdef_id(self, feature_name: str) -> Optional[int]:
        """
        Extract fxdef_id from feature name.

        Args:
            feature_name (str): Feature name (e.g., 'fx20_bp_alpha_O1_mean').

        Returns:
            Optional[int]: fxdef_id, or None if extraction fails.
        """
        try:
            # Feature name format: fx{id}_{shortname}_{channels}_{stat}
            if feature_name.startswith('fx'):
                parts = feature_name.split('_')
                if len(parts) >= 2:
                    fxdef_id_str = parts[0][2:]  # Remove 'fx' prefix
                    return int(fxdef_id_str)
        except (ValueError, IndexError):
            pass
        return None

    def get_experiment_statistics(self) -> Dict:
        """
        Get experiment statistics.

        Returns:
            Dict: Experiment statistics.
        """
        try:
            conn = sqlite3.connect(self.db_path)
            total_experiments = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM experiment_results", conn
            ).iloc[0]['count']
            experiments_by_type = pd.read_sql_query("""
                SELECT experiment_type, COUNT(*) as count
                FROM experiment_results
                GROUP BY experiment_type
            """, conn)
            total_feature_results = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM experiment_feature_results", conn
            ).iloc[0]['count']
            correlation_stats = pd.read_sql_query("""
                SELECT 
                    COUNT(*) as total_correlations,
                    COUNT(CASE WHEN significance_level != 'ns' THEN 1 END) as significant_correlations,
                    AVG(ABS(metric_value)) as avg_correlation
                FROM experiment_feature_results 
                WHERE result_type = 'correlation' 
                AND metric_name = 'correlation_coefficient'
            """, conn)
            conn.close()
            return {
                'total_experiments': total_experiments,
                'experiments_by_type': experiments_by_type.to_dict('records'),
                'total_feature_results': total_feature_results,
                'correlation_stats': correlation_stats.to_dict('records')[0] if len(correlation_stats) > 0 else {}
            }
        except Exception as e:
            logger.error(f"Failed to get experiment statistics: {e}")
            return {}