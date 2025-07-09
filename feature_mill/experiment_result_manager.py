"""
实验结果管理器

这个模块负责：
1. 存储实验运行记录
2. 存储特征级别的详细结果
3. 提供查询接口，支持按特征、实验类型、目标变量等条件查询
4. 生成特征实验历史报告
"""

import os
import json
import sqlite3
import pandas as pd
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
from logging_config import logger  # 使用全局logger


class ExperimentResultManager:
    """实验结果管理器"""
    
    def __init__(self, db_path: str = "database/eeg2go.db"):
        self.db_path = db_path
        self._init_database()
    
    def _init_database(self):
        """初始化数据库表"""
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # 检查表是否已存在
            c.execute("SELECT name FROM sqlite_master WHERE type='table' AND name='experiment_results'")
            if c.fetchone() is None:
                # 表不存在，执行完整的schema脚本
                schema_path = "database/experiment_schema.sql"
                if os.path.exists(schema_path):
                    with open(schema_path, 'r', encoding='utf-8') as f:
                        schema_sql = f.read()
                    
                    conn.executescript(schema_sql)
                    logger.info("实验管理数据库表初始化完成")
                else:
                    logger.warning(f"数据库模式文件不存在: {schema_path}")
            else:
                # 表已存在，只检查并创建缺失的视图
                self._ensure_views_exist(conn)
                logger.info("实验管理数据库表已存在，视图检查完成")
            
            conn.close()
            
        except Exception as e:
            logger.error(f"初始化数据库失败: {e}")
    
    def _ensure_views_exist(self, conn):
        """确保视图存在，如果不存在则创建"""
        try:
            c = conn.cursor()
            
            # 检查视图是否存在
            c.execute("SELECT name FROM sqlite_master WHERE type='view' AND name='feature_experiment_summary'")
            if c.fetchone() is None:
                # 创建视图
                views_sql = """
                -- 创建视图：特征实验结果汇总
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

                -- 创建视图：特征相关性历史记录
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

                -- 创建视图：特征重要性历史记录
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
                logger.info("实验管理视图创建完成")
            
        except Exception as e:
            logger.error(f"创建视图失败: {e}")
    
    def save_experiment_result(self, 
                             experiment_type: str,
                             dataset_id: int,
                             feature_set_id: int,
                             parameters: Dict,
                             output_dir: str,
                             summary: str,
                             duration: float,
                             experiment_name: str = None) -> int:
        """
        保存实验运行记录
        
        Returns:
            int: 实验结果ID
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
            
            logger.info(f"实验记录已保存，ID: {experiment_result_id}")
            return experiment_result_id
            
        except Exception as e:
            logger.error(f"保存实验记录失败: {e}")
            raise
    
    def save_correlation_results(self, 
                               experiment_result_id: int,
                               correlation_results: Dict,
                               target_vars: List[str]) -> None:
        """
        保存相关性分析结果
        
        Args:
            experiment_result_id: 实验结果ID
            correlation_results: 相关性分析结果字典
            target_vars: 目标变量列表
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
                
                # 保存每个特征的相关性结果
                for _, row in result['top_results'].iterrows():
                    feature_name = row['feature']
                    correlation = row['correlation']
                    p_value = row['p_value']
                    significance = row.get('significance_level', 'ns')
                    rank_pos = row.get('rank_position', 0)
                    
                    # 提取fxdef_id（从特征名称中解析）
                    fxdef_id = self._extract_fxdef_id(feature_name)
                    
                    # 保存相关系数
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
                    
                    # 保存p值
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
            logger.info(f"相关性分析结果已保存，实验ID: {experiment_result_id}")
            
        except Exception as e:
            logger.error(f"保存相关性分析结果失败: {e}")
            raise
    
    def save_classification_results(self,
                                  experiment_result_id: int,
                                  classification_results: Dict,
                                  target_var: str,
                                  feature_names: List[str]) -> None:
        """
        保存分类分析结果
        
        Args:
            experiment_result_id: 实验结果ID
            classification_results: 分类分析结果字典
            target_var: 目标变量
            feature_names: 特征名称列表
        """
        try:
            conn = sqlite3.connect(self.db_path)
            c = conn.cursor()
            
            # 保存模型性能指标
            for model_name, result in classification_results.items():
                # 保存整体性能指标
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
                
                # 保存特征重要性
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
            logger.info(f"分类分析结果已保存，实验ID: {experiment_result_id}")
            
        except Exception as e:
            logger.error(f"保存分类分析结果失败: {e}")
            raise
    
    def save_feature_selection_results(self,
                                     experiment_result_id: int,
                                     selection_results: Dict,
                                     target_var: str) -> None:
        """
        保存特征选择结果
        
        Args:
            experiment_result_id: 实验结果ID
            selection_results: 特征选择结果字典
            target_var: 目标变量
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
            logger.info(f"特征选择结果已保存，实验ID: {experiment_result_id}")
            
        except Exception as e:
            logger.error(f"保存特征选择结果失败: {e}")
            raise
    
    def get_feature_correlation_history(self, 
                                      fxdef_id: Optional[int] = None,
                                      feature_name: Optional[str] = None,
                                      target_variable: Optional[str] = None,
                                      min_correlation: Optional[float] = None,
                                      significant_only: bool = False) -> pd.DataFrame:
        """
        获取特征相关性历史记录
        
        Args:
            fxdef_id: 特征定义ID
            feature_name: 特征名称
            target_variable: 目标变量
            min_correlation: 最小相关系数
            significant_only: 只返回显著相关的结果
        
        Returns:
            pd.DataFrame: 相关性历史记录
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
            logger.error(f"获取特征相关性历史记录失败: {e}")
            return pd.DataFrame()
    
    def get_feature_importance_history(self,
                                     fxdef_id: Optional[int] = None,
                                     feature_name: Optional[str] = None,
                                     target_variable: Optional[str] = None,
                                     result_type: Optional[str] = None) -> pd.DataFrame:
        """
        获取特征重要性历史记录
        
        Args:
            fxdef_id: 特征定义ID
            feature_name: 特征名称
            target_variable: 目标变量
            result_type: 结果类型（'classification_importance', 'selection_score'）
        
        Returns:
            pd.DataFrame: 重要性历史记录
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
            logger.error(f"获取特征重要性历史记录失败: {e}")
            return pd.DataFrame()
    
    def get_feature_experiment_summary(self, 
                                     fxdef_id: Optional[int] = None,
                                     feature_name: Optional[str] = None) -> Dict:
        """
        获取特征实验汇总信息
        
        Args:
            fxdef_id: 特征定义ID
            feature_name: 特征名称
        
        Returns:
            Dict: 特征实验汇总信息
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 基础查询条件
            where_conditions = []
            params = []
            
            if fxdef_id is not None:
                where_conditions.append("fxdef_id = ?")
                params.append(fxdef_id)
            
            if feature_name is not None:
                where_conditions.append("feature_name LIKE ?")
                params.append(f"%{feature_name}%")
            
            where_clause = " AND ".join(where_conditions) if where_conditions else "1=1"
            
            # 获取相关性分析结果
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
            
            # 获取重要性分析结果
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
            
            # 获取实验历史
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
            logger.error(f"获取特征实验汇总失败: {e}")
            return {}
    
    def _extract_fxdef_id(self, feature_name: str) -> Optional[int]:
        """
        从特征名称中提取fxdef_id
        
        Args:
            feature_name: 特征名称（如 'fx20_bp_alpha_O1_mean'）
        
        Returns:
            Optional[int]: fxdef_id，如果无法提取则返回None
        """
        try:
            # 特征名称格式：fx{id}_{shortname}_{channels}_{stat}
            if feature_name.startswith('fx'):
                parts = feature_name.split('_')
                if len(parts) >= 2:
                    fxdef_id_str = parts[0][2:]  # 去掉'fx'前缀
                    return int(fxdef_id_str)
        except (ValueError, IndexError):
            pass
        
        return None
    
    def get_experiment_statistics(self) -> Dict:
        """
        获取实验统计信息
        
        Returns:
            Dict: 实验统计信息
        """
        try:
            conn = sqlite3.connect(self.db_path)
            
            # 总体统计
            total_experiments = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM experiment_results", conn
            ).iloc[0]['count']
            
            # 按类型统计
            experiments_by_type = pd.read_sql_query("""
                SELECT experiment_type, COUNT(*) as count
                FROM experiment_results
                GROUP BY experiment_type
            """, conn)
            
            # 特征结果统计
            total_feature_results = pd.read_sql_query(
                "SELECT COUNT(*) as count FROM experiment_feature_results", conn
            ).iloc[0]['count']
            
            # 相关性分析统计
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
            logger.error(f"获取实验统计信息失败: {e}")
            return {} 