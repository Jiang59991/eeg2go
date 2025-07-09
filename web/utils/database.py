"""
数据库连接工具模块
提供统一的数据库连接和查询接口
"""

import sqlite3
import os
from contextlib import contextmanager
from typing import Optional, List, Dict, Any
from ..config import DATABASE_PATH

class DatabaseManager:
    """数据库管理器"""
    
    def __init__(self, db_path: Optional[str] = None):
        self.db_path = db_path or DATABASE_PATH
    
    @contextmanager
    def get_connection(self):
        """获取数据库连接的上下文管理器"""
        conn = None
        try:
            conn = sqlite3.connect(self.db_path)
            conn.row_factory = sqlite3.Row
            yield conn
        except Exception as e:
            if conn:
                conn.rollback()
            raise e
        finally:
            if conn:
                conn.close()
    
    def execute_query(self, query: str, params: tuple = ()) -> List[Dict[str, Any]]:
        """执行查询并返回结果"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            return [dict(row) for row in cursor.fetchall()]
    
    def execute_single(self, query: str, params: tuple = ()) -> Optional[Dict[str, Any]]:
        """执行查询并返回单条结果"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            row = cursor.fetchone()
            return dict(row) if row else None
    
    def execute_update(self, query: str, params: tuple = ()) -> int:
        """执行更新操作并返回影响的行数"""
        with self.get_connection() as conn:
            cursor = conn.cursor()
            cursor.execute(query, params)
            conn.commit()
            return cursor.rowcount

# 全局数据库管理器实例
db_manager = DatabaseManager()

def get_db_connection():
    """获取数据库连接（兼容旧代码）"""
    return db_manager.get_connection()

def get_datasets() -> List[Dict[str, Any]]:
    """获取所有数据集"""
    return db_manager.execute_query('SELECT * FROM datasets')

def get_recordings(dataset_id: Optional[int] = None) -> List[Dict[str, Any]]:
    """获取录音文件列表"""
    if dataset_id:
        return db_manager.execute_query('''
            SELECT r.*, s.age, s.sex 
            FROM recordings r 
            LEFT JOIN subjects s ON r.subject_id = s.subject_id 
            WHERE r.dataset_id = ?
        ''', (dataset_id,))
    else:
        return db_manager.execute_query('''
            SELECT r.*, s.age, s.sex 
            FROM recordings r 
            LEFT JOIN subjects s ON r.subject_id = s.subject_id
        ''')

def get_feature_sets() -> List[Dict[str, Any]]:
    """获取特征集列表"""
    return db_manager.execute_query('SELECT * FROM feature_sets')

def get_feature_set_details(feature_set_id: int) -> Optional[Dict[str, Any]]:
    """获取特征集详细信息"""
    feature_set = db_manager.execute_single(
        'SELECT * FROM feature_sets WHERE id = ?', 
        (feature_set_id,)
    )
    
    if not feature_set:
        return None
    
    features = db_manager.execute_query('''
        SELECT f.*, p.shortname as pipeline_name, p.description as pipeline_desc
        FROM fxdef f
        JOIN feature_set_items fsi ON f.id = fsi.fxdef_id
        LEFT JOIN pipedef p ON f.pipedef_id = p.id
        WHERE fsi.feature_set_id = ?
    ''', (feature_set_id,))
    
    return {
        'feature_set': feature_set,
        'features': features
    }

def get_pipelines() -> List[Dict[str, Any]]:
    """获取pipeline列表"""
    return db_manager.execute_query('SELECT * FROM pipedef')

def get_pipeline_details(pipeline_id: int) -> Optional[Dict[str, Any]]:
    """获取pipeline详细信息"""
    return db_manager.execute_single(
        'SELECT * FROM pipedef WHERE id = ?', 
        (pipeline_id,)
    )

def get_fxdefs() -> List[Dict[str, Any]]:
    """获取特征定义列表"""
    return db_manager.execute_query('''
        SELECT f.*, p.shortname as pipeline_name 
        FROM fxdef f 
        LEFT JOIN pipedef p ON f.pipedef_id = p.id
    ''')

def get_fxdef_details(fxdef_id: int) -> Optional[Dict[str, Any]]:
    """获取特征定义详细信息"""
    return db_manager.execute_single('''
        SELECT f.*, p.shortname as pipeline_name, p.description as pipeline_desc
        FROM fxdef f
        LEFT JOIN pipedef p ON f.pipedef_id = p.id
        WHERE f.id = ?
    ''', (fxdef_id,)) 