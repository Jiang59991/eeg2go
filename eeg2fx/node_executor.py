"""
Node Executor - DAG节点执行器
模仿Apache Airflow的设计，专注于节点执行和状态管理
"""

import os
import json
import hashlib
import time
from enum import Enum
from dataclasses import dataclass, field
from typing import Dict, List, Any, Optional, Callable
from collections import defaultdict, deque
from eeg2fx.steps import load_recording
from eeg2fx.feature.common import standardize_channel_name
from eeg2fx.function_registry import PREPROCESSING_FUNCS, FEATURE_FUNCS, UTILITY_FUNCS
from logging_config import logger

class NodeStatus(Enum):
    """节点状态枚举"""
    PENDING = "pending"      # 等待执行
    RUNNING = "running"      # 正在执行
    SUCCESS = "success"      # 执行成功
    FAILED = "failed"        # 执行失败

@dataclass
class NodeExecutionInfo:
    """节点执行信息"""
    node_id: str
    func_name: str
    params: Dict[str, Any]
    status: NodeStatus = NodeStatus.PENDING
    start_time: Optional[float] = None
    end_time: Optional[float] = None
    duration: Optional[float] = None
    error_message: Optional[str] = None
    input_nodes: List[str] = field(default_factory=list)
    output_nodes: List[str] = field(default_factory=list)
    pipeline_count: int = 0
    fxdef_count: int = 0

class NodeExecutor:
    """DAG节点执行器"""
    
    def __init__(self, recording_id: int):
        self.recording_id = recording_id
        self.node_outputs = {}
        self.execution_info = {}
        self.execution_order = []
        
    def resolve_function(self, func_name: str) -> Callable:
        """解析函数"""
        if func_name in PREPROCESSING_FUNCS:
            return PREPROCESSING_FUNCS[func_name]
        if func_name in FEATURE_FUNCS:
            return FEATURE_FUNCS[func_name]
        if func_name in UTILITY_FUNCS:
            return UTILITY_FUNCS[func_name]
        raise ValueError(f"Function '{func_name}' is not registered in function_registry.")
    
    def get_input_from_upstream_paths(self, node: Dict[str, Any], dag: Dict[str, Any]) -> List[Any]:
        """从upstream_paths获取输入"""
        func_name = node["func"]
        input_ids = node["inputnodes"]
        
        if func_name == "raw":
            return []  # raw节点没有输入
        
        # 从input_ids获取所有输入节点的结果
        inputs = []
        for input_id in input_ids:
            # 需要找到input_id对应的upstream_hash
            input_upstream_hash = None
            for other_node_id, other_node in dag.items():
                if other_node_id == input_id:
                    other_upstream_paths = other_node.get("upstream_paths", set())
                    if other_upstream_paths:
                        input_upstream_hash = list(other_upstream_paths)[0][3]
                        break
            
            if input_upstream_hash and input_upstream_hash in self.node_outputs:
                inputs.append(self.node_outputs[input_upstream_hash])
            else:
                logger.warning(f"Input node {input_id} not found in node_outputs")
        
        return inputs
    
    def prepare_node_execution_info(self, node_id: str, node: Dict[str, Any]) -> NodeExecutionInfo:
        """准备节点执行信息"""
        pipeline_paths = node.get("pipeline_paths", set())
        fxdef_ids = node.get("fxdef_ids", [])
        
        return NodeExecutionInfo(
            node_id=node_id,
            func_name=node["func"],
            params=node["params"],
            input_nodes=node["inputnodes"],
            pipeline_count=len(pipeline_paths),
            fxdef_count=len(fxdef_ids)
        )
    
    def execute_node(self, node_id: str, node: Dict[str, Any], dag: Dict[str, Any]) -> Dict[str, Any]:
        """执行单个节点，返回upstream_hash到结果的映射"""
        # 准备执行信息
        exec_info = self.prepare_node_execution_info(node_id, node)
        self.execution_info[node_id] = exec_info
        
        # 执行节点
        exec_info.status = NodeStatus.RUNNING
        exec_info.start_time = time.time()
        
        try:
            # 获取输入
            inputs = self.get_input_from_upstream_paths(node, dag)
            
            # 解析函数
            func = self.resolve_function(node["func"])
            
            # 执行函数
            if node["func"] == "raw":
                result = func(self.recording_id)
            elif node["func"] == "split_channel":
                result = func(*inputs, **node["params"])
            else:
                result = func(*inputs, **node["params"])
            
            # 更新执行信息
            exec_info.end_time = time.time()
            exec_info.duration = exec_info.end_time - exec_info.start_time
            exec_info.status = NodeStatus.SUCCESS
            
            # 为所有upstream_paths存储结果
            upstream_paths = node.get("upstream_paths", set())
            results = {}
            for upstream_path in upstream_paths:
                if len(upstream_path) >= 4:
                    upstream_hash = upstream_path[3]
                    results[upstream_hash] = result
            
            logger.info(f"[EXECUTED] Node {node_id} ({node['func']}) - Duration: {exec_info.duration:.3f}s, Upstream Paths: {len(upstream_paths)}")
            
            return results
            
        except Exception as e:
            # 处理执行失败
            exec_info.end_time = time.time()
            exec_info.duration = exec_info.end_time - exec_info.start_time
            exec_info.status = NodeStatus.FAILED
            exec_info.error_message = str(e)
            
            logger.error(f"[FAILED] Node {node_id} ({node['func']}) - Error: {str(e)}")
            raise
    
    def toposort(self, dag: Dict[str, Any]) -> List[str]:
        """拓扑排序"""
        from eeg2fx.featureset_grouping import toposort
        return toposort(dag)
    
    def execute_dag(self, dag: Dict[str, Any]) -> Dict[str, Any]:
        """执行整个DAG"""
        logger.info(f"开始执行DAG，包含 {len(dag)} 个节点")
        
        # 拓扑排序
        self.execution_order = self.toposort(dag)
        logger.info(f"DAG执行顺序: {self.execution_order}")
        
        # 按顺序执行节点
        for node_id in self.execution_order:
            node = dag[node_id]
            try:
                results = self.execute_node(node_id, node, dag)
                # 将结果存储到node_outputs中（upstream_hash -> result）
                self.node_outputs.update(results)
            except Exception as e:
                # 节点执行失败，记录错误但继续执行其他节点
                logger.error(f"节点 {node_id} 执行失败，跳过后续依赖节点")
                # 可以选择在这里停止执行或继续执行其他独立节点
                break
        
        # 生成执行报告
        self.generate_execution_report()
        
        return self.node_outputs
    
    def generate_execution_report(self) -> Dict[str, Any]:
        """生成执行报告"""
        total_nodes = len(self.execution_info)
        status_counts = defaultdict(int)
        total_duration = 0.0
        
        for exec_info in self.execution_info.values():
            status_counts[exec_info.status.value] += 1
            if exec_info.duration:
                total_duration += exec_info.duration
        
        report = {
            "total_nodes": total_nodes,
            "status_counts": dict(status_counts),
            "total_duration": total_duration,
            "execution_order": self.execution_order,
            "node_details": {
                node_id: {
                    "status": info.status.value,
                    "duration": info.duration,
                    "pipeline_count": info.pipeline_count,
                    "fxdef_count": info.fxdef_count,
                    "error": info.error_message
                }
                for node_id, info in self.execution_info.items()
            }
        }
        
        logger.info(f"执行报告: {json.dumps(report, indent=2)}")
        return report
    
    def get_node_status(self, node_id: str) -> Optional[NodeStatus]:
        """获取节点状态"""
        if node_id in self.execution_info:
            return self.execution_info[node_id].status
        return None
    
    def get_dag_status_summary(self) -> Dict[str, Any]:
        """获取DAG状态摘要"""
        if not self.execution_info:
            return {"status": "not_started"}
        
        status_counts = defaultdict(int)
        for info in self.execution_info.values():
            status_counts[info.status.value] += 1
        
        total_nodes = len(self.execution_info)
        completed_nodes = status_counts.get("success", 0)
        failed_nodes = status_counts.get("failed", 0)
        
        if failed_nodes > 0:
            overall_status = "failed"
        elif completed_nodes == total_nodes:
            overall_status = "success"
        else:
            overall_status = "running"
        
        return {
            "status": overall_status,
            "total_nodes": total_nodes,
            "completed_nodes": completed_nodes,
            "failed_nodes": failed_nodes,
            "status_counts": dict(status_counts)
        }

# 保持向后兼容的函数
def execute_dag_nodes(dag, recording_id, value_cache=None):
    """向后兼容的DAG执行函数"""
    executor = NodeExecutor(recording_id)
    return executor.execute_dag(dag) 