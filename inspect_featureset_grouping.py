#!/usr/bin/env python3
"""
检测featureset_grouping函数
查看每个节点存储的详细信息
"""

import sys
import os
import json
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eeg2fx.featureset_grouping import build_feature_dag, load_fxdefs_for_set
from eeg2fx.featureset_grouping import load_pipeline_structure

def inspect_feature_set(feature_set_id):
    """检测特征集的详细信息"""
    print(f"=== 检测特征集 {feature_set_id} ===")
    
    # 加载特征定义
    fxdefs = load_fxdefs_for_set(feature_set_id)
    print(f"加载特征定义: {len(fxdefs)} 个")
    
    # 显示特征定义详情
    for i, fx in enumerate(fxdefs):
        print(f"\n特征 {i+1}:")
        print(f"  - ID: {fx['id']}")
        print(f"  - Pipeline ID: {fx['pipeid']}")
        print(f"  - 函数: {fx['func']}")
        print(f"  - 通道: {fx['chans']}")
        print(f"  - 参数: {fx['params']}")
        print(f"  - 输入节点: {fx['inputnodeid']}")
    
    return fxdefs

def inspect_pipeline_structures(fxdefs):
    """检测pipeline结构"""
    print(f"\n=== 检测Pipeline结构 ===")
    
    pipeline_ids = set(fx['pipeid'] for fx in fxdefs)
    pipeline_structures = {}
    
    for pipeid in pipeline_ids:
        print(f"\nPipeline {pipeid}:")
        pipeline = load_pipeline_structure(pipeid)
        pipeline_structures[pipeid] = pipeline
        
        print(f"  节点数: {len(pipeline)}")
        for nid, node in pipeline.items():
            print(f"    - {nid}: {node['func']}")
            print(f"      输入节点: {node.get('inputnodes', 'None')}")
            print(f"      参数: {node['params']}")
            print(f"      节点哈希: {node.get('node_hash', 'N/A')}")
            print(f"      父节点哈希: {node.get('parent_hash', 'N/A')}")
    
    return pipeline_structures

def inspect_dag_nodes(dag):
    """检测DAG节点的详细信息"""
    print(f"\n=== 检测DAG节点详细信息 ===")
    print(f"总节点数: {len(dag)}")
    
    # 按节点类型分类
    node_types = {}
    for node_id, node in dag.items():
        func = node['func']
        if func not in node_types:
            node_types[func] = []
        node_types[func].append(node_id)
    
    print(f"\n节点类型分布:")
    for func, nodes in node_types.items():
        print(f"  {func}: {len(nodes)} 个节点")
    
    # 详细检查每个节点
    for node_id, node in dag.items():
        print(f"\n{'='*60}")
        print(f"节点ID: {node_id}")
        print(f"函数: {node['func']}")
        print(f"输入节点: {node['inputnodes']}")
        print(f"参数: {json.dumps(node['params'], indent=4, ensure_ascii=False)}")
        
        # Pipeline路径信息
        pipeline_paths = node.get('pipeline_paths', set())
        print(f"Pipeline路径数: {len(pipeline_paths)}")
        for i, path in enumerate(pipeline_paths):
            print(f"  路径 {i+1}:")
            if len(path) == 4:
                pipeid, node_hash, parent_hash, upstream_hash = path
                print(f"    Pipeline ID: {pipeid}")
                print(f"    节点哈希: {node_hash}")
                print(f"    父节点哈希: {parent_hash}")
                print(f"    上游哈希: {upstream_hash}")
            else:
                print(f"    元组: {path}")
        
        # FxDef IDs
        fxdef_ids = node.get('fxdef_ids', [])
        print(f"FxDef IDs: {fxdef_ids}")
        
        # Upstream信息
        upstream_paths = node.get('upstream_paths', set())
        print(f"Upstream路径数: {len(upstream_paths)}")
        if upstream_paths:
            print(f"  Upstream详情:")
            for i, upstream_tuple in enumerate(upstream_paths):
                if len(upstream_tuple) == 4:
                    node_hash, parent_hash, input_node, upstream_hash = upstream_tuple
                    print(f"    {i+1}. 节点哈希: {node_hash}")
                    print(f"       父节点哈希: {parent_hash}")
                    print(f"       输入节点: {input_node}")
                    print(f"       上游哈希: {upstream_hash}")
                else:
                    print(f"    {i+1}. 元组: {upstream_tuple}")
        
        # Meta信息
        meta = node.get('meta', {})
        if meta:
            print(f"Meta信息: {meta}")

def analyze_dag_statistics(dag):
    """分析DAG统计信息"""
    print(f"\n=== DAG统计信息 ===")
    
    # 基本统计
    total_nodes = len(dag)
    total_pipeline_paths = sum(len(node.get('pipeline_paths', set())) for node in dag.values())
    total_fxdef_ids = sum(len(node.get('fxdef_ids', [])) for node in dag.values())
    total_upstream_paths = sum(len(node.get('upstream_paths', set())) for node in dag.values())
    
    print(f"总节点数: {total_nodes}")
    print(f"总Pipeline路径数: {total_pipeline_paths}")
    print(f"总FxDef ID数: {total_fxdef_ids}")
    print(f"总Upstream路径数: {total_upstream_paths}")
    
    # 节点类型统计
    node_types = {}
    for node in dag.values():
        func = node['func']
        node_types[func] = node_types.get(func, 0) + 1
    
    print(f"\n节点类型统计:")
    for func, count in sorted(node_types.items()):
        print(f"  {func}: {count}")
    
    # Pipeline路径分布
    pipeline_path_counts = {}
    for node in dag.values():
        path_count = len(node.get('pipeline_paths', set()))
        pipeline_path_counts[path_count] = pipeline_path_counts.get(path_count, 0) + 1
    
    print(f"\nPipeline路径分布:")
    for count, nodes in sorted(pipeline_path_counts.items()):
        print(f"  {count} 个路径: {nodes} 个节点")
    
    # FxDef ID分布
    fxdef_id_counts = {}
    for node in dag.values():
        fxdef_count = len(node.get('fxdef_ids', []))
        fxdef_id_counts[fxdef_count] = fxdef_id_counts.get(fxdef_count, 0) + 1
    
    print(f"\nFxDef ID分布:")
    for count, nodes in sorted(fxdef_id_counts.items()):
        print(f"  {count} 个FxDef: {nodes} 个节点")
    
    # Upstream信息分布
    upstream_paths_counts = {}
    for node in dag.values():
        upstream_count = len(node.get('upstream_paths', set()))
        upstream_paths_counts[upstream_count] = upstream_paths_counts.get(upstream_count, 0) + 1
    
    print(f"\nUpstream路径分布:")
    for count, nodes in sorted(upstream_paths_counts.items()):
        print(f"  {count} 个Upstream: {nodes} 个节点")

def save_dag_to_json(dag, filename):
    """将DAG保存为JSON文件"""
    # 转换set为list以便JSON序列化
    dag_for_json = {}
    for node_id, node in dag.items():
        node_copy = node.copy()
        # 转换pipeline_paths从set到list
        pipeline_paths = node.get('pipeline_paths', set())
        node_copy['pipeline_paths'] = [list(tuple_info) for tuple_info in pipeline_paths]
        # 转换upstream_paths从set到list
        upstream_paths = node.get('upstream_paths', set())
        node_copy['upstream_paths'] = [list(tuple_info) for tuple_info in upstream_paths]
        dag_for_json[node_id] = node_copy
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(dag_for_json, f, indent=2, ensure_ascii=False)
    
    print(f"\nDAG已保存到: {filename}")

def main():
    """主函数"""
    print("开始检测featureset_grouping函数...")
    
    # 检测特征集1
    feature_set_id = 3
    
    # 1. 检测特征定义
    fxdefs = inspect_feature_set(feature_set_id)
    
    # 2. 检测pipeline结构
    pipeline_structures = inspect_pipeline_structures(fxdefs)
    
    # 3. 构建DAG
    print(f"\n=== 构建DAG ===")
    dag = build_feature_dag(fxdefs)
    print(f"DAG构建完成")
    
    # 4. 检测DAG节点详细信息
    inspect_dag_nodes(dag)
    
    # 5. 分析DAG统计信息
    analyze_dag_statistics(dag)
    
    # 6. 保存DAG到JSON文件
    save_dag_to_json(dag, f"dag_inspection_{feature_set_id}.json")
    
    print(f"\n=== 检测完成 ===")

if __name__ == "__main__":
    main() 