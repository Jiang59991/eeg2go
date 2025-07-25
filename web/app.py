from flask import Flask, render_template, request, jsonify, send_file
import sqlite3
import os
import json
import pandas as pd
from datetime import datetime
import tempfile
import zipfile
from eeg2fx.featureset_fetcher import run_feature_set
from eeg2fx.featureset_grouping import load_fxdefs_for_set
from .utils.pipeline_visualizer import get_pipeline_visualization_data
from .config import DATABASE_PATH
from database.add_pipeline import add_pipeline, STEP_REGISTRY, validate_pipeline
from database.add_fxdef import add_fxdefs
from database.add_featureset import add_featureset

app = Flask(__name__)

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

def infer_pipeline_params(steps):
    params = {
        "chanset": None,
        "fs": None,
        "hp": None,
        "lp": None,
        "epoch": None,
        "output_type": "raw"
    }
    for step in steps:
        step_name, func, inputnames, step_params = step['step_name'], step['func'], step['inputnames'], step['params']
        if func == "pick_channels" and "include" in step_params:
            params["chanset"] = ",".join(step_params["include"])
        if func == "resample" and "sfreq" in step_params:
            params["fs"] = float(step_params["sfreq"])
        if func == "filter":
            if "hp" in step_params:
                params["hp"] = float(step_params["hp"])
            if "lp" in step_params:
                params["lp"] = float(step_params["lp"])
        if func == "epoch" and "duration" in step_params:
            params["epoch"] = float(step_params["duration"])
            params["output_type"] = "epochs"
    return params

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html', STEP_REGISTRY=STEP_REGISTRY)

@app.route('/api/datasets')
def get_datasets():
    """Get all datasets"""
    conn = get_db_connection()
    datasets = conn.execute('SELECT * FROM datasets').fetchall()
    conn.close()
    return jsonify([dict(dataset) for dataset in datasets])

@app.route('/api/recordings')
def get_recordings():
    """Get recording files list"""
    dataset_id = request.args.get('dataset_id', type=int)
    conn = get_db_connection()
    
    if dataset_id:
        recordings = conn.execute('''
            SELECT r.*, s.age, s.sex 
            FROM recordings r 
            LEFT JOIN subjects s ON r.subject_id = s.subject_id 
            WHERE r.dataset_id = ?
        ''', (dataset_id,)).fetchall()
    else:
        recordings = conn.execute('''
            SELECT r.*, s.age, s.sex 
            FROM recordings r 
            LEFT JOIN subjects s ON r.subject_id = s.subject_id
        ''').fetchall()
    
    conn.close()
    return jsonify([dict(recording) for recording in recordings])

@app.route('/api/feature_sets')
def get_feature_sets():
    """Get feature sets list"""
    conn = get_db_connection()
    feature_sets = conn.execute('SELECT * FROM feature_sets').fetchall()
    conn.close()
    return jsonify([dict(feature_set) for feature_set in feature_sets])

@app.route('/api/feature_set_details/<int:feature_set_id>')
def get_feature_set_details(feature_set_id):
    """Get feature set details"""
    conn = get_db_connection()
    
    # Get feature set basic info
    feature_set = conn.execute('SELECT * FROM feature_sets WHERE id = ?', (feature_set_id,)).fetchone()
    
    # Get feature definitions in the feature set
    features = conn.execute('''
        SELECT f.*, p.shortname as pipeline_name, p.description as pipeline_desc
        FROM fxdef f
        JOIN feature_set_items fsi ON f.id = fsi.fxdef_id
        LEFT JOIN pipedef p ON f.pipedef_id = p.id
        WHERE fsi.feature_set_id = ?
    ''', (feature_set_id,)).fetchall()
    
    conn.close()
    
    return jsonify({
        'feature_set': dict(feature_set),
        'features': [dict(feature) for feature in features]
    })

@app.route('/api/extract_features', methods=['POST'])
def extract_features():
    """Extract features"""
    data = request.json
    recording_ids = data.get('recording_ids', [])
    feature_set_id = data.get('feature_set_id')
    
    if not recording_ids or not feature_set_id:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    results = []
    errors = []
    
    for recording_id in recording_ids:
        try:
            print(f"Processing recording file {recording_id}...")
            result = run_feature_set(feature_set_id, recording_id)
            results.append({
                'recording_id': recording_id,
                'status': 'success',
                'features': result
            })
            print(f"Recording file {recording_id} processed successfully")
        except Exception as e:
            error_msg = f"Error processing recording file {recording_id}: {str(e)}"
            errors.append(error_msg)
            print(error_msg)
    
    return jsonify({
        'results': results,
        'errors': errors,
        'total_processed': len(recording_ids),
        'success_count': len(results),
        'error_count': len(errors)
    })

@app.route('/api/feature_values')
def get_feature_values():
    recording_id = request.args.get('recording_id', type=int)
    if not recording_id:
        return jsonify({'error': 'Missing required parameter: recording_id'}), 400

    conn = get_db_connection()
    # 查找该recording的所有特征定义
    fxdefs = conn.execute('''
        SELECT f.* FROM fxdef f
        JOIN feature_values fv ON f.id = fv.fxdef_id
        WHERE fv.recording_id = ?
    ''', (recording_id,)).fetchall()

    feature_values = {}
    for fxdef in fxdefs:
        fxdef_id = fxdef['id']
        value = conn.execute('''
            SELECT * FROM feature_values 
            WHERE fxdef_id = ? AND recording_id = ?
        ''', (fxdef_id, recording_id)).fetchone()
        
        if value:
            feature_values[fxdef['shortname']] = {
                'value': json.loads(value['value']) if value['value'] != 'null' else None,
                'dim': value['dim'],
                'shape': json.loads(value['shape']) if value['shape'] else [],
                'notes': value['notes']
            }
        else:
            feature_values[fxdef['shortname']] = {
                'value': None,
                'dim': 'unknown',
                'shape': [],
                'notes': 'Feature not calculated yet'
            }
    conn.close()
    return jsonify(feature_values)

@app.route('/api/export_features', methods=['POST'])
def export_features():
    """Export feature data"""
    data = request.json
    recording_ids = data.get('recording_ids', [])
    feature_set_id = data.get('feature_set_id')
    
    if not recording_ids or not feature_set_id:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    conn = get_db_connection()
    
    # Get feature set info
    feature_set = conn.execute('SELECT * FROM feature_sets WHERE id = ?', (feature_set_id,)).fetchone()
    
    # Get feature definitions
    fxdefs = conn.execute('''
        SELECT f.* FROM fxdef f
        JOIN feature_set_items fsi ON f.id = fsi.fxdef_id
        WHERE fsi.feature_set_id = ?
    ''', (feature_set_id,)).fetchall()
    
    # Get recording files info
    recordings = conn.execute('''
        SELECT r.*, s.age, s.sex 
        FROM recordings r 
        LEFT JOIN subjects s ON r.subject_id = s.subject_id 
        WHERE r.id IN ({})
    '''.format(','.join('?' * len(recording_ids))), recording_ids).fetchall()
    
    # Build data frame
    data_rows = []
    for recording in recordings:
        row = {
            'recording_id': recording['id'],
            'subject_id': recording['subject_id'],
            'filename': recording['filename'],
            'duration': recording['duration'],
            'channels': recording['channels'],
            'sampling_rate': recording['sampling_rate'],
            'age': recording['age'],
            'sex': recording['sex']
        }
        
        # Add feature values
        for fxdef in fxdefs:
            fxdef_id = fxdef['id']
            value = conn.execute('''
                SELECT * FROM feature_values 
                WHERE fxdef_id = ? AND recording_id = ?
            ''', (fxdef_id, recording['id'])).fetchone()
            
            if value and value['value'] != 'null':
                try:
                    parsed_value = json.loads(value['value'])
                    if isinstance(parsed_value, (int, float)):
                        row[fxdef['shortname']] = parsed_value
                    elif isinstance(parsed_value, list) and len(parsed_value) == 1:
                        row[fxdef['shortname']] = parsed_value[0]
                    else:
                        row[fxdef['shortname']] = str(parsed_value)
                except:
                    row[fxdef['shortname']] = None
            else:
                row[fxdef['shortname']] = None
        
        data_rows.append(row)
    
    conn.close()
    
    # Create DataFrame and export
    df = pd.DataFrame(data_rows)
    
    # Create temporary file
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        df.to_csv(f.name, index=False)
        temp_file = f.name
    
    return send_file(
        temp_file,
        as_attachment=True,
        download_name=f"eeg_features_{feature_set['name']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
        mimetype='text/csv'
    )

@app.route('/api/pipelines')
def get_pipelines():
    """Get all pipelines"""
    conn = get_db_connection()
    pipelines = conn.execute('SELECT * FROM pipedef').fetchall()
    conn.close()
    return jsonify([dict(pipeline) for pipeline in pipelines])

@app.route('/api/pipeline_details/<int:pipeline_id>')
def get_pipeline_details(pipeline_id):
    """Get pipeline details with nodes and edges"""
    conn = get_db_connection()
    
    # Get pipeline basic info
    pipeline = conn.execute('SELECT * FROM pipedef WHERE id = ?', (pipeline_id,)).fetchone()
    
    # Get pipeline nodes
    nodes = conn.execute('''
        SELECT pn.* FROM pipe_nodes pn
        JOIN pipe_edges pe ON pn.nodeid = pe.from_node OR pn.nodeid = pe.to_node
        WHERE pe.pipedef_id = ?
        GROUP BY pn.nodeid
    ''', (pipeline_id,)).fetchall()
    
    # Get pipeline edges
    edges = conn.execute('''
        SELECT * FROM pipe_edges WHERE pipedef_id = ?
    ''', (pipeline_id,)).fetchall()
    
    # Get fxdefs that use this pipeline
    fxdefs = conn.execute('''
        SELECT * FROM fxdef WHERE pipedef_id = ?
    ''', (pipeline_id,)).fetchall()
    
    conn.close()
    
    return jsonify({
        'pipeline': dict(pipeline),
        'nodes': [dict(node) for node in nodes],
        'edges': [dict(edge) for edge in edges],
        'fxdefs': [dict(fxdef) for fxdef in fxdefs]
    })

@app.route('/api/pipeline_visualization/<int:pipeline_id>')
def get_pipeline_visualization(pipeline_id):
    """Get pipeline visualization data including SVG"""
    try:
        visualization_data = get_pipeline_visualization_data(pipeline_id)
        if visualization_data:
            return jsonify(visualization_data)
        else:
            return jsonify({'error': 'Pipeline not found'}), 404
    except Exception as e:
        return jsonify({'error': f'Error generating visualization: {str(e)}'}), 500

@app.route('/api/fxdefs')
def get_fxdefs():
    """Get all feature definitions"""
    conn = get_db_connection()
    fxdefs = conn.execute('''
        SELECT f.*, p.shortname as pipeline_name 
        FROM fxdef f 
        LEFT JOIN pipedef p ON f.pipedef_id = p.id
    ''').fetchall()
    conn.close()
    return jsonify([dict(fxdef) for fxdef in fxdefs])

@app.route('/api/fxdef_details/<int:fxdef_id>')
def get_fxdef_details(fxdef_id):
    """Get feature definition details"""
    conn = get_db_connection()
    
    # Get fxdef basic info
    fxdef = conn.execute('''
        SELECT f.*, p.shortname as pipeline_name, p.description as pipeline_desc
        FROM fxdef f
        LEFT JOIN pipedef p ON f.pipedef_id = p.id
        WHERE f.id = ?
    ''', (fxdef_id,)).fetchone()
    
    # Get feature sets that include this fxdef
    feature_sets = conn.execute('''
        SELECT fs.* FROM feature_sets fs
        JOIN feature_set_items fsi ON fs.id = fsi.feature_set_id
        WHERE fsi.fxdef_id = ?
    ''', (fxdef_id,)).fetchall()
    
    # Get sample feature values (first 10)
    sample_values = conn.execute('''
        SELECT fv.*, r.filename 
        FROM feature_values fv
        JOIN recordings r ON fv.recording_id = r.id
        WHERE fv.fxdef_id = ?
        LIMIT 10
    ''', (fxdef_id,)).fetchall()
    
    conn.close()
    
    return jsonify({
        'fxdef': dict(fxdef),
        'feature_sets': [dict(fs) for fs in feature_sets],
        'sample_values': [dict(sv) for sv in sample_values]
    })

@app.route('/api/featuresets_detailed')
def get_featuresets_detailed():
    """Get all feature sets with detailed information"""
    conn = get_db_connection()
    
    feature_sets = conn.execute('SELECT * FROM feature_sets').fetchall()
    detailed_sets = []
    
    for fs in feature_sets:
        # Get fxdefs in this feature set
        fxdefs = conn.execute('''
            SELECT f.*, p.shortname as pipeline_name
            FROM fxdef f
            JOIN feature_set_items fsi ON f.id = fsi.fxdef_id
            LEFT JOIN pipedef p ON f.pipedef_id = p.id
            WHERE fsi.feature_set_id = ?
        ''', (fs['id'],)).fetchall()
        
        detailed_sets.append({
            'feature_set': dict(fs),
            'fxdefs': [dict(fxdef) for fxdef in fxdefs],
            'fxdef_count': len(fxdefs)
        })
    
    conn.close()
    return jsonify(detailed_sets)

@app.route('/api/add_pipeline', methods=['POST'])
def api_add_pipeline():
    pipeline_def = request.json
    steps = pipeline_def.get("steps", [])
    # 1. 校验步骤参数
    try:
        # 转换为add_pipeline需要的格式
        steps_for_validate = [[s["step_name"], s["func"], s["inputnames"], s["params"]] for s in steps]
        validate_pipeline(steps_for_validate, STEP_REGISTRY)
    except Exception as e:
        return jsonify({'success': False, 'error': f'Parameter validation failed: {str(e)}'}), 400

    # 2. 检查重复/无效步骤（如重复epoch）
    seen = set()
    for s in steps:
        if s["func"] in seen and s["func"] in ["epoch"]:  # 可扩展更多只允许出现一次的步骤
            return jsonify({'success': False, 'error': f"Step '{s['func']}' is duplicated, which is not allowed."}), 400
        seen.add(s["func"])

    # 3. 推断参数
    inferred = infer_pipeline_params(steps)
    # 4. 合成完整pipeline_def
    full_pipeline_def = {
        "shortname": pipeline_def.get("shortname"),
        "description": pipeline_def.get("description"),
        "source": pipeline_def.get("source"),
        "chanset": pipeline_def.get("chanset"),   # <--- 改这里
        "fs": inferred["fs"],
        "hp": inferred["hp"],
        "lp": inferred["lp"],
        "epoch": inferred["epoch"],
        "steps": steps_for_validate,
        "sample_rating": 5.0  # 可选
    }
    try:
        pipeline_id = add_pipeline(full_pipeline_def)
        return jsonify({'success': True, 'pipeline_id': pipeline_id})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/steps/registry')
def get_steps_registry():
    return jsonify(STEP_REGISTRY)

@app.route('/api/pipeline/validate', methods=['POST'])
def validate_pipeline_api():
    pipeline = request.json.get("steps")
    try:
        validate_pipeline(pipeline, STEP_REGISTRY)
        return jsonify({"valid": True})
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)}), 400

@app.route('/api/feature_functions')
def api_feature_functions():
    from eeg2fx.function_registry import FEATURE_METADATA
    def infer_dim(meta_type):
        if meta_type == "scalar":
            return "scalar"
        else:
            return "1d"
    return jsonify([
        {"name": k, "dim": infer_dim(v["type"])}
        for k, v in FEATURE_METADATA.items()
    ])

@app.route('/api/add_fxdef', methods=['POST'])
def api_add_fxdef():
    data = request.json
    try:
        # 这里直接用add_fxdefs
        add_fxdefs(data)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/add_featureset', methods=['POST'])
def api_add_featureset():
    data = request.json
    try:
        add_featureset(data)
        return jsonify({'success': True})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 400

@app.route('/api/featureset_dag/<int:feature_set_id>')
def get_featureset_dag(feature_set_id):
    """Get merged DAG visualization data for a feature set"""
    try:
        from eeg2fx.featureset_grouping import load_fxdefs_for_set, build_feature_dag
        
        # 加载feature set中的所有fxdefs
        fxdefs = load_fxdefs_for_set(feature_set_id)
        
        if not fxdefs:
            return jsonify({'error': 'No feature definitions found in this feature set'}), 404
        
        # 构建合并的DAG
        dag = build_feature_dag(fxdefs)
        
        # 转换为Cytoscape.js格式
        nodes = []
        edges = []
        node_id_map = {}  # 用于映射内部ID到Cytoscape ID
        
        # 添加节点
        for nid, node in dag.items():
            # 跳过split_channel节点
            if "split_channel" in node["func"]:
                continue
                
            cytoscape_id = f"node_{nid}"
            node_id_map[nid] = cytoscape_id
            
            # 确定节点类型和样式
            node_type = "process"
            if "feature" in node["func"]:
                node_type = "output"
            elif node["func"] in ["raw", "load_recording"]:
                node_type = "input"
            
            # 格式化参数显示
            params_str = ""
            if node["params"]:
                param_pairs = []
                for key, value in node["params"].items():
                    param_pairs.append(f"{key}={value}")
                params_str = "(" + ", ".join(param_pairs) + ")"
            
            # 组合函数名和参数
            node_label = f"{node['func']}\n{params_str}" if params_str else node["func"]
            
            nodes.append({
                "data": {
                    "id": cytoscape_id,
                    "label": node_label,
                    "params": json.dumps(node["params"], separators=(',', ':')),
                    "type": node_type
                },
                "classes": node_type
            })
        
        # 添加边
        edge_set = set()
        for nid, node in dag.items():
            # 跳过split_channel节点
            if "split_channel" in node["func"]:
                continue
                
            from_id = node_id_map[nid]
            for input_node in node["inputnodes"]:
                # 如果输入节点是split_channel，找到它的输入节点
                if input_node in dag and "split_channel" in dag[input_node]["func"]:
                    # 跳过split_channel，直接连接到它的输入
                    split_inputs = dag[input_node]["inputnodes"]
                    for split_input in split_inputs:
                        if split_input in node_id_map:
                            to_id = node_id_map[split_input]
                            edge_key = (to_id, from_id)
                            if edge_key not in edge_set:
                                edges.append({
                                    "data": {
                                        "id": f"edge_{to_id}_{from_id}",
                                        "source": to_id,
                                        "target": from_id
                                    }
                                })
                                edge_set.add(edge_key)
                elif input_node in node_id_map:
                    to_id = node_id_map[input_node]
                    edge_key = (to_id, from_id)
                    if edge_key not in edge_set:
                        edges.append({
                            "data": {
                                "id": f"edge_{to_id}_{from_id}",
                                "source": to_id,
                                "target": from_id
                            }
                        })
                        edge_set.add(edge_key)
        
        return jsonify({
            'success': True,
            'cytoscape_data': {
                'nodes': nodes,
                'edges': edges
            },
            'feature_count': len(fxdefs),
            'node_count': len(nodes),
            'edge_count': len(edges)
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to generate DAG: {str(e)}'}), 500

@app.route('/api/experiments')
def get_experiments():
    """Get all experiment results"""
    conn = get_db_connection()
    
    # 获取实验列表，包含数据集和特征集信息
    experiments = conn.execute('''
        SELECT er.*, 
               d.name as dataset_name,
               fs.name as feature_set_name
        FROM experiment_results er
        LEFT JOIN datasets d ON er.dataset_id = d.id
        LEFT JOIN feature_sets fs ON er.feature_set_id = fs.id
        ORDER BY er.run_time DESC
    ''').fetchall()
    
    conn.close()
    
    return jsonify([dict(exp) for exp in experiments])

@app.route('/api/experiment_details/<int:experiment_id>')
def get_experiment_details(experiment_id):
    """Get detailed experiment results"""
    conn = get_db_connection()
    
    # 获取实验基本信息
    experiment = conn.execute('''
        SELECT er.*, 
               d.name as dataset_name,
               fs.name as feature_set_name
        FROM experiment_results er
        LEFT JOIN datasets d ON er.dataset_id = d.id
        LEFT JOIN feature_sets fs ON er.feature_set_id = fs.id
        WHERE er.id = ?
    ''', (experiment_id,)).fetchone()
    
    if not experiment:
        conn.close()
        return jsonify({'error': 'Experiment not found'}), 404
    
    # 获取特征级别的结果
    feature_results = conn.execute('''
        SELECT efr.*, fd.shortname as feature_shortname, fd.chans as feature_channels
        FROM experiment_feature_results efr
        LEFT JOIN fxdef fd ON efr.fxdef_id = fd.id
        WHERE efr.experiment_result_id = ?
        ORDER BY efr.rank_position ASC, efr.metric_value DESC
    ''', (experiment_id,)).fetchall()
    
    # 获取实验元数据
    metadata = conn.execute('''
        SELECT key, value, value_type
        FROM experiment_metadata
        WHERE experiment_result_id = ?
        ORDER BY key
    ''', (experiment_id,)).fetchall()
    
    conn.close()
    
    # 检查输出目录是否存在文件
    output_files = []
    if experiment['output_dir'] and os.path.exists(experiment['output_dir']):
        try:
            for filename in os.listdir(experiment['output_dir']):
                file_path = os.path.join(experiment['output_dir'], filename)
                if os.path.isfile(file_path):
                    stat = os.stat(file_path)
                    file_info = {
                        'name': filename,
                        'size': stat.st_size,
                        'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                        'type': get_file_type(filename)
                    }
                    output_files.append(file_info)
            
            # 按文件类型排序
            output_files.sort(key=lambda x: get_file_sort_order(x['type']))
        except Exception as e:
            print(f"Error listing output files: {e}")
    
    return jsonify({
        'experiment': dict(experiment),
        'feature_results': [dict(fr) for fr in feature_results],
        'metadata': [dict(md) for md in metadata],
        'output_files': output_files
    })

@app.route('/api/experiment_summary/<int:experiment_id>')
def get_experiment_summary(experiment_id):
    """Get experiment summary statistics"""
    conn = get_db_connection()
    
    # 获取实验基本信息
    experiment = conn.execute('SELECT * FROM experiment_results WHERE id = ?', (experiment_id,)).fetchone()
    
    if not experiment:
        conn.close()
        return jsonify({'error': 'Experiment not found'}), 404
    
    # 根据实验类型获取不同的统计信息
    if experiment['experiment_type'] == 'correlation':
        # 相关性实验统计
        stats = conn.execute('''
            SELECT 
                COUNT(*) as total_features,
                COUNT(CASE WHEN metric_value >= 0.3 THEN 1 END) as significant_correlations,
                COUNT(CASE WHEN significance_level IN ('p<0.001', 'p<0.01', 'p<0.05') THEN 1 END) as significant_features,
                AVG(metric_value) as avg_correlation,
                MAX(metric_value) as max_correlation,
                MIN(metric_value) as min_correlation
            FROM experiment_feature_results
            WHERE experiment_result_id = ? AND metric_name = 'correlation_coefficient'
        ''', (experiment_id,)).fetchone()
        
    elif experiment['experiment_type'] == 'classification':
        # 分类实验统计
        stats = conn.execute('''
            SELECT 
                COUNT(*) as total_features,
                COUNT(CASE WHEN metric_value > 0.5 THEN 1 END) as important_features,
                AVG(metric_value) as avg_importance,
                MAX(metric_value) as max_importance,
                MIN(metric_value) as min_importance
            FROM experiment_feature_results
            WHERE experiment_result_id = ? AND metric_name = 'importance_score'
        ''', (experiment_id,)).fetchone()
        
    elif experiment['experiment_type'] == 'feature_statistics':
        # 特征统计实验统计
        stats = conn.execute('''
            SELECT 
                COUNT(*) as total_features,
                COUNT(CASE WHEN metric_name = 'importance_score' THEN 1 END) as important_features,
                AVG(CASE WHEN metric_name = 'importance_score' THEN metric_value END) as avg_importance,
                MAX(CASE WHEN metric_name = 'importance_score' THEN metric_value END) as max_importance,
                MIN(CASE WHEN metric_name = 'importance_score' THEN metric_value END) as min_importance
            FROM experiment_feature_results
            WHERE experiment_result_id = ?
        ''', (experiment_id,)).fetchone()
        
    else:
        # 其他类型实验的通用统计
        stats = conn.execute('''
            SELECT 
                COUNT(*) as total_features,
                AVG(metric_value) as avg_metric,
                MAX(metric_value) as max_metric,
                MIN(metric_value) as min_metric
            FROM experiment_feature_results
            WHERE experiment_result_id = ?
        ''', (experiment_id,)).fetchone()
    
    conn.close()
    
    return jsonify({
        'experiment': dict(experiment),
        'statistics': dict(stats)
    })

@app.route('/api/experiment_files/<int:experiment_id>')
def get_experiment_files(experiment_id):
    """Get list of files generated by an experiment"""
    conn = get_db_connection()
    
    # 获取实验的输出目录
    experiment = conn.execute('SELECT output_dir FROM experiment_results WHERE id = ?', (experiment_id,)).fetchone()
    conn.close()
    
    if not experiment or not experiment['output_dir']:
        return jsonify({'error': 'Experiment output directory not found'}), 404
    
    output_dir = experiment['output_dir']
    
    if not os.path.exists(output_dir):
        return jsonify({'error': 'Output directory does not exist'}), 404
    
    try:
        files = []
        for filename in os.listdir(output_dir):
            file_path = os.path.join(output_dir, filename)
            if os.path.isfile(file_path):
                # 获取文件信息
                stat = os.stat(file_path)
                file_info = {
                    'name': filename,
                    'size': stat.st_size,
                    'modified': datetime.fromtimestamp(stat.st_mtime).isoformat(),
                    'type': get_file_type(filename)
                }
                files.append(file_info)
        
        # 按文件类型排序：图片在前，然后是CSV，最后是其他
        files.sort(key=lambda x: get_file_sort_order(x['type']))
        
        return jsonify({
            'experiment_id': experiment_id,
            'output_dir': output_dir,
            'files': files
        })
        
    except Exception as e:
        return jsonify({'error': f'Failed to list files: {str(e)}'}), 500

@app.route('/api/experiment_file/<int:experiment_id>/<path:filename>')
def get_experiment_file(experiment_id, filename):
    """Download or view a specific experiment file"""
    conn = get_db_connection()
    
    # 获取实验的输出目录
    experiment = conn.execute('SELECT output_dir FROM experiment_results WHERE id = ?', (experiment_id,)).fetchone()
    conn.close()
    
    if not experiment or not experiment['output_dir']:
        return jsonify({'error': 'Experiment output directory not found'}), 404
    
    output_dir = experiment['output_dir']
    file_path = os.path.join(output_dir, filename)
    
    if not os.path.exists(file_path):
        return jsonify({'error': 'File not found'}), 404
    
    try:
        # 根据文件类型决定如何返回
        file_type = get_file_type(filename)
        
        if file_type == 'image':
            # 图片文件直接返回
            return send_file(file_path, mimetype='image/png')
        elif file_type == 'csv':
            # CSV文件返回内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({
                'filename': filename,
                'type': 'csv',
                'content': content
            })
        elif file_type == 'text':
            # 文本文件返回内容
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            return jsonify({
                'filename': filename,
                'type': 'text',
                'content': content
            })
        else:
            # 其他文件类型下载
            return send_file(file_path, as_attachment=True, download_name=filename)
            
    except Exception as e:
        return jsonify({'error': f'Failed to read file: {str(e)}'}), 500

def get_file_type(filename):
    """Determine file type based on extension"""
    ext = filename.lower().split('.')[-1] if '.' in filename else ''
    
    if ext in ['png', 'jpg', 'jpeg', 'gif', 'svg']:
        return 'image'
    elif ext == 'csv':
        return 'csv'
    elif ext in ['txt', 'log', 'json']:
        return 'text'
    else:
        return 'other'

def get_file_sort_order(file_type):
    """Get sort order for file types"""
    order_map = {
        'image': 1,
        'csv': 2,
        'text': 3,
        'other': 4
    }
    return order_map.get(file_type, 5)

import threading
import time

@app.route('/api/run_experiment', methods=['POST'])
def run_experiment_api():
    """Start a new experiment asynchronously"""
    try:
        data = request.json
        
        # 验证必需参数
        required_fields = ['experiment_type', 'dataset_id', 'feature_set_id']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        experiment_type = data['experiment_type']
        dataset_id = int(data['dataset_id'])
        feature_set_id = int(data['feature_set_id'])
        experiment_name = data.get('experiment_name', '')
        notes = data.get('notes', '')
        
        # 获取额外参数
        extra_args = {}
        if 'parameters' in data:
            extra_args.update(data['parameters'])
        
        # 创建输出目录
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_dir = f"logs/experiments/{experiment_type}_{dataset_id}_{feature_set_id}_{timestamp}"
        
        # 先在数据库中创建实验记录（状态为running）
        conn = get_db_connection()
        cursor = conn.execute('''
            INSERT INTO experiment_results (
                experiment_type, experiment_name, dataset_id, feature_set_id,
                parameters, output_dir, status, notes, run_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        ''', (
            experiment_type, experiment_name, dataset_id, feature_set_id,
            json.dumps(extra_args), output_dir, 'running', notes, datetime.now()
        ))
        experiment_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # 在后台线程中运行实验
        def run_experiment_async():
            try:
                # 导入实验引擎
                from feature_mill.experiment_engine import run_experiment
                
                # 运行实验
                result = run_experiment(
                    experiment_type=experiment_type,
                    dataset_id=dataset_id,
                    feature_set_id=feature_set_id,
                    output_dir=output_dir,
                    extra_args=extra_args
                )
                
                # 更新实验状态为完成
                conn = get_db_connection()
                conn.execute('''
                    UPDATE experiment_results 
                    SET status = 'completed', duration_seconds = ?, result_summary = ?
                    WHERE id = ?
                ''', (result['duration'], result['summary'], experiment_id))
                conn.commit()
                conn.close()
                
                logger.info(f"Experiment {experiment_id} completed successfully")
                
            except Exception as e:
                # 更新实验状态为失败
                conn = get_db_connection()
                conn.execute('''
                    UPDATE experiment_results 
                    SET status = 'failed', notes = ?
                    WHERE id = ?
                ''', (f"Experiment failed: {str(e)}", experiment_id))
                conn.commit()
                conn.close()
                
                logger.error(f"Experiment {experiment_id} failed: {e}")
        
        # 启动后台线程
        thread = threading.Thread(target=run_experiment_async)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'started',
            'message': f'Experiment {experiment_type} started successfully',
            'experiment_id': experiment_id,
            'output_dir': output_dir
        })
        
    except Exception as e:
        logger.error(f"Failed to start experiment: {e}")
        return jsonify({'error': f'Failed to start experiment: {str(e)}'}), 500

@app.route('/api/experiment_types')
def get_experiment_types():
    """Get available experiment types and their parameters"""
    try:
        from feature_mill.experiment_engine import list_experiments, get_experiment_info
        
        experiment_types = []
        available_types = list_experiments()
        
        for exp_type in available_types:
            info = get_experiment_info(exp_type)
            if 'error' not in info:
                experiment_types.append({
                    'type': exp_type,
                    'name': info.get('name', exp_type),
                    'description': info.get('docstring', 'No description available'),
                    'has_run_function': info.get('has_run_function', False)
                })
        
        return jsonify(experiment_types)
        
    except Exception as e:
        logger.error(f"Failed to get experiment types: {e}")
        return jsonify({'error': f'Failed to get experiment types: {str(e)}'}), 500

@app.route('/api/experiment_status/<int:experiment_id>')
def get_experiment_status(experiment_id):
    """Get experiment status and progress"""
    try:
        conn = get_db_connection()
        experiment = conn.execute('''
            SELECT id, experiment_type, experiment_name, status, run_time, 
                   duration_seconds, result_summary, notes, output_dir
            FROM experiment_results 
            WHERE id = ?
        ''', (experiment_id,)).fetchone()
        conn.close()
        
        if not experiment:
            return jsonify({'error': 'Experiment not found'}), 404
        
        # 计算运行时间
        run_time = datetime.fromisoformat(experiment['run_time']) if experiment['run_time'] else None
        duration = None
        if run_time:
            if experiment['status'] == 'running':
                duration = (datetime.now() - run_time).total_seconds()
            else:
                duration = experiment['duration_seconds']
        
        return jsonify({
            'id': experiment['id'],
            'experiment_type': experiment['experiment_type'],
            'experiment_name': experiment['experiment_name'],
            'status': experiment['status'],
            'run_time': experiment['run_time'],
            'duration': duration,
            'result_summary': experiment['result_summary'],
            'notes': experiment['notes'],
            'output_dir': experiment['output_dir']
        })
        
    except Exception as e:
        logger.error(f"Failed to get experiment status: {e}")
        return jsonify({'error': f'Failed to get experiment status: {str(e)}'}), 500

@app.route('/api/start_feature_extraction', methods=['POST'])
def start_feature_extraction_api():
    """Start a new feature extraction task asynchronously"""
    try:
        data = request.json
        
        # 验证必需参数
        required_fields = ['dataset_id', 'feature_set_id']
        for field in required_fields:
            if not data.get(field):
                return jsonify({'error': f'Missing required field: {field}'}), 400
        
        dataset_id = int(data['dataset_id'])
        feature_set_id = int(data['feature_set_id'])
        task_name = data.get('task_name', '')
        
        # 获取数据集中的recording数量
        conn = get_db_connection()
        recording_count = conn.execute('''
            SELECT COUNT(*) as count FROM recordings WHERE dataset_id = ?
        ''', (dataset_id,)).fetchone()['count']
        
        # 创建提取任务记录
        cursor = conn.execute('''
            INSERT INTO feature_extraction_tasks (
                dataset_id, feature_set_id, task_name, status, 
                total_recordings, processed_recordings, start_time
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        ''', (
            dataset_id, feature_set_id, task_name, 'running',
            recording_count, 0, datetime.now()
        ))
        task_id = cursor.lastrowid
        conn.commit()
        conn.close()
        
        # 在后台线程中运行特征提取
        def run_extraction_async():
            try:
                from eeg2fx.featureset_fetcher import run_feature_set
                
                conn = get_db_connection()
                recordings = conn.execute('''
                    SELECT id FROM recordings WHERE dataset_id = ?
                ''', (dataset_id,)).fetchall()
                conn.close()
                
                processed_count = 0
                failed_count = 0
                
                for recording in recordings:
                    recording_id = recording['id']
                    try:
                        # 运行特征提取
                        result = run_feature_set(feature_set_id, recording_id)
                        
                        # 更新进度
                        processed_count += 1
                        conn = get_db_connection()
                        conn.execute('''
                            UPDATE feature_extraction_tasks 
                            SET processed_recordings = ?
                            WHERE id = ?
                        ''', (processed_count, task_id))
                        conn.commit()
                        conn.close()
                        
                    except Exception as e:
                        failed_count += 1
                        logger.warning(f"Failed to extract features for recording {recording_id}: {e}")
                
                # 更新任务状态为完成
                conn = get_db_connection()
                conn.execute('''
                    UPDATE feature_extraction_tasks 
                    SET status = 'completed', end_time = ?, duration_seconds = ?,
                        failed_recordings = ?, result_file = ?
                    WHERE id = ?
                ''', (
                    datetime.now(),
                    (datetime.now() - datetime.fromisoformat(conn.execute('SELECT start_time FROM feature_extraction_tasks WHERE id = ?', (task_id,)).fetchone()['start_time'])).total_seconds(),
                    failed_count,
                    f"logs/extractions/extraction_{task_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv"
                ))
                conn.commit()
                conn.close()
                
                logger.info(f"Feature extraction task {task_id} completed successfully")
                
            except Exception as e:
                # 更新任务状态为失败
                conn = get_db_connection()
                conn.execute('''
                    UPDATE feature_extraction_tasks 
                    SET status = 'failed', notes = ?
                    WHERE id = ?
                ''', (f"Extraction failed: {str(e)}", task_id))
                conn.commit()
                conn.close()
                
                logger.error(f"Feature extraction task {task_id} failed: {e}")
        
        # 启动后台线程
        thread = threading.Thread(target=run_extraction_async)
        thread.daemon = True
        thread.start()
        
        return jsonify({
            'status': 'started',
            'message': f'Feature extraction started successfully',
            'task_id': task_id,
            'total_recordings': recording_count
        })
        
    except Exception as e:
        logger.error(f"Failed to start feature extraction: {e}")
        return jsonify({'error': f'Failed to start feature extraction: {str(e)}'}), 500

@app.route('/api/feature_extraction_tasks')
def get_feature_extraction_tasks():
    """Get all feature extraction tasks"""
    try:
        conn = get_db_connection()
        tasks = conn.execute('''
            SELECT t.*, d.name as dataset_name, fs.name as feature_set_name
            FROM feature_extraction_tasks t
            LEFT JOIN datasets d ON t.dataset_id = d.id
            LEFT JOIN feature_sets fs ON t.feature_set_id = fs.id
            ORDER BY t.start_time DESC
        ''').fetchall()
        conn.close()
        
        return jsonify([dict(task) for task in tasks])
        
    except Exception as e:
        logger.error(f"Failed to get feature extraction tasks: {e}")
        return jsonify({'error': f'Failed to get feature extraction tasks: {str(e)}'}), 500

@app.route('/api/feature_extraction_status/<int:task_id>')
def get_feature_extraction_status(task_id):
    """Get feature extraction task status"""
    try:
        conn = get_db_connection()
        task = conn.execute('''
            SELECT * FROM feature_extraction_tasks WHERE id = ?
        ''', (task_id,)).fetchone()
        conn.close()
        
        if not task:
            return jsonify({'error': 'Task not found'}), 404
        
        # 计算进度
        progress = 0
        if task['total_recordings'] > 0:
            progress = (task['processed_recordings'] / task['total_recordings']) * 100
        
        # 计算运行时间
        start_time = datetime.fromisoformat(task['start_time']) if task['start_time'] else None
        duration = None
        if start_time:
            if task['status'] == 'running':
                duration = (datetime.now() - start_time).total_seconds()
            else:
                duration = task['duration_seconds']
        
        return jsonify({
            'id': task['id'],
            'dataset_id': task['dataset_id'],
            'feature_set_id': task['feature_set_id'],
            'task_name': task['task_name'],
            'status': task['status'],
            'total_recordings': task['total_recordings'],
            'processed_recordings': task['processed_recordings'],
            'failed_recordings': task['failed_recordings'],
            'progress': progress,
            'start_time': task['start_time'],
            'end_time': task['end_time'],
            'duration': duration,
            'result_file': task['result_file'],
            'notes': task['notes']
        })
        
    except Exception as e:
        logger.error(f"Failed to get feature extraction status: {e}")
        return jsonify({'error': f'Failed to get feature extraction status: {str(e)}'}), 500

@app.route('/api/download_extraction_result/<int:task_id>')
def download_extraction_result(task_id):
    """Download feature extraction result file"""
    try:
        conn = get_db_connection()
        task = conn.execute('''
            SELECT result_file FROM feature_extraction_tasks WHERE id = ?
        ''', (task_id,)).fetchone()
        conn.close()
        
        if not task or not task['result_file']:
            return jsonify({'error': 'Result file not found'}), 404
        
        result_file = task['result_file']
        if not os.path.exists(result_file):
            return jsonify({'error': 'Result file does not exist'}), 404
        
        return send_file(result_file, as_attachment=True, download_name=f"extraction_{task_id}.csv")
        
    except Exception as e:
        logger.error(f"Failed to download extraction result: {e}")
        return jsonify({'error': f'Failed to download extraction result: {str(e)}'}), 500

@app.route('/api/execute_dag/<int:feature_set_id>', methods=['POST'])
def execute_dag_api(feature_set_id):
    """执行DAG并返回执行报告"""
    try:
        data = request.get_json()
        recording_id = data.get('recording_id', 22)
        
        # 导入必要的模块
        from eeg2fx.featureset_grouping import build_feature_dag, load_fxdefs_for_set
        from eeg2fx.node_executor import NodeExecutor
        
        # 加载特征定义
        fxdefs = load_fxdefs_for_set(feature_set_id)
        
        # 构建DAG
        dag = build_feature_dag(fxdefs)
        
        # 执行DAG
        executor = NodeExecutor(recording_id)
        node_outputs = executor.execute_dag(dag)
        
        # 获取执行报告
        execution_report = executor.generate_execution_report()
        
        return jsonify(execution_report)
        
    except Exception as e:
        logger.error(f"Failed to execute DAG: {e}")
        return jsonify({'error': f'Failed to execute DAG: {str(e)}'}), 500

@app.route('/api/dag_status/<int:feature_set_id>')
def get_dag_status(feature_set_id):
    """获取DAG执行状态（模拟数据）"""
    try:
        # 这里应该从数据库获取实际的执行状态
        # 暂时返回模拟数据
        mock_status = {
            "total_nodes": 6,
            "status_counts": {
                "success": 6,
                "failed": 0
            },
            "total_duration": 7.89,
            "execution_order": [
                "raw", "filter_36485993", "epoch_c7f91ab6", 
                "notch_filter_537173d2", "spectral_entropy_e3c4e4d6", 
                "spectral_entropy_e3c4e4d6__C3"
            ],
            "node_details": {
                "raw": {
                    "status": "success",
                    "duration": 0.521,
                    "pipeline_count": 2,
                    "fxdef_count": 2,
                    "error": None
                },
                "filter_36485993": {
                    "status": "success",
                    "duration": 2.436,
                    "pipeline_count": 2,
                    "fxdef_count": 2,
                    "error": None
                },
                "spectral_entropy_e3c4e4d6": {
                    "status": "success",
                    "duration": 1.728,
                    "pipeline_count": 2,
                    "fxdef_count": 2,
                    "error": None
                },
                "spectral_entropy_e3c4e4d6__C3": {
                    "status": "success",
                    "duration": 0.0,
                    "pipeline_count": 1,
                    "fxdef_count": 2,
                    "error": None
                }
            }
        }
        
        return jsonify(mock_status)
        
    except Exception as e:
        logger.error(f"Failed to get DAG status: {e}")
        return jsonify({'error': f'Failed to get DAG status: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 