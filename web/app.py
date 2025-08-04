from flask import Flask, render_template, request, jsonify, send_file, send_from_directory
import sqlite3
import os
import json
import pandas as pd
from datetime import datetime
import tempfile
import zipfile
import logging
from eeg2fx.featureset_fetcher import run_feature_set
from eeg2fx.featureset_grouping import load_fxdefs_for_set
from .utils.pipeline_visualizer import get_pipeline_visualization_data
from .config import DATABASE_PATH
from database.add_pipeline import add_pipeline, STEP_REGISTRY, validate_pipeline
from database.add_fxdef import add_fxdefs
from database.add_featureset import add_featureset
from web.api.task_api import task_api

# 设置日志
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = Flask(__name__)

# 注册任务API蓝图
app.register_blueprint(task_api)

# ====== 静态文件路由和 MIME 类型设置 ======
@app.route('/static/js/<path:filename>')
def serve_js(filename):
    """Serve JavaScript files with correct MIME type for ES6 modules"""
    return send_from_directory('static/js', filename, mimetype='application/javascript')

@app.route('/static/js/modules/<path:filename>')
def serve_js_modules(filename):
    """Serve JavaScript module files with correct MIME type"""
    return send_from_directory('static/js/modules', filename, mimetype='application/javascript')

@app.route('/static/css/<path:filename>')
def serve_css(filename):
    """Serve CSS files with correct MIME type"""
    return send_from_directory('static/css', filename, mimetype='text/css')

@app.route('/static/images/<path:filename>')
def serve_images(filename):
    """Serve image files with correct MIME type"""
    import os
    _, ext = os.path.splitext(filename)
    mime_types = {
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml',
        '.ico': 'image/x-icon'
    }
    mimetype = mime_types.get(ext.lower(), 'image/png')
    return send_from_directory('static/images', filename, mimetype=mimetype)

@app.route('/static/<path:filename>')
def serve_static(filename):
    """Serve static files with correct MIME types"""
    import os
    _, ext = os.path.splitext(filename)
    mime_types = {
        '.js': 'application/javascript',
        '.css': 'text/css',
        '.html': 'text/html',
        '.htm': 'text/html',
        '.png': 'image/png',
        '.jpg': 'image/jpeg',
        '.jpeg': 'image/jpeg',
        '.gif': 'image/gif',
        '.svg': 'image/svg+xml',
        '.ico': 'image/x-icon',
        '.json': 'application/json',
        '.xml': 'application/xml',
        '.pdf': 'application/pdf',
        '.txt': 'text/plain',
        '.woff': 'font/woff',
        '.woff2': 'font/woff2',
        '.ttf': 'font/ttf',
        '.eot': 'application/vnd.ms-fontobject'
    }
    
    mimetype = mime_types.get(ext.lower(), 'application/octet-stream')
    
    return send_from_directory('static', filename, mimetype=mimetype)

# ====== 其他现有代码 ======
# 任务处理现在由Celery Worker处理，不需要在Web应用中初始化

def init_app():
    """初始化应用"""
    # 任务处理由Celery Worker处理
    pass

# 在应用启动时初始化
init_app()

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

# 在 app = Flask(__name__) 之后，其他路由之前添加


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

@app.route('/api/experiment_types')
def get_experiment_types():
    """Get available experiment types with parameter definitions"""
    try:
        from feature_mill.experiments import AVAILABLE_EXPERIMENTS
        
        experiment_types = []
        for exp_type, exp_info in AVAILABLE_EXPERIMENTS.items():
            experiment_types.append({
                'type': exp_type,
                'name': exp_info['name'],
                'description': exp_info['description'],
                'parameters': exp_info.get('parameters', {})
            })
        
        return jsonify(experiment_types)
    except Exception as e:
        import logging
        logging.error(f"Failed to get experiment types: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/experiment_parameters/<experiment_type>')
def get_experiment_parameters(experiment_type):
    """Get parameter definitions for a specific experiment type"""
    try:
        from feature_mill.experiments import get_experiment_parameters
        
        parameters = get_experiment_parameters(experiment_type)
        return jsonify(parameters)
    except Exception as e:
        import logging
        logging.error(f"Failed to get experiment parameters: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/run_experiment', methods=['POST'])
def run_experiment_api():
    """Run an experiment with parameters"""
    try:
        data = request.get_json()
        
        experiment_type = data.get('experiment_type')
        dataset_id = data.get('dataset_id')
        feature_set_id = data.get('feature_set_id')
        parameters = data.get('parameters', {})
        
        if not all([experiment_type, dataset_id, feature_set_id]):
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # 确保数据类型正确
        dataset_id = int(dataset_id)
        feature_set_id = int(feature_set_id)
        
        # 导入必要的模块
        from web.api.task_api import task_manager
        from task_queue.models import Task
        
        # Create task for experiment execution
        task = Task(
            task_type='experiment',
            parameters={
                'experiment_type': experiment_type,
                'dataset_id': dataset_id,
                'feature_set_id': feature_set_id,
                'parameters': parameters
            },
            dataset_id=dataset_id,
            feature_set_id=feature_set_id,
            experiment_type=experiment_type
        )
        
        task_id = task_manager.create_task(task)
        
        return jsonify({
            'success': True,
            'task_id': task_id,
            'message': f'Experiment task created with ID: {task_id}'
        })
        
    except Exception as e:
        import logging
        logging.error(f"Failed to create experiment task: {e}")
        return jsonify({'error': str(e)}), 500

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
        import logging
        logging.error(f"Failed to get experiment status: {e}")
        return jsonify({'error': f'Failed to get experiment status: {str(e)}'}), 500

@app.route('/api/start_feature_extraction', methods=['POST'])
def start_feature_extraction_api():
    """异步特征提取API - 创建任务而不是直接执行"""
    try:
        data = request.get_json()
        dataset_id = data.get('dataset_id')
        feature_set_id = data.get('feature_set_id')
        
        if not dataset_id or not feature_set_id:
            return jsonify({'error': 'Missing required parameters'}), 400
        
        # 确保数据类型正确
        dataset_id = int(dataset_id)
        feature_set_id = int(feature_set_id)
        
        print(f"Creating feature extraction task with dataset_id={dataset_id}, feature_set_id={feature_set_id}")
        
        # 创建任务而不是直接执行
        from web.api.task_api import task_manager
        from task_queue.models import Task
        
        task = Task('feature_extraction', {
            'dataset_id': dataset_id,
            'feature_set_id': feature_set_id
        }, dataset_id=dataset_id, feature_set_id=feature_set_id)
        
        task_id = task_manager.create_task(task)
        
        print(f"Task created with ID: {task_id}")
        
        return jsonify({
            'task_id': task_id,
            'status': 'pending',
            'message': 'Feature extraction task created successfully'
        }), 202
        
    except Exception as e:
        import logging
        logging.error(f"Failed to create feature extraction task: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/feature_extraction_tasks')
def get_feature_extraction_tasks():
    """Get all feature extraction tasks"""
    try:
        conn = get_db_connection()
        tasks = conn.execute('''
            SELECT t.*, d.name as dataset_name, fs.name as feature_set_name
            FROM tasks t
            LEFT JOIN datasets d ON t.dataset_id = d.id
            LEFT JOIN feature_sets fs ON t.feature_set_id = fs.id
            WHERE t.task_type = 'feature_extraction'
            ORDER BY t.created_at DESC
        ''').fetchall()
        conn.close()
        
        return jsonify([dict(task) for task in tasks])
        
    except Exception as e:
        import logging
        logging.error(f"Failed to get feature extraction tasks: {e}")
        return jsonify({'error': f'Failed to get feature extraction tasks: {str(e)}'}), 500

@app.route('/api/feature_extraction_status/<int:task_id>')
def get_feature_extraction_status(task_id):
    """Get feature extraction task status"""
    try:
        print(f"Getting feature extraction status for task ID: {task_id}")
        
        conn = get_db_connection()
        
        # 简化查询，先只获取基本信息
        task = conn.execute('SELECT * FROM tasks WHERE id = ?', (task_id,)).fetchone()
        conn.close()
        
        if not task:
            print(f"Task {task_id} not found")
            return jsonify({'error': 'Task not found'}), 404
        
        print(f"Task found, ID: {task[0]}, Type: {task[1]}, Status: {task[2]}")
        
        # 安全地处理JSON字段
        def safe_json_loads(json_str):
            """安全地解析JSON字符串"""
            if not json_str:
                return None
            try:
                parsed = json.loads(json_str)
                
                # 如果结果太大，创建摘要
                if isinstance(parsed, dict) and len(json_str) > 10000:  # 超过10KB
                    print(f"JSON对象过大 ({len(json_str)} 字符)，创建摘要")
                    summary = {}
                    for key, value in parsed.items():
                        if isinstance(value, dict):
                            summary[key] = {
                                "type": "object",
                                "size": len(str(value)),
                                "keys": list(value.keys()) if isinstance(value, dict) else None
                            }
                        elif isinstance(value, list):
                            summary[key] = {
                                "type": "array",
                                "length": len(value),
                                "size": len(str(value))
                            }
                        else:
                            summary[key] = {
                                "type": type(value).__name__,
                                "value": str(value)[:100] if len(str(value)) > 100 else value
                            }
                    return {
                        "summary": summary,
                        "original_size": len(json_str),
                        "note": "Large object, showing summary only"
                    }
                else:
                    return parsed
                    
            except Exception as e:
                print(f"JSON解析失败: {e}")
                # 只显示前200个字符
                preview = str(json_str)[:200] + "..." if len(str(json_str)) > 200 else str(json_str)
                return {"parse_error": str(e), "raw_content_preview": preview}
        
        # 构建基本响应
        response_data = {
            'id': task[0],
            'task_type': task[1],
            'status': task[2],
            'parameters': safe_json_loads(task[3]),
            'result': safe_json_loads(task[4]),
            'error_message': task[5],
            'created_at': task[6],
            'started_at': task[7],
            'completed_at': task[8],
            'priority': task[9],
            'dataset_id': task[10],
            'feature_set_id': task[11],
            'experiment_type': task[12],
            'progress': task[13],
            'processed_count': task[14],
            'total_count': task[15],
            'notes': task[16]
        }
        
        print(f"Response data prepared successfully")
        return jsonify(response_data)
        
    except Exception as e:
        import logging
        import traceback
        error_msg = f"Failed to get feature extraction status: {str(e)}"
        print(error_msg)
        print(f"Traceback: {traceback.format_exc()}")
        logging.error(error_msg)
        return jsonify({'error': error_msg}), 500

@app.route('/api/download_extraction_result/<int:task_id>')
def download_extraction_result(task_id):
    """Download feature extraction result file"""
    try:
        conn = get_db_connection()
        task = conn.execute('''
            SELECT result FROM tasks WHERE id = ?
        ''', (task_id,)).fetchone()
        conn.close()
        
        if not task or not task['result']:
            return jsonify({'error': 'Result file not found'}), 404
        
        result_data = json.loads(task['result'])
        if not result_data.get('output_dir'):
            return jsonify({'error': 'Output directory not found in result'}), 404
        
        output_dir = result_data['output_dir']
        if not os.path.exists(output_dir):
            return jsonify({'error': 'Output directory does not exist'}), 404
        
        # 尝试查找CSV文件，如果没有则返回所有文件
        csv_files = [f for f in os.listdir(output_dir) if f.endswith('.csv')]
        
        if not csv_files:
            return jsonify({'error': 'No CSV file found in output directory'}), 404
        
        # 假设只有一个CSV文件，或者选择第一个
        filename = csv_files[0]
        file_path = os.path.join(output_dir, filename)
        
        return send_file(file_path, as_attachment=True, download_name=f"extraction_{task_id}.csv")
        
    except Exception as e:
        import logging
        logging.error(f"Failed to download extraction result: {e}")
        return jsonify({'error': f'Failed to download extraction result: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 