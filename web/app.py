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

app = Flask(__name__)

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DATABASE_PATH)
    conn.row_factory = sqlite3.Row
    return conn

@app.route('/')
def index():
    """Home page"""
    return render_template('index.html')

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
    """Get feature values"""
    recording_id = request.args.get('recording_id', type=int)
    feature_set_id = request.args.get('feature_set_id', type=int)
    
    if not recording_id or not feature_set_id:
        return jsonify({'error': 'Missing required parameters'}), 400
    
    conn = get_db_connection()
    
    # Get all feature definitions in the feature set
    fxdefs = conn.execute('''
        SELECT f.* FROM fxdef f
        JOIN feature_set_items fsi ON f.id = fsi.fxdef_id
        WHERE fsi.feature_set_id = ?
    ''', (feature_set_id,)).fetchall()
    
    # Get feature values
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
    try:
        pipeline_id = add_pipeline(pipeline_def)
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 