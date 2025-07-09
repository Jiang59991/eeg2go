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

app = Flask(__name__)

# Database path
DB_PATH = os.path.abspath(os.path.join(os.path.dirname(__file__), "database", "eeg2go.db"))

def get_db_connection():
    """Get database connection"""
    conn = sqlite3.connect(DB_PATH)
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

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000) 