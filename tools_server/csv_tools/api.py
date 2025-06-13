from flask import Flask, request, jsonify, send_file
import pandas as pd
import threading
import sys
import io

app = Flask(__name__)

# Thread-safe CSV storage
csv_lock = threading.Lock()
datasets = {}

@app.route('/api/csv/create', methods=['POST'])
def create_csv():
    """Create a new CSV dataset"""
    data = request.get_json()
    name = data.get('name')
    columns = data.get('columns', [])
    rows = data.get('rows', [])
    
    if not name:
        return jsonify({"error": "Dataset name is required"}), 400
    
    with csv_lock:
        try:
            if columns and rows:
                df = pd.DataFrame(rows, columns=columns)
            elif columns:
                df = pd.DataFrame(columns=columns)
            elif rows:
                df = pd.DataFrame(rows)
            else:
                df = pd.DataFrame()
            
            datasets[name] = df
            return jsonify({
                "name": name,
                "shape": df.shape,
                "columns": df.columns.tolist()
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/api/csv/load', methods=['POST'])
def load_csv():
    """Load CSV from uploaded file"""
    if 'file' not in request.files:
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    name = request.form.get('name') or file.filename.rsplit('.', 1)[0]
    
    if not file.filename:
        return jsonify({"error": "No file selected"}), 400
    
    with csv_lock:
        try:
            df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
            datasets[name] = df
            
            return jsonify({
                "name": name,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "preview": df.head().to_dict('records')
            })
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/api/csv/info/<name>', methods=['GET'])
def get_info(name):
    """Get dataset information"""
    with csv_lock:
        if name not in datasets:
            return jsonify({"error": "Dataset not found"}), 404
        
        df = datasets[name]
        return jsonify({
            "name": name,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        })

@app.route('/api/csv/data/<name>', methods=['GET'])
def get_data(name):
    """Get dataset data with optional pagination"""
    with csv_lock:
        if name not in datasets:
            return jsonify({"error": "Dataset not found"}), 404
        
        df = datasets[name]
        limit = request.args.get('limit', type=int)
        offset = request.args.get('offset', 0, type=int)
        
        if limit:
            data = df.iloc[offset:offset+limit].to_dict('records')
        else:
            data = df.iloc[offset:].to_dict('records')
        
        return jsonify({
            "data": data,
            "total_rows": len(df)
        })

@app.route('/api/csv/add_row/<name>', methods=['POST'])
def add_row(name):
    """Add a row to dataset"""
    with csv_lock:
        if name not in datasets:
            return jsonify({"error": "Dataset not found"}), 404
        
        data = request.get_json()
        row_data = data.get('row')
        
        if not row_data:
            return jsonify({"error": "Row data is required"}), 400
        
        try:
            df = datasets[name]
            new_row = pd.DataFrame([row_data])
            datasets[name] = pd.concat([df, new_row], ignore_index=True)
            
            return jsonify({"shape": datasets[name].shape})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/api/csv/update_row/<name>/<int:index>', methods=['PUT'])
def update_row(name, index):
    """Update a specific row"""
    with csv_lock:
        if name not in datasets:
            return jsonify({"error": "Dataset not found"}), 404
        
        df = datasets[name]
        if index >= len(df):
            return jsonify({"error": "Row index out of range"}), 400
        
        data = request.get_json()
        row_data = data.get('row')
        
        if not row_data:
            return jsonify({"error": "Row data is required"}), 400
        
        try:
            for column, value in row_data.items():
                if column in df.columns:
                    df.loc[index, column] = value
            
            return jsonify({"updated_row": df.iloc[index].to_dict()})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/api/csv/delete_row/<name>/<int:index>', methods=['DELETE'])
def delete_row(name, index):
    """Delete a specific row"""
    with csv_lock:
        if name not in datasets:
            return jsonify({"error": "Dataset not found"}), 404
        
        df = datasets[name]
        if index >= len(df):
            return jsonify({"error": "Row index out of range"}), 400
        
        try:
            datasets[name] = df.drop(index).reset_index(drop=True)
            return jsonify({"shape": datasets[name].shape})
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/api/csv/export/<name>', methods=['GET'])
def export_csv(name):
    """Export dataset as CSV"""
    with csv_lock:
        if name not in datasets:
            return jsonify({"error": "Dataset not found"}), 404
        
        try:
            output = io.StringIO()
            datasets[name].to_csv(output, index=False)
            output.seek(0)
            
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'{name}.csv'
            )
        except Exception as e:
            return jsonify({"error": str(e)}), 500

@app.route('/api/csv/list', methods=['GET'])
def list_datasets():
    """List all datasets"""
    with csv_lock:
        dataset_list = []
        for name, df in datasets.items():
            dataset_list.append({
                "name": name,
                "shape": df.shape,
                "columns": df.columns.tolist()
            })
        
        return jsonify({"datasets": dataset_list})

@app.route('/api/csv/delete/<name>', methods=['DELETE'])
def delete_dataset(name):
    """Delete a dataset"""
    with csv_lock:
        if name in datasets:
            del datasets[name]
            return jsonify({"message": "Dataset deleted"})
        else:
            return jsonify({"error": "Dataset not found"}), 404

if __name__ == '__main__':
    port = 5001
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}. Using default port 5001.")
    
    app.run(host='0.0.0.0', port=port, debug=True)
