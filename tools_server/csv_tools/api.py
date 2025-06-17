from flask import Flask, request, jsonify, send_file
import pandas as pd
import threading
import sys
import io
import logging

app = Flask(__name__)

# Thread-safe CSV storage
csv_lock = threading.Lock()
datasets = {}

@app.route('/api/csv/create', methods=['POST'])
def create_csv():
    print("Received request to create CSV dataset")
    data = request.get_json()
    name = data.get('name')
    columns = data.get('columns', [])
    rows = data.get('rows', [])
    
    if not name:
        print("Dataset name is missing in create_csv")
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
            print(f"Dataset '{name}' created with shape {df.shape}")
            return jsonify({
                "name": name,
                "shape": df.shape,
                "columns": df.columns.tolist()
            })
        except Exception as e:
            print(f"Error creating dataset '{name}': {e}")
            return jsonify({"error": str(e)}), 500

@app.route('/api/csv/load', methods=['POST'])
def load_csv():
    print("Received request to load CSV from file")
    if 'file' not in request.files:
        print("No file provided in load_csv")
        return jsonify({"error": "No file provided"}), 400
    
    file = request.files['file']
    name = request.form.get('name') or file.filename.rsplit('.', 1)[0]
    
    if not file.filename:
        print("No file selected in load_csv")
        return jsonify({"error": "No file selected"}), 400
    
    with csv_lock:
        try:
            df = pd.read_csv(io.StringIO(file.read().decode('utf-8')))
            datasets[name] = df
            print(f"Dataset '{name}' loaded from file with shape {df.shape}")
            return jsonify({
                "name": name,
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "preview": df.head().to_dict('records')
            })
        except Exception as e:
            print(f"Error loading dataset '{name}': {e}")
            return jsonify({"error": str(e)}), 500

@app.route('/api/csv/info/<name>', methods=['GET'])
def get_info(name):
    print(f"Received request for info of dataset '{name}'")
    with csv_lock:
        if name not in datasets:
            print(f"Dataset '{name}' not found in get_info")
            return jsonify({"error": "Dataset not found"}), 404
        
        df = datasets[name]
        print(f"Returning info for dataset '{name}'")
        return jsonify({
            "name": name,
            "shape": df.shape,
            "columns": df.columns.tolist(),
            "dtypes": {col: str(dtype) for col, dtype in df.dtypes.items()}
        })

@app.route('/api/csv/data/<name>', methods=['GET'])
def get_data(name):
    print(f"Received request for data of dataset '{name}'")
    with csv_lock:
        if name not in datasets:
            print(f"Dataset '{name}' not found in get_data")
            return jsonify({"error": "Dataset not found"}), 404
        
        df = datasets[name]
        limit = request.args.get('limit', type=int)
        offset = request.args.get('offset', 0, type=int)
        
        if limit:
            data = df.iloc[offset:offset+limit].to_dict('records')
        else:
            data = df.iloc[offset:].to_dict('records')

        print(f"Returning data for dataset '{name}' (offset={offset}, limit={limit})")
        return jsonify({
            "data": data,
            "total_rows": len(df)
        })

@app.route('/api/csv/add_row/<name>', methods=['POST'])
def add_row(name):
    print(f"Received request to add row to dataset '{name}'")
    with csv_lock:
        if name not in datasets:
            print(f"Dataset '{name}' not found in add_row")
            return jsonify({"error": "Dataset not found"}), 404
        
        data = request.get_json()
        row_data = data.get('row')
        
        if not row_data:
            print("Row data is missing in add_row")
            return jsonify({"error": "Row data is required"}), 400
        
        try:
            df = datasets[name]
            new_row = pd.DataFrame([row_data])
            datasets[name] = pd.concat([df, new_row], ignore_index=True)
            print(f"Row added to dataset '{name}', new shape: {datasets[name].shape}")
            return jsonify({"shape": datasets[name].shape})
        except Exception as e:
            print(f"Error adding row to dataset '{name}': {e}")
            return jsonify({"error": str(e)}), 500

@app.route('/api/csv/update_row/<name>/<int:index>', methods=['PUT'])
def update_row(name, index):
    print(f"Received request to update row {index} in dataset '{name}'")
    with csv_lock:
        if name not in datasets:
            print(f"Dataset '{name}' not found in update_row")
            return jsonify({"error": "Dataset not found"}), 404
        
        df = datasets[name]
        if index >= len(df):
            print(f"Row index {index} out of range in update_row for dataset '{name}'")
            return jsonify({"error": "Row index out of range"}), 400
        
        data = request.get_json()
        row_data = data.get('row')
        
        if not row_data:
            print("Row data is missing in update_row")
            return jsonify({"error": "Row data is required"}), 400
        
        try:
            for column, value in row_data.items():
                if column in df.columns:
                    df.loc[index, column] = value
            print(f"Row {index} updated in dataset '{name}'")
            return jsonify({"updated_row": df.iloc[index].to_dict()})
        except Exception as e:
            print(f"Error updating row {index} in dataset '{name}': {e}")
            return jsonify({"error": str(e)}), 500

@app.route('/api/csv/delete_row/<name>/<int:index>', methods=['DELETE'])
def delete_row(name, index):
    print(f"Received request to delete row {index} from dataset '{name}'")
    with csv_lock:
        if name not in datasets:
            print(f"Dataset '{name}' not found in delete_row")
            return jsonify({"error": "Dataset not found"}), 404
        
        df = datasets[name]
        if index >= len(df):
            print(f"Row index {index} out of range in delete_row for dataset '{name}'")
            return jsonify({"error": "Row index out of range"}), 400
        
        try:
            datasets[name] = df.drop(index).reset_index(drop=True)
            print(f"Row {index} deleted from dataset '{name}', new shape: {datasets[name].shape}")
            return jsonify({"shape": datasets[name].shape})
        except Exception as e:
            print(f"Error deleting row {index} from dataset '{name}': {e}")
            return jsonify({"error": str(e)}), 500

@app.route('/api/csv/export/<name>', methods=['GET'])
def export_csv(name):
    print(f"Received request to export dataset '{name}' as CSV")
    with csv_lock:
        if name not in datasets:
            print(f"Dataset '{name}' not found in export_csv")
            return jsonify({"error": "Dataset not found"}), 404
        
        try:
            output = io.StringIO()
            datasets[name].to_csv(output, index=False)
            output.seek(0)
            print(f"Dataset '{name}' exported as CSV")
            return send_file(
                io.BytesIO(output.getvalue().encode('utf-8')),
                mimetype='text/csv',
                as_attachment=True,
                download_name=f'{name}.csv'
            )
        except Exception as e:
            print(f"Error exporting dataset '{name}': {e}")
            return jsonify({"error": str(e)}), 500

@app.route('/api/csv/list', methods=['GET'])
def list_datasets():
    print("Received request to list all datasets")
    with csv_lock:
        dataset_list = []
        for name, df in datasets.items():
            dataset_list.append({
                "name": name,
                "shape": df.shape,
                "columns": df.columns.tolist()
            })
        print(f"Returning list of {len(dataset_list)} datasets")
        return jsonify({"datasets": dataset_list})

@app.route('/api/csv/delete/<name>', methods=['DELETE'])
def delete_dataset(name):
    print(f"Received request to delete dataset '{name}'")
    with csv_lock:
        if name in datasets:
            del datasets[name]
            print(f"Dataset '{name}' deleted")
            return jsonify({"message": "Dataset deleted"})
        else:
            print(f"Dataset '{name}' not found in delete_dataset")
            return jsonify({"error": "Dataset not found"}), 404

if __name__ == '__main__':
    port = 5001
    if len(sys.argv) > 1:
        try:
            port = int(sys.argv[1])
        except ValueError:
            print(f"Invalid port: {sys.argv[1]}. Using default port 5001.")

    app.run(host='0.0.0.0', port=port, debug=True)
