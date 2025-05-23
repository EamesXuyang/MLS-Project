from flask import Flask, request, jsonify
from urllib.parse import urljoin
import requests
from ..aggregate import aggregate_funcs
from .server_task import ServerTask
import pickle

app = Flask(__name__)


tasks = {}  

@app.route('/create_task', methods=['POST'])
def create_task():
    data = request.json
    name = data['name']
    if name in tasks:
        return jsonify({"error": "Task already exists"}), 400

    client_num = data['client_num']
    model = data['model']
    epoch = data['epoch']
    aggregate_func = aggregate_funcs[data['aggregate_func']]

    task = ServerTask(name, client_num, model, epoch, aggregate_func)
    tasks[name] = task
    return jsonify({"message": f"Task '{name}' created"}), 201

@app.route('/<task_name>/add_client', methods=['POST'])
def add_client(task_name):
    if task_name not in tasks:
        return jsonify({"error": "Task not found"}), 404

    client_url = request.json.get('client_url')
    success = tasks[task_name].add_client(client_url)
    if not success:
        return jsonify({"error": "Max clients reached or task full"}), 400
    return jsonify({"message": f"Client {client_url} added to task {task_name}"}), 200

@app.route('/<task_name>/update_client', methods=['POST'])
def update_client(task_name):
    if task_name not in tasks:
        return jsonify({"error": "Task not found"}), 404

    client_url = request.json.get('client_url')
    params_file = request.files.get('params')
    try:
        params = pickle.load(params_file.stream)  # 从文件流中反序列化
    except Exception as e:
        return jsonify({"error": f"Failed to load parameters: {str(e)}"}), 500
    
    if client_url not in tasks[task_name].clients:
        return jsonify({"error": "Client not registered"}), 400

    tasks[task_name].update_client(client_url, params)
    return jsonify({
        "message": f"Client {client_url} updated",
        "task_status": tasks[task_name].get_status()
    }), 200

@app.route('/<task_name>/status', methods=['GET'])
def task_status(task_name):
    if task_name not in tasks:
        return jsonify({"error": "Task not found"}), 404

    task = tasks[task_name]
    return jsonify({
        "task": task.name,
        "status": task.get_status(),
        "completed_epoch": task.completed_epoch
    }), 200

if __name__ == '__main__':
    import sys
    port = int(sys.argv[1]) if len(sys.argv) > 1 else 5000
    app.run(host='0.0.0.0', port=port, debug=True)
