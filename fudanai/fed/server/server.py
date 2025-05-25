from flask import Flask, request, jsonify, render_template
from urllib.parse import urljoin
import requests
from ..aggregate import aggregate_funcs
from .server_task import ServerTask
import pickle
from ..util import encode_parameters, decode_parameters
app = Flask(__name__)


tasks = {}  

@app.route('/create_task', methods=['POST'])
def create_task():
    data = request.json
    name = data.get('name')
    if name in tasks:
        return jsonify({"error": "Task already exists"}), 400

    client_num = data.get('client_num')
    epochs = data.get('epochs')
    aggregate_func = aggregate_funcs[data.get('aggregate_func')]
    params = decode_parameters(data.get('params'))

    task = ServerTask(name, client_num, epochs, aggregate_func, params)
    tasks[name] = task
    return jsonify({"message": f"Task '{name}' created"}), 201

@app.route('/<task_name>/add_client', methods=['POST'])
def add_client(task_name: str):
    if task_name not in tasks:
        return jsonify({"error": "Task not found"}), 404

    data = request.json
    client_url = data.get('client')
    id = data.get('id')
    success = tasks[task_name].add_client((client_url, id))
    if not success:
        return jsonify({"error": "Max clients reached or task full"}), 400
    return jsonify({"message": f"Client {client_url}{id} added to task {task_name}"}), 200

@app.route('/<task_name>/update_client', methods=['POST'])
def update_client(task_name: str):
    if task_name not in tasks:
        return jsonify({"error": "Task not found"}), 404

    data = request.json
    client_url = data.get('client')
    id = data.get('id')

    if (client_url, id) not in tasks[task_name].clients:
        return jsonify({"error": "Client not registered"}), 400

    params = decode_parameters(data.get('params'))

    tasks[task_name].update_client((client_url, id), params)
    return jsonify({
        "message": f"Client {client_url}({id}) updated",
        "task_status": tasks[task_name].get_status()
    }), 200

@app.route('/<task_name>/status', methods=['GET'])
def task_status(task_name: str):
    if task_name not in tasks:
        return jsonify({"error": "Task not found"}), 404

    return jsonify(tasks[task_name].get_snapshot()), 200

@app.route('/delete_task', methods=['DELETE'])
def delete_task():
    task_name = request.args.get('name')
    if task_name not in tasks:
        return jsonify({"error": "Task not found"}), 404

    del tasks[task_name]
    return jsonify({"message": f"Task '{task_name}' deleted"}), 200

@app.route('/list_tasks', methods=['GET'])
def list_tasks():
    return jsonify({"tasks": list(tasks.keys())}), 200

@app.route('/')
def dashboard():
    return render_template('dashboard.html')

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
