from flask import Flask, request, jsonify
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
    name = data['name']
    if name in tasks:
        return jsonify({"error": "Task already exists"}), 400

    client_num = data['client_num']
    epoch = data['epoch']
    aggregate_func = aggregate_funcs[data['aggregate_func']]
    params = decode_parameters(data['params'])

    task = ServerTask(name, client_num, epoch, aggregate_func, params)
    tasks[name] = task
    return jsonify({"message": f"Task '{name}' created"}), 201

@app.route('/<task_name>/add_client', methods=['POST'])
def add_client(task_name):
    if task_name not in tasks:
        return jsonify({"error": "Task not found"}), 404

    client_url = request.json.get('client')
    success = tasks[task_name].add_client(client_url)
    if not success:
        return jsonify({"error": "Max clients reached or task full"}), 400
    return jsonify({"message": f"Client {client_url} added to task {task_name}"}), 200

@app.route('/<task_name>/update_client', methods=['POST'])
def update_client(task_name):
    if task_name not in tasks:
        return jsonify({"error": "Task not found"}), 404

    data = request.json
    client_url = data['client']

    if client_url not in tasks[task_name].clients:
        return jsonify({"error": "Client not registered"}), 400

    params = data['params']

    params = decode_parameters(params)

    tasks[task_name].update_client(client_url, params)
    return jsonify({
        "message": f"Client {client_url} updated",
        "task_status": tasks[task_name].get_status().value
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
    app.run(host='0.0.0.0', port=5000)
