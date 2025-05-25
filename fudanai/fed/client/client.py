from flask import Flask, request, jsonify
import pickle
from io import BytesIO
import threading
import requests
from .client_task import ClientTask, TaskStatus
import threading
from urllib.parse import urljoin
from ..util import encode_parameters, decode_parameters

app = Flask(__name__)

tasks = {}
conditions = {}
lock = threading.Lock()

@app.route("/register_task", methods=['POST'])
def register_task():
    data = request.json

    id = data.get('id')
    if id in tasks:
        return jsonify({"error": "Task already exists"}), 400

    name = data.get('name')
    server = data.get('server')
    client = data.get('client')

    task = ClientTask(name, server, client, id)
    tasks[id] = task

    requests.post(urljoin(server, f'/{name}/add_client'), json={'client': task.client, 'id': id})
    return jsonify({"message": f"Task {name}({id}) registered"}), 200


@app.route("/send_init_params", methods=['POST'])
def send_init_params():
    data = request.json
    id = data.get('id')
    if id not in tasks:
        return jsonify({"error": "Task not found"}), 404

    epochs = data['epochs']
    params = data['params']

    task = tasks[id]
    with lock:
        task.init_params(epochs, params)
        cond  = conditions.get(id)

    if cond:
        with cond:
            cond.notify_all()

    return jsonify({"message": f"Task {task.name}({task.task_id}) parameters initialized"}), 200

@app.route("/get_init_params", methods=['GET'])
def get_init_params():
    id = request.args.get('id')
    if id not in tasks:
        return jsonify({"error": "Task not found"}), 404
    task = tasks[id]
    with lock:
        init_params = task.get_init_params()
        if init_params is not None:
            epochs, params = init_params
            task.clean_params()
            return jsonify({"epochs": epochs, "params": params}), 200
        cond = conditions.setdefault(id, threading.Condition())

    with cond:
        cond.wait()
    
    with lock:
        init_params = task.get_init_params()
        epochs, params = init_params
        task.clean_params()
        return jsonify({"epochs": epochs, "params": params}), 200
    

@app.route("/send_params_to_server", methods=['POST'])
def send_params_to_server():
    data = request.json
    id = data.get('id')
    if id not in tasks:
        return jsonify({"error": "Task not found"}), 404
    task = tasks.get(id)

    task_name = task.name
    task.status = TaskStatus.WAITING
    
    params = data.get('params')
    requests.post(urljoin(task.server, f'/{task_name}/update_client'), json={'params': params, 'id': id, 'client': task.client})
    return jsonify({"message": f"Task '{task_name}' parameters sent to server"}), 200


@app.route('/send_params_to_client', methods=['POST'])
def send_params_to_client():
    data = request.json
    id = data.get('id')
    if id not in tasks:
        return jsonify({"error": "Task not found"}), 404
    task = tasks.get(id)
    task_name = task.name

    params = data.get('params')
    with lock:
        task.set_params(params)
        cond = conditions.get(id)
    
    if cond:
        with cond:
            cond.notify_all()
    return jsonify({"message": f"Task {task_name}({id}) parameters sent to client"}), 200


@app.route('/get_params_from_server', methods=['GET'])
def get_params_from_server():
    id = request.args.get('id')
    if id not in tasks:
        return jsonify({"error": "Task not found"}), 404
    task = tasks.get(id)

    with lock:
        params = task.get_params()
        if params is not None:
            task.clean_params()
            return jsonify({"params": params}), 200
        cond = conditions.setdefault(id, threading.Condition())

    with cond:
        cond.wait()

    with lock:
        params = task.get_params()
        task.clean_params()
        return jsonify({"params": params}), 200


if __name__ == '__main__':
    app.run(port=5001)