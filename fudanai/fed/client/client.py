from flask import Flask, request, jsonify
import pickle
from io import BytesIO
import threading
import requests
from .client_task import ClientTask, TaskStatus
import threading
from urllib.parse import urljoin

app = Flask(__name__)

tasks = {}
conditions = {}
lock = threading.Lock()

@app.route("/register_task", methods=['POST'])
def register_task():
    data = request.json
    name = data['name']
    if name in tasks:
        return jsonify({"error": "Task already exists"}), 400
    
    server = data['server']
    task = ClientTask(name, server)
    tasks[name] = task
    return jsonify({"message": f"Task '{name}' registered"}), 200


@app.route("/<task_name>/send_init_params", methods=['POST'])
def send_init_params(task_name):
    if task_name not in tasks:
        return jsonify({"error": "Task not found"}), 404

    data = request.json
    epochs = data['epochs']
    params = request.files['params'] 
    task = tasks[task_name]
    with lock:
        task.init_params(epochs, params)
        cond  = conditions.get(task_name)

    if cond:
        with cond:
            cond.notify_all()

    return jsonify({"message": f"Task '{task_name}' parameters initialized"}), 200

@app.route("/<task_name>/get_init_params", methods=['GET'])
def get_init_params(task_name):
    if task_name not in tasks:
        return jsonify({"error": "Task not found"}), 404
    task = tasks[task_name]
    with lock:
        init_params = task.get_init_params()
        if init_params is not None:
            epochs, params = init_params
            task.status = TaskStatus.RUNNING
            task.clean_params()
            return jsonify({"epochs": epochs, "params": params}), 200
        cond = conditions.setdefault(task_name, threading.Condition())

    with cond:
        cond.wait()
    
    with lock:
        init_params = task.get_init_params()
        epochs, params = init_params
        task.status = TaskStatus.WAITING
        task.clean_params()
        return jsonify({"epochs": epochs, "params": params}), 200
    

@app.route("/<task_name>/send_params_to_server")
def send_params_to_server(task_name):
    if task_name not in tasks:
        return jsonify({"error": "Task not found"}), 404
    task = tasks.get(task_name)
    task.status = TaskStatus.WAITING
    params = request.files['params']
    requests.post(urljoin(task.server, f'/{task_name}/update_params'), files={'params': params})
    return jsonify({"message": f"Task '{task_name}' parameters sent to server"}), 200


@app.route('<taks_name>/send_params_to_client')
def send_params_to_client(task_name):
    if task_name not in tasks:
        return jsonify({"error": "Task not found"}), 404
    task = tasks.get(task_name)
    params = request.json['params']
    with lock():
        task.set_params(params)
        cond = conditions.get(task_name)
    
    if cond:
        with cond:
            cond.notify_all()



@app.route('<task_name>/get_params_from_server')
def get_params_from_server(task_name):
    if task_name not in tasks:
        return jsonify({"error": "Task not found"}), 404
    task = tasks.get(task_name)
    with lock():
        params = task.get_params()
        if params is not None:
            task.status = TaskStatus.RUNNING
            return jsonify({"params": params}), 200
        cond = conditions.setdefault(task_name, threading.Condition())
    with cond:
        cond.wait()

    with lock():
        params = task.get_params()
        task.status = TaskStatus.RUNNING
        return jsonify({"params": params}), 200




