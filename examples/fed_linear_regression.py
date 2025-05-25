import numpy as np
import sys
import os
import requests
import threading
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fudanai.tensor import Tensor
from fudanai.layers.linear import Linear
from fudanai.optimizers.optimizer import SGD
from fudanai.fed.task import Task
from fudanai.fed.util import encode_parameters, decode_parameters



def generate_data(n_samples=100):
    # 生成 y = 2x + 1 + noise 的数据
    x = np.random.randn(n_samples, 1)
    y = 2 * x + 1 + 0.1 * np.random.randn(n_samples, 1)
    return x, y

def train_local_func(epochs, model, trainloader):    
    # 创建优化器
    optimizer = SGD(model.parameters().values(), lr=0.001)  # 降低学习率
    
    # 训练循环
    batch_size = 32  # 增加批次大小
    x_train, y_train = trainloader(n_samples=1000)

    for epoch in range(epochs):
        total_loss = 0
        n_batches = len(x_train) // batch_size
        
        for i in range(n_batches):
            # 获取批次数据
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            x_batch = Tensor(x_train[start_idx:end_idx])
            y_batch = Tensor(y_train[start_idx:end_idx])
            
            # 前向传播
            pred = model.forward(x_batch)
            
            # 计算损失（MSE）
            loss = ((pred - y_batch) ** 2).sum() / batch_size
            total_loss += loss.data
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        avg_loss = total_loss / n_batches
    
# python -m fudanai.fed.server.server
# python -m fudanai.fed.client.client

def run_task():
    model = Linear(1, 1)
    task = Task('fed_linear_regression', server="http://127.0.0.1:5000/", client_port=5001, model=model, train_local_func=train_local_func, trainloader=generate_data)
    task.run()
    params = model.parameters()
    for param in params:
        print(f'{param}: {params[param].data}')


model = Linear(1, 1)
requests.post('http://127.0.0.1:5000/create_task', json={"name": 'fed_linear_regression', 'client_num': 3, 'epochs': 50, 'aggregate_func': 'avg', 'params': encode_parameters(model.parameters()), 'client': "http://127.0.0.1:5001"})

thread0 = threading.Thread(target=run_task)
thread1 = threading.Thread(target=run_task)
thread2 = threading.Thread(target=run_task)

thread0.start()
thread1.start()
thread2.start()

thread0.join()
thread1.join()
thread2.join()

requests.delete("http://127.0.0.1:5000/delete_task", params={"name": 'fed_linear_regression'})