import threading
import requests
from urllib.parse import urljoin
from enum import Enum
from ..util import encode_parameters, decode_parameters
from typing import Callable, Dict, Tuple
from ...tensor import Tensor
from ..aggregate import aggregate_average, median_aggregate, prox_aggregate, trimmed_mean_aggregate, weighed_avg_aggregate

class ClientStatus(Enum):
    INIT = 0
    RUNNING = 1
    FINISHED = 2

class TaskStatus(Enum):
    INIT = 0
    WAITING = 1
    AGGREGATING = 2
    FINISHED = 3

class ServerTask:
    def __init__(self, name: str, client_num: int, epochs: int, aggregate_func: Callable[[Dict[str, Tensor]], Dict[str, str]], params: Dict[str, Tensor]):
        self.name = name
        self.client_num = client_num
        self.params = params
        self.epochs = epochs
        self.completed_epoch = 0
        self.aggregate_func = aggregate_func
        self.clients = {}
        self.client_params = {}
        self.status = TaskStatus.INIT
        self.lock = threading.Lock()

    def add_client(self, client: Tuple[str, str]) -> bool:
        with self.lock:
            if len(self.clients) < self.client_num:
                self.clients[client] = ClientStatus.INIT
                self.client_params[client] = None
            else:
                return False

            if len(self.clients) == self.client_num:
                for client in self.clients:
                    requests.post(urljoin(client[0], f'send_init_params', ), json={'epochs': self.epochs, 'params': encode_parameters(self.params), 'id': client[1]})
                    self.clients[client] = ClientStatus.RUNNING
                self.status = TaskStatus.WAITING
            return True

    def update_client(self, client: Tuple[str, str], params: Dict[str, Tensor]) -> None:
        with self.lock:
            self.client_params[client] = params
            self.clients[client] = ClientStatus.FINISHED

            if all(self.clients[client] == ClientStatus.FINISHED for client in self.clients):
                self.status = TaskStatus.AGGREGATING

                # TODO
                if self.aggregate_func is prox_aggregate:
                    self.params = self.aggregate_func(self.client_params, self.params)
                elif self.aggregate_func is weighed_avg_aggregate:
                    self.params = self.aggregate_func(self.client_params, {client[1]: 1 / self.client_num for client in self.clients})
                else:
                    self.params = self.aggregate_func(self.client_params)
                self.completed_epoch += 1
                if self.completed_epoch < self.epochs:
                    for client in self.clients:
                        requests.post(urljoin(client[0], f'send_params_to_client'), json={'task': self.name, 'params': encode_parameters(self.params), 'id': client[1]})
                        self.clients[client] = ClientStatus.RUNNING
                    self.status = TaskStatus.WAITING
                else:
                    self.status = TaskStatus.FINISHED

    def get_status(self) -> None:
        return self.task_status_to_text(self.status)
    
    def get_client_statuses(self):
        return {
            f"{client[1]} @ {client[0]}": self.client_status_to_text(status)
            for client, status in self.clients.items()
        }

    def get_snapshot(self):
        with self.lock:
            return {
                "task": self.name,
                "status": self.task_status_to_text(self.status),
                "completed_epoch": self.completed_epoch,
                "clients": {
                    f"{client[1]} @ {client[0]}": self.client_status_to_text(status)
                    for client, status in self.clients.items()
                }
            }
        
    @staticmethod
    def task_status_to_text(status: TaskStatus) -> str:
        return {
            TaskStatus.INIT: "初始化",
            TaskStatus.WAITING: "等待客户端提交",
            TaskStatus.AGGREGATING: "聚合中",
            TaskStatus.FINISHED: "已完成"
        }.get(status, "未知状态")

    @staticmethod
    def client_status_to_text(status: ClientStatus) -> str:
        return {
            ClientStatus.INIT: "等待初始化参数",
            ClientStatus.RUNNING: "训练中",
            ClientStatus.FINISHED: "已提交更新"
        }.get(status, "未知状态")