import requests
from urllib.parse import urljoin
from enum import Enum
from ..util import encode_parameters, decode_parameters

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
    def __init__(self, name, client_num, epochs, aggregate_func, params):
        self.name = name
        self.client_num = client_num
        self.params = params
        self.epochs = epochs
        self.completed_epoch = 0
        self.aggregate_func = aggregate_func
        self.clients = {}
        self.client_params = {}
        self.status = TaskStatus.INIT

    def add_client(self, client):
        if len(self.clients) < self.client_num:
            self.clients[client] = ClientStatus.INIT
            self.client_params[client] = None
        else:
            return False

        if len(self.clients) == self.client_num:
            for client in self.clients:
                requests.post(urljoin(client, f'{self.name}/send_init_params', ), json={'epochs': self.epochs, 'params': encode_parameters(self.params)})
                self.clients[client] = ClientStatus.RUNNING
            self.status = TaskStatus.WAITING
        return True

    def update_client(self, client, params):
        self.client_params[client] = params
        self.clients[client] = ClientStatus.FINISHED

        if all(self.clients[client] == ClientStatus.FINISHED for client in self.clients):
            self.status = TaskStatus.AGGREGATING

            self.params = self.aggregate_func(self.client_params)
            self.completed_epoch += 1
            if self.completed_epoch < self.epochs:
                for client in self.clients:
                    requests.post(urljoin(client, f'{self.name}/send_params_to_client'), json={'task': self.name, 'params': encode_parameters(self.params)})
                    self.clients[client] = ClientStatus.RUNNING
                self.status = TaskStatus.WAITING
            else:
                self.status = TaskStatus.FINISHED

    def get_status(self):
        return self.status

