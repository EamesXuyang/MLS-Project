import requests
from urllib.parse import urljoin
from enum import Enum

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
    def __init__(self, name, client_num, model, epochs, aggregate_func):
        self.name = name
        self.clent_num = client_num
        self.model = model
        self.epochs = epochs
        self.completed_epoch = 0
        self.aggregate_func = aggregate_func
        self.clients = {}
        self.params = {}
        self.status = TaskStatus.INIT

    def add_client(self, client):
        if len(self.clients) < self.client_num:
            self.clients[client] = ClientStatus.INIT
            self.params[client] = None
        else:
            return False

        if len(self.clients) == self.client_num:
            for client in self.clients:
                # TODO
                requests.post(urljoin(client, 'init_model'), files={'params': self.model}, json={'task': self.name, 'epoch': self.epochs})
                self.clients[client] = ClientStatus.RUNNING
            self.status = TaskStatus.WAITING
        return True

    def update_client(self, client, params):
        self.params[client] = params
        self.clients[client] = ClientStatus.FINISHED

        if all(self.clients[client] == ClientStatus.FINISHED for client in self.clients):
            self.status = TaskStatus.AGGREGATING

            self.model = self.aggregate_func(self.params)
            self.completed_epoch += 1
            if self.completed_epoch < self.epochs:
                for client in self.clients:
                    # TODO
                    requests.post(urljoin(client, 'send_model'), files={'params': self.model}, json={'task': self.name})
                    self.clients[client] = ClientStatus.RUNNING
                self.status = TaskStatus.WAITING
            else:
                self.status = TaskStatus.FINISHED

    def get_status(self):
        return self.status

