import requests
from urllib.parse import urljoin
from ..tensor import Tensor
import numpy as np
from .util import decode_parameters, encode_parameters

class Task:
    def __init__(self, name, server, client_port, model, train_local_func, trainloader, valloader=None):
        self.name = name
        self.server = server
        self.client = f'http://127.0.0.1:{client_port}/'
        self.model = model
        self.train = train_local_func
        self.trainloader = trainloader
        self.valloader = valloader

    def set_parameters(self, params):
        self.model.load_parameters(params)

    def get_parameters(self):
        return self.model.parameters()

    def train(self, epochs, model, trainloader):
        pass
    
    def evalute(self):
        pass


    def run(self):
        response = requests.post(urljoin(self.client, 'register_task'), json={'name': self.name, "server": self.server, 'client': self.client})
        if response.status_code != 200:
            raise Exception('Register task failed')
        
        response = requests.get(urljoin(self.client, f'{self.name}/get_init_params'))
        if response.status_code != 200:
            raise Exception('Get init params failed')
        data = response.json()
        self.epochs = data['epochs']
        self.set_parameters(decode_parameters(data['params']))

        for epoch in range(self.epochs):
            self.train(1, self.model, self.trainloader)
            response = requests.post(urljoin(self.client, f'{self.name}/send_params_to_server'), json={'params': encode_parameters(self.get_parameters())})
            if response.status_code != 200:
                raise Exception('Update params failed')
            if epoch != self.epochs - 1:
                response = requests.get(urljoin(self.client, f'{self.name}/get_params_from_server'))
                if response.status_code != 200:
                    raise Exception('Get params from server failed')
                data = response.json()
                self.set_parameters(decode_parameters(data['params']))
