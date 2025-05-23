import requests
from urllib.parse import urljoin

class Task:
    def __init__(self, name, server, client_port, model, trainloader, valloader):
        self.name = name
        self.server = server
        self.client = f'http://127.0.0.1:{client_port}'
        self.model = model
        self.trainloader = trainloader
        self.valloader = valloader

    def set_parameters(self):
        pass

    def get_parameters(self):
        pass

    def train(self, epochs):
        pass
    
    def evalute(self):
        pass

    def run(self):
        response = requests.post(urljoin(self.client, 'register_task'), json={'name': self.name, "server": self.server})
        if response.status_code != 200:
            raise Exception('Register task failed')
        
        response = requests.get(urljoin(self.client, self.name ,'get_init_params'))
        if response.status_code != 200:
            raise Exception('Get init params failed')
        data = response.json()
        self.set_parameters(data['params'])

        for epoch in range(self.epochs):
            self.train(1)
            response = requests.post(urljoin(self.client, self.name, 'send_params_to_server'), json={'params': self.get_parameters()})
            if response.status_code != 200:
                raise Exception('Update params failed')
            response = requests.get(urljoin(self.client, self.name, 'get_params_from_server'))
            if response.status_code != 200:
                raise Exception('Get params from server failed')
            data = response.json()
            self.set_parameters(data['params'])
