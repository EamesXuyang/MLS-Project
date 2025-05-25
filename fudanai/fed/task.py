import requests
from urllib.parse import urljoin
from ..layers.base import Layer
from ..tensor import Tensor
import numpy as np
from .util import decode_parameters, encode_parameters
import logging
import uuid
from typing import Callable, Optional, Dict

logger = logging.getLogger(__name__)
logger.setLevel(logging.DEBUG)
logger.addHandler(logging.NullHandler())


class Task:
    def __init__(self, name: str, server: str, client_port: int, model: Layer, train_local_func: Callable[[int, Layer, object], None], trainloader: object, valloader: Optional[object]=None, log: bool=True) -> None:
        self.name = name
        self.task_id = str(uuid.uuid4())
        self.server = server
        self.client = f'http://127.0.0.1:{client_port}/'
        self.model = model
        self.train = train_local_func
        self.trainloader = trainloader
        self.valloader = valloader
        self.log_enabled = log

        if self.log_enabled:
            handler = logging.StreamHandler()
            formatter = logging.Formatter('[%(asctime)s] [%(levelname)s] %(message)s')
            handler.setFormatter(formatter)
            logger.addHandler(handler)
            logger.setLevel(logging.INFO)
            logger.info(f"Initialized Task: {self.name}({self.task_id}) with client at {self.client} connecting to server {self.server}")
        else:
            logger.disabled = True

    def set_parameters(self, params: Dict[str, Tensor]) -> None:
        self.model.load_parameters(params)

    def get_parameters(self) -> Dict[str, Tensor]:
        return self.model.parameters()

    def train(self, epochs: int, model: Layer, trainloader: object) -> None:
        pass
    
    def evalute(self):
        pass

    def run(self) -> None:
        logger.info("Registering task to client...")
        response = requests.post(urljoin(self.client, 'register_task'), json={'name': self.name, "server": self.server, 'client': self.client, 'id': self.task_id})
        if response.status_code != 200:
            logger.error("Register task failed")
            raise Exception('Register task failed')
        
        logger.info("Fetching initial parameters from client...")
        response = requests.get(urljoin(self.client, 'get_init_params'), params={'id': self.task_id})
        if response.status_code != 200:
            raise Exception('Get init params failed')
        
        data = response.json()
        self.epochs = data.get('epochs')
        logger.info(f"Training for {self.epochs} epochs.")
        self.set_parameters(decode_parameters(data.get('params')))

        for epoch in range(self.epochs):
            logger.info(f"Epoch {epoch + 1}/{self.epochs} - Training locally")
            self.train(1, self.model, self.trainloader)

            logger.info("Sending parameters to server...")
            response = requests.post(urljoin(self.client, 'send_params_to_server'), json={'id': self.task_id, 'params': encode_parameters(self.get_parameters())})
            if response.status_code != 200:
                logger.error("Failed to send parameters to server")
                raise Exception('Update params failed')
            
            if epoch != self.epochs - 1:
                logger.info("Fetching parameters from server...")
                response = requests.get(urljoin(self.client, f'get_params_from_server'), params={'id': self.task_id})
                if response.status_code != 200:
                    logger.error("Failed to fetch parameters from server")
                    raise Exception('Get params from server failed')
                
                data = response.json()
                self.set_parameters(decode_parameters(data.get('params')))
                logger.info("Parameters updated from server.")
