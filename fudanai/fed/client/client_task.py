import requests
from urllib.parse import urljoin
from enum import Enum

class TaskStatus(Enum):
    INIT = 0
    WAITING = 1
    RUNNING = 2
    FINISHED = 3


class ClientTask:
    def __init__(self, name, server):
        self.name = name
        self.server = server
        self.params_buffer = None
        self.stataus = TaskStatus.INIT

    def init_params(self, epochs, params):
        self.epochs = epochs
        self.params_buffer = params
        self.status = TaskStatus.WAITING
    
    def get_init_params(self):
        if self.status == TaskStatus.INIT:
            return None
        else:
            return self.epochs, self.params_buffer


    def get_params(self):
        return self.params_buffer
    
    def clean_params(self):
        self.params_buffer = None

    def set_params(self, params):
        self.params_buffer = params
