import requests
from urllib.parse import urljoin
from enum import Enum
from typing import Dict, Optional, Tuple
from ...tensor import Tensor

class TaskStatus(Enum):
    INIT = 0
    WAITING = 1
    RUNNING = 2
    FINISHED = 3


class ClientTask:
    def __init__(self, name: str, server: str, client: str, id: str):
        self.name = name
        self.server = server
        self.client = client
        self.task_id = id
        self.params_buffer = None
        self.status = TaskStatus.INIT

    def init_params(self, epochs: int, params: Dict[str, Tensor]) -> None:
        self.epochs = epochs
        self.params_buffer = params
        self.status = TaskStatus.WAITING
    
    def get_init_params(self) -> Optional[Tuple[int, Dict[str, Tensor]]]:
        if self.status == TaskStatus.INIT:
            return None
        else:
            self.stataus = TaskStatus.RUNNING
            return self.epochs, self.params_buffer


    def get_params(self) -> Optional[Dict[str, Tensor]]:
        self.stataus = TaskStatus.RUNNING
        return self.params_buffer
    
    def clean_params(self) -> None:
        self.params_buffer = None

    def set_params(self, params: Dict[str, Tensor]) -> None:
        self.params_buffer = params
