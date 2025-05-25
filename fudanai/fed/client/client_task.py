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
        self.completed_epoch = 0
        self.status = TaskStatus.INIT

    def init_params(self, epochs: int, params: Dict[str, Tensor]) -> None:
        self.epochs = epochs
        self.params_buffer = params
    
    def get_init_params(self) -> Optional[Tuple[int, Dict[str, Tensor]]]:
        if self.params_buffer is None:
            return None
        else:
            return self.epochs, self.params_buffer

    def get_params(self) -> Optional[Dict[str, Tensor]]:
        return self.params_buffer
    
    def clean_params(self) -> None:
        self.params_buffer = None

    def set_params(self, params: Dict[str, Tensor]) -> None:
        self.params_buffer = params

    def set_status(self, status: TaskStatus) -> None:
        self.status = status

    def get_status(self) -> str:
        return self.task_status_to_text(self.status)
    
    def increase_completed_epoch(self) -> None:
        self.completed_epoch += 1

    def check_finished(self) -> bool:
        return self.completed_epoch == self.epochs

    @staticmethod
    def task_status_to_text(status: TaskStatus) -> str:
        return {
            TaskStatus.INIT: "等待初始化参数",
            TaskStatus.WAITING: "等待服务端聚合参数",
            TaskStatus.RUNNING: "训练中",
            TaskStatus.FINISHED: "训练完成"
        }.get(status, "未知状态")