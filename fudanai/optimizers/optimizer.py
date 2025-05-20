from typing import Dict, List
from ..tensor import Tensor
from ..layers.base import Layer
import numpy as np

class Optimizer:
    def __init__(self, params: List[Tensor], lr: float = 0.01):
        self.params = params
        self.lr = lr
        
    def step(self):
        raise NotImplementedError
        
    def zero_grad(self):
        for param in self.params:
            param.zero_grad()

class SGD(Optimizer):
    def __init__(self, params: List[Tensor], lr: float = 0.01, momentum: float = 0.0):
        super().__init__(params, lr)
        self.momentum = momentum
        self.velocities = [0.0 for _ in params]
        
    def step(self):
        for i, param in enumerate(self.params):
            if param.grad is not None:
                self.velocities[i] = self.momentum * self.velocities[i] - self.lr * param.grad.data
                param.data += self.velocities[i]

class Adam(Optimizer):
    def __init__(
        self,
        params: List[Tensor],
        lr: float = 0.001,
        betas: tuple = (0.9, 0.999),
        eps: float = 1e-8
    ):
        super().__init__(params, lr)
        self.betas = betas
        self.eps = eps
        self.m = [np.zeros_like(param.data) for param in params]  # First moment
        self.v = [np.zeros_like(param.data) for param in params]  # Second moment
        self.t = 0  # Time step
        
    def step(self):
        self.t += 1
        for i, param in enumerate(self.params):
            if param.grad is not None:
                g = param.grad.data
                self.m[i] = self.betas[0] * self.m[i] + (1 - self.betas[0]) * g
                self.v[i] = self.betas[1] * self.v[i] + (1 - self.betas[1]) * g * g
                
                # Bias correction
                m_hat = self.m[i] / (1 - self.betas[0] ** self.t)
                v_hat = self.v[i] / (1 - self.betas[1] ** self.t)
                
                param.data -= self.lr * m_hat / (np.sqrt(v_hat) + self.eps) 