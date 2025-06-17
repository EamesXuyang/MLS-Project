import numpy as np
from ..tensor import Tensor
from ..layers.base import Layer

class ReLU(Layer):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor.maximum(Tensor(0, device=x.device), x)

class Sigmoid(Layer):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

class Tanh(Layer):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()