import numpy as np
from ..tensor import Tensor

class Activation:
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
        
    def backward(self, grad: np.ndarray) -> np.ndarray:
        raise NotImplementedError
        
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

class ReLU(Activation):
    def forward(self, x: Tensor) -> Tensor:
        result = Tensor(np.maximum(0, x.data), requires_grad=x.requires_grad)
        if result.requires_grad:
            def _backward(grad):
                x.backward(grad * (x.data > 0))
            result._grad_fn = _backward
            result.is_leaf = False
        return result

class Sigmoid(Activation):
    def forward(self, x: Tensor) -> Tensor:
        result = Tensor(1 / (1 + np.exp(-x.data)), requires_grad=x.requires_grad)
        if result.requires_grad:
            def _backward(grad):
                x.backward(grad * result.data * (1 - result.data))
            result._grad_fn = _backward
            result.is_leaf = False
        return result

class Tanh(Activation):
    def forward(self, x: Tensor) -> Tensor:
        result = Tensor(np.tanh(x.data), requires_grad=x.requires_grad)
        if result.requires_grad:
            def _backward(grad):
                x.backward(grad * (1 - result.data ** 2))
            result._grad_fn = _backward
            result.is_leaf = False
        return result 