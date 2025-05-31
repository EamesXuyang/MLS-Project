import numpy as np
from ..tensor import Tensor

class Activation:
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
        
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)

class ReLU(Activation):
    def forward(self, x: Tensor) -> Tensor:
        return Tensor.maximum(Tensor(0, device=x.device), x)

class Sigmoid(Activation):
    def forward(self, x: Tensor) -> Tensor:
        return x.sigmoid()

class Tanh(Activation):
    def forward(self, x: Tensor) -> Tensor:
        return x.tanh()