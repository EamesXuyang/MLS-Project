from typing import Dict, Any
from ..tensor import Tensor

class Layer:
    def __init__(self):
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
        self.training = True
        
    def forward(self, x: Tensor) -> Tensor:
        raise NotImplementedError
        
    def __call__(self, x: Tensor) -> Tensor:
        return self.forward(x)
        
    def train(self):
        self.training = True
        
    def eval(self):
        self.training = False
        
    def zero_grad(self):
        for param in self.params.values():
            param.zero_grad()
            
    def parameters(self) -> Dict[str, Tensor]:
        return self.params 