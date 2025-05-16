import numpy as np
from typing import Optional
from .base import Layer
from ..tensor import Tensor

class Linear(Layer):
    def __init__(self, in_features: int, out_features: int, bias: bool = True):
        super().__init__()
        
        # He initialization
        self.params["weight"] = Tensor(
            np.random.randn(in_features, out_features) * np.sqrt(2.0 / in_features),
            requires_grad=True
        )
        
        if bias:
            self.params["bias"] = Tensor(
                np.zeros(out_features),
                requires_grad=True
            )
        else:
            self.params["bias"] = None
            
    def forward(self, x: Tensor) -> Tensor:
        output = x @ self.params["weight"]
        
        if self.params["bias"] is not None:
            output = output + self.params["bias"]
            
        return output 