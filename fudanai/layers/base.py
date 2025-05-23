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
            
    def parameters(self, prefix: str="") -> Dict[str, Tensor]:
        flat = {}
        if 'params' in self.__dict__:
            for name, val in self.params.items():
                key = f'{prefix}.{name}' if prefix else name
                flat[key] = val
        
        for attr_name, attr_val in self.__dict__.items():
            if isinstance(attr_val, Layer) and attr_name != 'params':
                sub_prefix = f'{prefix}.{attr_name}' if prefix else attr_name
                sub_params = attr_val.parameters(sub_prefix)
                flat.update(sub_params)
        return flat
    

    def load_parameters(self, params: Dict[str, Tensor]):
        for full_key, value in params.items():
            keys = full_key.split('.')
            current = self
            for key in keys[:-1]:
                if hasattr(current, key) and isinstance(getattr(current, key), Layer):
                    current = getattr(current, key)
                else:
                    raise KeyError(f'Cannot find Layer for key: {key} in path {full_key}')
            last_key = keys[-1]
            current.params[last_key].data = value.data