from typing import Dict, Any
from ..tensor import Tensor

class Layer:
    def __init__(self):
        # 用于注册参数中的Tensor，若类型为Layer则无需在此注册，可由parameters方法递归注册
        self.params: Dict[str, Tensor] = {}
        self.grads: Dict[str, Tensor] = {}
        self.device = 'cpu'
        self.training = True
        
    def forward(self, *args) -> Tensor:
        raise NotImplementedError
        
    def __call__(self, *args) -> Tensor:
        return self.forward(*args)
        
    def train(self):
        self.training = True
        
    def eval(self):
        self.training = False
        
    def zero_grad(self):
        for param in self.parameters().values():
            param.zero_grad()

    def to(self, device: str):
        if device == self.device:
            return
        else:
            self.device = device
            if 'params' in self.__dict__:
                for name, val in self.params.items():
                    self.params[name] = val.to(device)
            for attr_name, attr_val in self.__dict__.items():
                if isinstance(attr_val, Layer):
                    attr_val.to(device)
                elif isinstance(attr_val, (list, tuple)):
                    for idx, item in enumerate(attr_val):
                        if isinstance(item, Layer):
                            attr_val[idx].to(device)
                elif isinstance(attr_val, dict):
                    for k, v in attr_val.items():
                        if isinstance(v, Layer):
                            attr_val[k].to(device)
            
    def parameters(self, prefix: str="") -> Dict[str, Tensor]:
        flat = {}
        if 'params' in self.__dict__:
            for name, val in self.params.items():
                key = f'{prefix}.{name}' if prefix else name
                flat[key] = val
        
        def recurse(obj, parent_prefix):
            if isinstance(obj, Layer):
                flat.update(obj.parameters(parent_prefix))
            elif isinstance(obj, (list, tuple)):
                for idx, item in enumerate(obj):
                    recurse(item, f'{parent_prefix}.{idx}' if parent_prefix else f'{idx}')
            elif isinstance(obj, dict):
                for k, v in obj.items():
                    recurse(v, f'{parent_prefix}.{k}' if parent_prefix else f'{k}')

        for attr_name, attr_val in self.__dict__.items():
            if attr_name == 'params':
                continue
            recurse(attr_val, f'{prefix}.{attr_name}' if prefix else attr_name)
        return flat
    

    def load_parameters(self, params: Dict[str, Tensor]):
        for full_key, value in params.items():
            keys = full_key.split('.')
            current = self
            for key in keys[:-1]:
                if isinstance(current, (list, tuple)):
                    key = int(key)
                    current = current[key]
                elif isinstance(current, dict):
                    current = current[key]
                elif hasattr(current, key):
                    current = getattr(current, key)
                else:
                    raise KeyError(f'Key {key} not found in {full_key}')
            last_key = keys[-1]
            if isinstance(current, Layer) and last_key in current.params:
                current.params[last_key].data = value.data
                current.params[last_key].device = value.device
            else:
                raise KeyError(f'Key {last_key} not found in {full_key}')