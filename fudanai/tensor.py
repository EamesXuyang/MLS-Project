import numpy as np
from typing import Union, Tuple, List, Optional

class Tensor:
    def __init__(self, data: Union[np.ndarray, list, float], requires_grad: bool = False):
        if isinstance(data, np.ndarray):
            self.data = data
        else:
            self.data = np.array(data)
        
        self.requires_grad = requires_grad
        self.grad = None
        self._grad_fn = None
        self.is_leaf = True
        
    @property
    def shape(self) -> Tuple:
        return self.data.shape
        
    @property
    def dtype(self):
        return self.data.dtype
        
    def backward(self, grad: Optional[Union['Tensor', np.ndarray]] = None):
        if not self.requires_grad:
            return
            
        if grad is None:
            if self.shape == ():  # scalar
                grad = Tensor(1.0)
            else:
                raise RuntimeError("grad must be specified for non-scalar tensors")
                
        if isinstance(grad, Tensor):
            grad = grad.data
            
        if self.grad is None:
            self.grad = Tensor(grad)
        else:
            self.grad.data += grad
            
        if self._grad_fn is not None:
            self._grad_fn(grad)
            
    def zero_grad(self):
        if self.grad is not None:
            self.grad.data.fill(0)
            
    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    self.backward(grad)
                if other.requires_grad:
                    other.backward(grad)
            result._grad_fn = _backward
            result.is_leaf = False
            
        return result
        
    def __mul__(self, other: Union['Tensor', float]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    self.backward(grad * other.data)
                if other.requires_grad:
                    other.backward(grad * self.data)
            result._grad_fn = _backward
            result.is_leaf = False
            
        return result
        
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        result = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    self.backward(grad @ other.data.T)
                if other.requires_grad:
                    other.backward(self.data.T @ grad)
            result._grad_fn = _backward
            result.is_leaf = False
            
        return result
        
    def sum(self, axis=None, keepdims=False) -> 'Tensor':
        result = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad)
        
        if result.requires_grad:
            def _backward(grad):
                if not keepdims and axis is not None:
                    grad = np.expand_dims(grad, axis)
                grad_expanded = np.broadcast_to(grad, self.shape)
                self.backward(grad_expanded)
            result._grad_fn = _backward
            result.is_leaf = False
            
        return result 