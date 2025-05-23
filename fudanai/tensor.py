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
            
    def unbroadcast_to(self, grad: np.ndarray, shape: Tuple[int, ...]) -> np.ndarray:
        """
        将 grad 从广播后的形状还原回 shape。
        """
        # 多余的维度先 sum 掉
        while grad.ndim > len(shape):
            grad = grad.sum(axis=0)

        # 对应 shape 中为 1 的维度也要 sum
        for i in range(len(shape)):
            if shape[i] == 1 and grad.shape[i] != 1:
                grad = grad.sum(axis=i, keepdims=True)
        return grad
    def __neg__(self) -> 'Tensor':
        result = Tensor(-self.data, requires_grad=self.requires_grad)
        
        if result.requires_grad:
            def _backward(grad):
                self.backward(-grad)
            result._grad_fn = _backward
            result.is_leaf = False
            
        return result
            
    def __add__(self, other: Union['Tensor', float]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    grad_self = self.unbroadcast_to(grad, self.data.shape)
                    self.backward(grad_self)
                if other.requires_grad:
                    grad_other = self.unbroadcast_to(grad, other.data.shape)
                    other.backward(grad_other)
            result._grad_fn = _backward
            result.is_leaf = False
            
        return result
        
    def __radd__(self, other: Union['Tensor', float]) -> 'Tensor':
        return self + other
        
    def __sub__(self, other: Union['Tensor', float]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    self.backward(grad)
                if other.requires_grad:
                    other.backward(-grad)
            result._grad_fn = _backward
            result.is_leaf = False
            
        return result
        
    def __rsub__(self, other: Union['Tensor', float]) -> 'Tensor':
        return -self + other
        
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
        
    def __rmul__(self, other: Union['Tensor', float]) -> 'Tensor':
        return self * other
        
    def __truediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    self.backward(grad / other.data)
                if other.requires_grad:
                    other.backward(-grad * self.data / (other.data * other.data))
            result._grad_fn = _backward
            result.is_leaf = False
            
        return result
        
    def __rtruediv__(self, other: Union['Tensor', float]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other / self
        
    def __pow__(self, other: Union['Tensor', float]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        result = Tensor(self.data ** other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    self.backward(grad * other.data * (self.data ** (other.data - 1)))
                if other.requires_grad:
                    other.backward(grad * (self.data ** other.data) * np.log(self.data))
            result._grad_fn = _backward
            result.is_leaf = False
            
        return result
        
    def __rpow__(self, other: Union['Tensor', float]) -> 'Tensor':
        other = other if isinstance(other, Tensor) else Tensor(other)
        return other ** self
        
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
        
    def log(self) -> 'Tensor':
        result = Tensor(np.log(self.data), requires_grad=self.requires_grad)
        
        if result.requires_grad:
            def _backward(grad):
                self.backward(grad / self.data)
            result._grad_fn = _backward
            result.is_leaf = False
            
        return result 