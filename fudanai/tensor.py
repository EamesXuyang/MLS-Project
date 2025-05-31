import numpy as np
from typing import Union, Tuple, List, Optional, Set, Dict

try:
    import cupy as cp
    HAS_CUPY = True
except ImportError:
    HAS_CUPY = False

class Tensor:
    def __init__(self, data: Union[np.ndarray, cp.ndarray, list, float, int, 'Tensor'], requires_grad: bool = False, device=None):
        if device is None:
            if isinstance(data, (np.ndarray, list, float, int)):
                device = 'cpu'
            elif isinstance(data, cp.ndarray):
                device = 'cuda'
            elif isinstance(data, Tensor):
                device = data.device
                data = data.data.copy()
                
        
        if device == 'cuda':
            if not HAS_CUPY:
                raise RuntimeError("CuPy is not available")
            self.xp = cp
            self.device = 'cuda'
            if isinstance(data, cp.ndarray):
                self.data = data
            else:
                self.data = cp.array(data)
        else:
            self.xp = np
            self.device = 'cpu'
            if isinstance(data, np.ndarray):
                self.data = data
            else:
                self.data = np.array(data)
        
        self.requires_grad = requires_grad
        self.grad = None
        self._grad = None # 单次反向传播的梯度
        self._grad_fn = None
        self._prev = None
        self.is_leaf = True

    def __repr__(self):
        return f"Tensor(data={self.data}, requires_grad={self.requires_grad}, device='{self.device}')"
    
    def __getitem__(self, idx):
        if isinstance(idx, tuple):
            idx = tuple([i.data if isinstance(i, Tensor) else i for i in idx])
        elif isinstance(idx, Tensor):
            idx = idx.data
        elif isinstance(idx, slice):
            idx = idx

        sliced_data = self.data[idx]
        result = Tensor(sliced_data, requires_grad=self.requires_grad, device=self.device)

        if self.requires_grad:
            is_tuple = isinstance(idx, tuple)
            all_arrays = is_tuple and all(
                isinstance(i, (self.xp.ndarray, list, slice)) and not isinstance(i, slice) for i in idx)
            is_advcanced = (not is_tuple and isinstance(idx, (self.xp.ndarray, list))) or all_arrays

            def _backward(grad):
                full_grad = self.xp.zeros_like(self.data)

                if is_advcanced:
                    self.xp.add.at(full_grad, idx, grad)
                else:
                    full_grad[idx] = grad

                self._backward_grad(full_grad)

            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False

        return result

    def __setitem__(self, idx, value):
        if isinstance(value, Tensor):
            value = value.data
        else:
            raise ValueError("Only Tensor can be assigned to Tensor")
        
        self.data[idx] = value
    
    def to(self, device: str) -> 'Tensor':
        if device == self.device:
            return self

        if device =='cuda' and not HAS_CUPY:
            raise RuntimeError("CuPy is not available")
        
        if device == 'cuda':
            self.xp = cp
            return Tensor(cp.array(self.data), requires_grad=self.requires_grad, device='cuda')
        else:
            self.xp = np
            return Tensor(np.array(self.data), requires_grad=self.requires_grad, device='cpu')
        
    @property
    def shape(self) -> Tuple:
        return self.data.shape
    
    @property
    def size(self) -> int:
        return self.data.size
        
    @property
    def dtype(self):
        return self.data.dtype
    
    @staticmethod
    def zeros(shape: Tuple[int, ...], requires_grad: bool = False, device: str = 'cpu') -> 'Tensor':
        return Tensor(np.zeros(shape), requires_grad=requires_grad, device=device)
    
    @staticmethod
    def ones(shape: Tuple[int, ...], requires_grad: bool = False, device: str = 'cpu') -> 'Tensor':
        return Tensor(np.ones(shape), requires_grad=requires_grad, device=device)

    @staticmethod
    def ones(shape: Tuple[int, ...], requires_grad: bool = False, device: str = 'cpu') -> 'Tensor':
        return Tensor(np.ones(shape), requires_grad=requires_grad, device=device)

    @staticmethod
    def _ensure_same_device(*args):
        '''
        检查Tensor是否在同一个设备上
        '''
        tensors = [arg for arg in args if isinstance(arg, Tensor)]
        device = tensors[0].device
        for tensor in tensors:
            if tensor.device != device:
                raise ValueError(f'Tensor device mismatch: {device} vs {tensor.device}')
        
    @staticmethod
    def _topological_sort(tensor: 'Tensor', visited: Set['Tensor'], order: List['Tensor']):
        if tensor in visited:
            return
        visited.add(tensor)
        if tensor._prev:
            for parent in tensor._prev:
                Tensor._topological_sort(parent, visited, order)
        order.append(tensor)

    def _update_grad(self, grad: Union[np.ndarray, cp.ndarray, np.generic]):
        if self.grad is None:
            self.grad = Tensor(grad, requires_grad=False, device=self.device)
        else:
            self.grad.data += grad

    def _backward_grad(self, grad: Union[np.ndarray, cp.ndarray, np.generic]):
        if self._grad is None:
            self._grad = grad
        else:
            self._grad += grad
            
    def backward(self, grad: Optional[Union[np.ndarray, cp.ndarray, np.generic]] = None):
        '''
        通过前一节点返回的梯度计算当前节点的梯度，
        并通过_grad_fn通知前驱节点的梯度信息便于其反向传播
        '''
        if not self.requires_grad:
            return
            
        if self._grad is None:
            if self.shape == ():  # scalar
                self._grad = self.xp.array(1.0)
            else:
                raise RuntimeError("grad must be specified for non-scalar tensors")
        
        visited = set()
        topo_order = []
        Tensor._topological_sort(self, visited, topo_order)
        for node in reversed(topo_order):
            if node._grad_fn is not None:
                node._grad_fn(node._grad)
            if node.requires_grad:
                node._update_grad(node._grad)
            
        for node in reversed(topo_order):
            node._grad = None

    def zero_grad(self):
        if self.grad is not None:
            self.grad.data.fill(0)
    
    @staticmethod
    def maximum(a: 'Tensor', b: 'Tensor') -> 'Tensor':
        result = Tensor(np.maximum(a.data, b.data), requires_grad=True, device=a.device)

        if a.requires_grad or b.requires_grad:
            def _backward(grad):
                a._backward_grad(grad * (a.data == result.data))
                b._backward_grad(grad * (b.data == result.data))
            result._grad_fn = _backward
            result._prev = [a, b]
            result.is_leaf = False
        return result
    
    @staticmethod
    def minimum(a: 'Tensor', b: 'Tensor') -> 'Tensor':
        result = Tensor(np.minimum(a.data, b.data), requires_grad=True, device=a.device)

        if a.requires_grad or b.requires_grad:
            def _backward(grad):
                a._backward_grad(grad * (a.data == result.data))
                b._backward_grad(grad * (b.data == result.data))
            result._grad_fn = _backward
            result._prev = [a, b]
            result.is_leaf = False
        return result
    
    @staticmethod
    def concat(tensors: List['Tensor'], axis=0) -> 'Tensor':
        Tensor._ensure_same_device(*tensors)
        datas = [t.data for t in tensors]
        result = Tensor(np.concatenate(datas, axis=axis), requires_grad=True, device=tensors[0].device)
        xp = tensors[0].xp

        if any(t.requires_grad for t in tensors):
            def _backward(grad):
                splits = xp.cumsum([t.shape[axis] for t in tensors])[:-1]
                grads = xp.split(grad, splits, axis=axis)
                for t, g in zip(tensors, grads):
                    t._backward_grad(g)
            result._grad_fn = _backward
            result._prev = tensors
            result.is_leaf = False

        return result

    @staticmethod
    def stack(tensors: List['Tensor'], axis=0) -> 'Tensor':
        Tensor._ensure_same_device(*tensors)
        datas = [t.data for t in tensors]
        result = Tensor(np.stack(datas, axis=axis), requires_grad=True, device=tensors[0].device)
        xp = tensors[0].xp

        if any(t.requires_grad for t in tensors):
            def _backward(grad):
                grads = xp.split(grad, grad.shape[axis], axis=axis)
                grads = [xp.squeeze(g, axis=axis) for g in grads]
                for t, g in zip(tensors, grads):
                    t._backward_grad(g)
            result._grad_fn = _backward
            result._prev = tensors
            result.is_leaf = False

        return result
            
    def _unbroadcast_to(self, grad: Union[np.ndarray, cp.ndarray], shape: Tuple[int, ...]) -> Union[np.ndarray, cp.ndarray]:
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
    
    @property
    def T(self) -> 'Tensor':
        transposed_data = self.data.T
        result = Tensor(transposed_data, requires_grad=self.requires_grad, device=self.device)
        
        if self.requires_grad:
            def _backward(grad):
                self._backward_grad(grad.T)
            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False
        
        return result
    
    def __eq__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        Tensor._ensure_same_device(self, other)
        result = Tensor(self.data == (other.data if isinstance(other, Tensor) else other), device=self.device)
        return result
    
    def __hash__(self):
        return id(self)
        
    def __neg__(self) -> 'Tensor':
        '''
        实现 -self
        '''
        result = Tensor(-self.data, requires_grad=self.requires_grad, device=self.device)
        
        if result.requires_grad:
            def _backward(grad):
                self._backward_grad(-grad)
            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False
            
        return result
            
    def __add__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        '''
        实现 self + other
        '''
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        Tensor._ensure_same_device(self, other)

        result = Tensor(self.data + other.data, requires_grad=self.requires_grad or other.requires_grad, device=self.device)
        
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    grad_self = self._unbroadcast_to(grad, self.data.shape)
                    self._backward_grad(grad_self)
                if other.requires_grad:
                    grad_other = self._unbroadcast_to(grad, other.data.shape)
                    other._backward_grad(grad_other)
            result._grad_fn = _backward
            result._prev = [self, other]
            result.is_leaf = False
            
        return result
        
    def __radd__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        '''
        实现 other + self
        '''
        return self + other
        
    def __sub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        '''
        实现 self - other
        '''
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        Tensor._ensure_same_device(self, other)

        result = Tensor(self.data - other.data, requires_grad=self.requires_grad or other.requires_grad, device=self.device)
        
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    grad_self = self._unbroadcast_to(grad, self.data.shape)
                    self._backward_grad(grad_self)
                if other.requires_grad:
                    grad_other = self._unbroadcast_to(-grad, other.data.shape)
                    other._backward_grad(grad_other)
            result._grad_fn = _backward
            result._prev = [self, other]
            result.is_leaf = False
            
        return result
        
    def __rsub__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        '''
        实现 other - self
        '''
        return -self + other
        
    def __mul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        '''
        实现 self * other
        '''
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        Tensor._ensure_same_device(self, other)

        result = Tensor(self.data * other.data, requires_grad=self.requires_grad or other.requires_grad, device=self.device)
        
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    grad_self = self._unbroadcast_to(grad * other.data, self.data.shape)
                    self._backward_grad(grad_self)
                if other.requires_grad:
                    grad_other = self._unbroadcast_to(grad * self.data, other.data.shape)
                    other._backward_grad(grad_other)
            result._grad_fn = _backward
            result._prev = [self, other]
            result.is_leaf = False
            
        return result
        
    def __rmul__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        '''
        实现 other * self
        '''
        return self * other
        
    def __truediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        '''
        实现 self / other
        '''
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        Tensor._ensure_same_device(self, other)

        result = Tensor(self.data / other.data, requires_grad=self.requires_grad or other.requires_grad, device=self.device)
        
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    grad_self = self._unbroadcast_to(grad / other.data, self.data.shape)
                    self._backward_grad(grad_self)
                if other.requires_grad:
                    grad_other = self._unbroadcast_to(-grad * self.data / ( other.data * other.data), other.data.shape)
                    other._backward_grad(grad_other)
            result._grad_fn = _backward
            result._prev = [self, other]
            result.is_leaf = False
            
        return result
        
    def __rtruediv__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        '''
        实现 other / self
        '''
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        return other / self
        
    def __pow__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        '''
        实现 self ** other
        '''
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        Tensor._ensure_same_device(self, other)

        result = Tensor(self.data ** other.data, requires_grad=self.requires_grad or other.requires_grad, device=self.device)
        
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    grad_self = self._unbroadcast_to(grad * other.data * (self.data ** (other.data - 1)), self.data.shape)
                    self._backward_grad(grad_self)
                if other.requires_grad:
                    grad_other = self._unbroadcast_to(grad * (self.data ** other.data) * np.log(self.data), other.data.shape)
                    other._backward_grad(grad_other)
            result._grad_fn = _backward
            result._prev = [self, other]
            result.is_leaf = False
            
        return result
        
    def __rpow__(self, other: Union['Tensor', float, int]) -> 'Tensor':
        '''
        实现 other ** self
        '''
        other = other if isinstance(other, Tensor) else Tensor(other, device=self.device)
        return other ** self
        
    def __matmul__(self, other: 'Tensor') -> 'Tensor':
        '''
        实现 self @ other
        '''
        Tensor._ensure_same_device(self, other)
        result = Tensor(self.data @ other.data, requires_grad=self.requires_grad or other.requires_grad)
        
        if result.requires_grad:
            def _backward(grad):
                if self.requires_grad:
                    self._backward_grad(grad @ other.data.swapaxes(-1, -2))
                if other.requires_grad:
                    other._backward_grad(self.data.swapaxes(-1, -2) @ grad)
            result._grad_fn = _backward
            result._prev = [self, other]
            result.is_leaf = False

        return result

    def sqrt(self) -> 'Tensor':
        result = Tensor(self.xp.sqrt(self.data), requires_grad=self.requires_grad, device=self.device)

        if result.requires_grad:
            def _backward(grad):
                self._backward_grad(grad / (2 * result.data))

            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False

        return result

    def log(self) -> 'Tensor':
        result = Tensor(self.xp.log(self.data), requires_grad=self.requires_grad, device=self.device)
        
        if result.requires_grad:
            def _backward(grad):
                self._backward_grad(grad / self.data)
            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False
            
        return result
    
    def exp(self) -> 'Tensor':
        result = Tensor(self.xp.exp(self.data), requires_grad=self.requires_grad, device=self.device)

        if result.requires_grad:
            def _backward(grad):
                self._backward_grad(grad * result.data)
            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False

        return result

    def mean(self, axis=None, keepdims=False) -> 'Tensor':
        result = Tensor(self.data.mean(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, device=self.device)

        if result.requires_grad:
            def _backward(grad):
                if not keepdims and axis is not None:
                    grad = self.xp.expand_dims(grad, axis)
                shape = self.data.shape
                if axis is None:
                    denom = self.data.size
                elif isinstance(axis, int):
                    denom = shape[axis]
                else:
                    denom = 1
                    for ax in axis:
                        denom *= self.data.shape[ax]
                grad_expanded = grad * self.xp.ones_like(self.data) / denom
                self._backward_grad(grad_expanded)
            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False

        return result
    
    def var(self, axis=None, keepdims=False) -> 'Tensor':
        mean = self.data.mean(axis=axis, keepdims=keepdims)
        diff = self.data - mean
        var_data = (diff ** 2).sum(axis=axis, keepdims=keepdims) / (self.data.shape[axis] - 1 if axis is not None else self.data.size - 1)
        result = Tensor(var_data, requires_grad=self.requires_grad, device=self.device)

        if result.requires_grad:
            def _backward(grad):
                n = self.data.shape[axis] if axis is not None else self.data.size
                grad_expanded = grad
                if not keepdims and axis is not None:
                    if isinstance(axis, int):
                        grad_expanded = self.xp.expand_dims(grad, axis)
                    else:
                        for ax in sorted(axis):
                            grad_expanded = self.xp.expand_dims(grad_expanded, ax)
                grad_input = grad_expanded * 2 / (n - 1) * diff
                self._backward_grad(grad_input)
            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False

        return result
    
    def max(self, axis=None, keepdims=False) -> 'Tensor':
        result = Tensor(self.data.max(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, device=self.device)

        if self.requires_grad:
            def _backward(grad):
                if not keepdims and axis is not None:
                    grad = self.xp.expand_dims(grad, axis)
                grad_expanded = self.xp.where(self.data == result.data, grad, 0)
                self._backward_grad(grad_expanded)
            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False
        
        return result

    def argmax(self, axis=None, keepdims=False) -> 'Tensor':
        data = self.data.argmax(axis=axis, keepdims=keepdims)
        
        if keepdims and axis is not None:
            data = self.xp.expand_dims(data, axis)
        
        return Tensor(data, requires_grad=False, device=self.device)

        
    def sum(self, axis=None, keepdims=False) -> 'Tensor':
        result = Tensor(self.data.sum(axis=axis, keepdims=keepdims), requires_grad=self.requires_grad, device=self.device)
        
        if result.requires_grad:
            def _backward(grad):
                if not keepdims and axis is not None:
                    grad = self.xp.expand_dims(grad, axis)
                grad_expanded = self.xp.broadcast_to(grad, self.shape).copy()
                self._backward_grad(grad_expanded)
            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False
            
        return result 
    
    def reshape(self, shape: Tuple[int, ...]) -> 'Tensor':
        result = Tensor(self.data.reshape(shape), requires_grad=self.requires_grad, device=self.device)

        if result.requires_grad:
            def _backward(grad):
                self._backward_grad(grad.reshape(self.shape))
            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False
        
        return result
    
    def flatten(self) -> 'Tensor':
        result = Tensor(self.data.flatten(), requires_grad=self.requires_grad, device=self.device)

        if result.requires_grad:
            def _backward(grad):
                self._backward_grad(grad.reshape(self.shape))
            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False

        return result
    
    def squeeze(self, axis: Optional[Union[int, Tuple[int, ...]]] = None) -> 'Tensor':
        result = Tensor(self.data.squeeze(axis), requires_grad=self.requires_grad, device=self.device)

        if result.requires_grad:
            def _backward(grad):
                self._backward_grad(grad.reshape(self.shape))
            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False

        return result
    
    def unsqueeze(self, axis: int) -> 'Tensor':
        result = Tensor(self.xp.expand_dims(self.data, axis), requires_grad=self.requires_grad, device=self.device)

        if result.requires_grad:
            def _backward(grad):
                self._backward_grad(self.xp.squeeze(grad, axis))
            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False

        return result
    
    def transpose(self, axes: Tuple[int, int]) -> 'Tensor':
        result = Tensor(self.data.swapaxes(*axes), requires_grad=self.requires_grad, device=self.device)
        
        if result.requires_grad:
            def _backward(grad):
                self._backward_grad(grad.swapaxes(*axes))
            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False
        
        return result
    
    def permute(self, axes: Tuple[int, ...]) -> 'Tensor':
        result = Tensor(self.data.transpose(axes), requires_grad=self.requires_grad, device=self.device)

        if result.requires_grad:
            def _backward(grad):
                reverse_axes = [axes.index(i) for i in range(len(axes))]
                self._backward_grad(grad.transpose(reverse_axes))
            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False

        return result

    def chunk(self, chunks: int, axis: int=0) -> list['Tensor']:
        data_chunks = self.xp.array_split(self.data, chunks, axis=axis)
        results = [Tensor(chunk, requires_grad=self.requires_grad, device=self.device) for chunk in data_chunks]

        if self.requires_grad:
            grads_collected = [None] * chunks
            call_count = [0]
            def _backward(i):
                def _inner(grad):
                    grads_collected[i] = grad
                    call_count[0] += 1
                    if call_count[0] == chunks:
                        grads = [g if g is not None else self.xp.zeros_like(data_chunks[i]) for i, g in enumerate(grads_collected)]
                        self._backward_grad(self.xp.concatenate(grads, axis=axis))
                return _inner
            for i, result in enumerate(results):
                result._grad_fn = _backward(i)
                result._prev = [self]
                result.is_leaf = False

        return results
    
    def masked_fill(self, mask: 'Tensor', value: Union[int, float]) -> 'Tensor':
        result = Tensor(self.xp.where(mask.data, value, self.data), requires_grad=self.requires_grad, device=self.device)

        if result.requires_grad:
            def _backward(grad):
                self._backward_grad(grad * mask.data)
            result._grad_fn = _backward
            result._prev = [self]

        return result

    def sigmoid(self) -> 'Tensor':
        result = Tensor(1 / (1 + self.xp.exp(-self.data)), requires_grad=self.requires_grad, device=self.device)

        if result.requires_grad:
            def _backward(grad):
                self._backward_grad(grad * result.data * (1 - result.data))
            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False
        
        return result

    def sin(self) -> 'Tensor':
        result = Tensor(self.xp.sin(self.data), requires_grad=self.requires_grad, device=self.device)

        if result.requires_grad:
            def _backward(grad):
                self._backward_grad(grad * self.xp.cos(self.data))
            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False

        return result

    def cos(self) -> 'Tensor':
        result = Tensor(self.xp.cos(self.data), requires_grad=self.requires_grad, device=self.device)

        if result.requires_grad:
            def _backward(grad):
                self._backward_grad(grad * -self.xp.sin(self.data))
            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False

        return result
    
    def tan(self) -> 'Tensor':
        result = Tensor(self.xp.tan(self.data), requires_grad=self.requires_grad, device=self.device)

        if result.requires_grad:
            def _backward(grad):
                self._backward_grad(grad * (1 + self.data ** 2))
            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False

        return result
    
    def tanh(self) -> 'Tensor':
        result = Tensor(self.xp.tanh(self.data), requires_grad=self.requires_grad, device=self.device)

        if result.requires_grad:
            def _backward(grad):
                self._backward_grad(grad * (1 - result.data ** 2))
            result._grad_fn = _backward
            result._prev = [self]
            result.is_leaf = False
        
        return result