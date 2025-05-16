import numpy as np
from typing import Tuple, Optional
from .base import Layer
from ..tensor import Tensor

class Conv2d(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        padding: int = 0,
        bias: bool = True
    ):
        super().__init__()
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # He initialization
        self.params["weight"] = Tensor(
            np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * \
            np.sqrt(2.0 / (in_channels * kernel_size * kernel_size)),
            requires_grad=True
        )
        
        if bias:
            self.params["bias"] = Tensor(np.zeros(out_channels), requires_grad=True)
        else:
            self.params["bias"] = None
            
    def _im2col(self, x: np.ndarray) -> np.ndarray:
        N, C, H, W = x.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        img = np.pad(x, [(0, 0), (0, 0), (self.padding, self.padding), (self.padding, self.padding)], 'constant')
        col = np.zeros((N, C, self.kernel_size, self.kernel_size, out_h, out_w))
        
        for y in range(self.kernel_size):
            y_max = y + self.stride * out_h
            for x in range(self.kernel_size):
                x_max = x + self.stride * out_w
                col[:, :, y, x, :, :] = img[:, :, y:y_max:self.stride, x:x_max:self.stride]
                
        col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
        return col
        
    def _col2im(self, col: np.ndarray, x_shape: Tuple) -> np.ndarray:
        N, C, H, W = x_shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        col = col.reshape(N, out_h, out_w, C, self.kernel_size, self.kernel_size).transpose(0, 3, 4, 5, 1, 2)
        img = np.zeros((N, C, H + 2 * self.padding, W + 2 * self.padding))
        
        for y in range(self.kernel_size):
            y_max = y + self.stride * out_h
            for x in range(self.kernel_size):
                x_max = x + self.stride * out_w
                img[:, :, y:y_max:self.stride, x:x_max:self.stride] += col[:, :, y, x, :, :]
                
        return img[:, :, self.padding:H + self.padding, self.padding:W + self.padding]
        
    def forward(self, x: Tensor) -> Tensor:
        N, C, H, W = x.data.shape
        out_h = (H + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_w = (W + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        col = self._im2col(x.data)
        weight_col = self.params["weight"].data.reshape(self.out_channels, -1).T
        
        out = col @ weight_col
        out = out.reshape(N, out_h, out_w, self.out_channels).transpose(0, 3, 1, 2)
        
        if self.params["bias"] is not None:
            out += self.params["bias"].data.reshape(1, -1, 1, 1)
            
        result = Tensor(out, requires_grad=x.requires_grad)
        
        if result.requires_grad:
            def _backward(grad):
                if x.requires_grad:
                    dcol = grad.reshape(N * out_h * out_w, -1) @ self.params["weight"].data.reshape(self.out_channels, -1)
                    dx = self._col2im(dcol, x.data.shape)
                    x.backward(dx)
                    
                dw = x.data.transpose(1, 2, 3, 0).reshape(-1, N) @ grad.transpose(0, 2, 3, 1).reshape(N, -1)
                self.params["weight"].backward(dw.reshape(self.params["weight"].shape))
                
                if self.params["bias"] is not None:
                    db = grad.sum(axis=(0, 2, 3))
                    self.params["bias"].backward(db)
                    
            result._grad_fn = _backward
            result.is_leaf = False
            
        return result 