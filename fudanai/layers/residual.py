import numpy as np
from typing import Optional, List
from .base import Layer
from .conv import Conv2d
from ..tensor import Tensor
from ..activations.activation import ReLU

class ResidualBlock(Layer):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stride: int = 1,
        downsample: Optional[Layer] = None
    ):
        super().__init__()
        self.conv1 = Conv2d(
            in_channels, out_channels,
            kernel_size=3, stride=stride, padding=1
        )
        self.bn1 = BatchNorm2d(out_channels)
        self.relu = ReLU()
        self.conv2 = Conv2d(
            out_channels, out_channels,
            kernel_size=3, stride=1, padding=1
        )
        self.bn2 = BatchNorm2d(out_channels)
        self.downsample = downsample
        
    def forward(self, x: Tensor) -> Tensor:
        identity = x
        
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        if self.downsample is not None:
            identity = self.downsample(x)
            
        out = out + identity
        out = self.relu(out)
        
        return out

class BatchNorm2d(Layer):
    def __init__(self, num_features: int, eps: float = 1e-5, momentum: float = 0.1):
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.momentum = momentum
        
        # 可学习参数
        self.params["gamma"] = Tensor(np.ones(num_features), requires_grad=True)
        self.params["beta"] = Tensor(np.zeros(num_features), requires_grad=True)
        
        # 运行时统计量
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        
    def forward(self, x: Tensor) -> Tensor:
        if self.training:
            # 计算批次统计量
            mean = np.mean(x.data, axis=(0, 2, 3))
            var = np.var(x.data, axis=(0, 2, 3))
            
            # 更新运行时统计量
            self.running_mean = (1 - self.momentum) * self.running_mean + self.momentum * mean
            self.running_var = (1 - self.momentum) * self.running_var + self.momentum * var
        else:
            mean = self.running_mean
            var = self.running_var
            
        # 归一化
        x_normalized = (x.data - mean.reshape(1, -1, 1, 1)) / np.sqrt(var.reshape(1, -1, 1, 1) + self.eps)
        
        # 缩放和平移
        out = self.params["gamma"].data.reshape(1, -1, 1, 1) * x_normalized + \
              self.params["beta"].data.reshape(1, -1, 1, 1)
              
        result = Tensor(out, requires_grad=x.requires_grad)
        
        if result.requires_grad:
            def _backward(grad):
                if x.requires_grad:
                    N, C, H, W = grad.shape
                    dbeta = grad.sum(axis=(0, 2, 3))
                    dgamma = (grad * x_normalized).sum(axis=(0, 2, 3))
                    
                    dx_normalized = grad * self.params["gamma"].data.reshape(1, -1, 1, 1)
                    dvar = (-0.5 * dx_normalized * (x.data - mean.reshape(1, -1, 1, 1)) / 
                           (var.reshape(1, -1, 1, 1) + self.eps) ** 1.5).sum(axis=(0, 2, 3))
                    dmean = (-dx_normalized / np.sqrt(var.reshape(1, -1, 1, 1) + self.eps)).sum(axis=(0, 2, 3))
                    
                    dx = (dx_normalized / np.sqrt(var.reshape(1, -1, 1, 1) + self.eps) + 
                         2 * dvar * (x.data - mean.reshape(1, -1, 1, 1)) / (N * H * W) +
                         dmean / (N * H * W))
                    
                    x.backward(dx)
                    self.params["gamma"].backward(dgamma)
                    self.params["beta"].backward(dbeta)
                    
            result._grad_fn = _backward
            result.is_leaf = False
            
        return result 