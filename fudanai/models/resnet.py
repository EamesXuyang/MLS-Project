import numpy as np
from typing import List, Type
from ..layers.base import Layer
from ..layers.conv import Conv2d
from ..layers.linear import Linear
from ..layers.residual import ResidualBlock, BatchNorm2d
from ..activations.activation import ReLU
from ..tensor import Tensor

class ResNet(Layer):
    def __init__(
        self,
        block: Type[ResidualBlock],
        layers: List[int],
        num_classes: int = 1000,
        input_channels: int = 3
    ):
        super().__init__()
        self.in_channels = 64
        
        # 初始卷积层
        self.conv1 = Conv2d(
            input_channels, 64,
            kernel_size=7, stride=2, padding=3
        )
        self.bn1 = BatchNorm2d(64)
        self.relu = ReLU()
        
        # 残差层
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)
        
        # 分类头
        self.avgpool = lambda x: Tensor(
            np.mean(x.data, axis=(2, 3), keepdims=True)
        )
        self.fc = Linear(512, num_classes)
        
    def _make_layer(
        self,
        block: Type[ResidualBlock],
        out_channels: int,
        blocks: int,
        stride: int = 1
    ) -> Layer:
        downsample = None
        if stride != 1 or self.in_channels != out_channels:
            downsample = Sequential([
                Conv2d(
                    self.in_channels, out_channels,
                    kernel_size=1, stride=stride
                ),
                BatchNorm2d(out_channels)
            ])
            
        layers = []
        layers.append(
            block(self.in_channels, out_channels, stride, downsample)
        )
        
        self.in_channels = out_channels
        for _ in range(1, blocks):
            layers.append(block(out_channels, out_channels))
            
        return Sequential(layers)
        
    def forward(self, x: Tensor) -> Tensor:
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        
        x = self.avgpool(x)
        x = Tensor(x.data.reshape(x.data.shape[0], -1))
        x = self.fc(x)
        
        return x

class Sequential(Layer):
    def __init__(self, layers: List[Layer]):
        super().__init__()
        self.layers = layers
        
    def forward(self, x: Tensor) -> Tensor:
        for layer in self.layers:
            x = layer(x)
        return x
        
def resnet18(num_classes: int = 1000, input_channels: int = 3) -> ResNet:
    """构建ResNet-18模型"""
    return ResNet(ResidualBlock, [2, 2, 2, 2],
                 num_classes=num_classes,
                 input_channels=input_channels)
    
def resnet34(num_classes: int = 1000, input_channels: int = 3) -> ResNet:
    """构建ResNet-34模型"""
    return ResNet(ResidualBlock, [3, 4, 6, 3],
                 num_classes=num_classes,
                 input_channels=input_channels) 