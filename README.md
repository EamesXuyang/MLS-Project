# FudanAI

FudanAI 是一个基于 NumPy 的深度学习框架，提供了类似 PyTorch 的 API 和功能。该框架支持自动微分，并实现了多种常用的神经网络层、损失函数和优化器。

## 特性

- 基于 NumPy 的张量操作
- 自动微分系统
- 常用神经网络层的实现：
  - 全连接层 (Linear)
  - 卷积层 (Conv2d)
  - LSTM
- 激活函数：
  - ReLU
  - Sigmoid
  - Tanh
- 损失函数：
  - MSE Loss
  - Cross Entropy Loss
- 优化器：
  - SGD（支持动量）
  - Adam
- 联邦学习

## 快速开始

* 以下是一个简单的示例，展示如何使用 FudanAI 创建和训练一个神经网络：

```python
import numpy as np
from fudanai.tensor import Tensor
from fudanai.layers.linear import Linear
from fudanai.activations.activation import ReLU
from fudanai.losses.loss import MSELoss
from fudanai.optimizers.optimizer import Adam

# 创建模型
class SimpleModel:
    def __init__(self):
        self.fc1 = Linear(10, 5)
        self.fc2 = Linear(5, 1)
        self.relu = ReLU()
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# 准备数据
X = Tensor(np.random.randn(100, 10))
y = Tensor(np.random.randn(100, 1))

# 创建模型和优化器
model = SimpleModel()
criterion = MSELoss()
optimizer = Adam([p for layer in [model.fc1, model.fc2] 
                 for p in layer.parameters().values()], lr=0.01)

# 训练循环
for epoch in range(100):
    # 前向传播
    pred = model.forward(X)
    loss = criterion(pred, y)
  
    # 反向传播
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
  
    if (epoch + 1) % 10 == 0:
        print(f"Epoch {epoch+1}, Loss: {loss.data:.4f}")
```

* examples/fed_linear_regression.py 是一个联邦线性回归的示例，展示了如何使用 FudanAI 实现联邦学习。
  * 首先使用python -m fudanai.fed.server.server以及python -m fudanai.fed.client.client启动服务器和客户端。
  * 然后运行python fed_linear_regression.py即可进行联邦学习示例训练。

更多示例可以在 `examples` 目录中找到。

## 项目结构

```
fudanai/
├── tensor.py          # 张量类实现
├── layers/            # 神经网络层
│   ├── base.py       # 基础层类
│   ├── linear.py     # 全连接层
│   ├── conv.py       # 卷积层
│   └── lstm.py       # LSTM层
├── activations/       # 激活函数
│   └── activation.py
├── losses/           # 损失函数
│   └── loss.py
├── optimizers/       # 优化器
│   └── optimizer.py
├── fed/       # 联邦学习实现
│   ├── server    # 服务端
│   ├── client    # 客户端
│   ├── aggregate # 聚合方法
│   └── task.py   # 创建并提交训练任务
└── utils/           # 工具函数
```
