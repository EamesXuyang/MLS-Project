import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fudanai.models.resnet import resnet18, resnet34
from fudanai.tensor import Tensor

def test_resnet_forward():
    # 创建一个随机输入张量：batch_size=2, channels=3, height=224, width=224
    input_data = np.random.randn(2, 3, 224, 224).astype(np.float32)
    input_tensor = Tensor(input_data, requires_grad=False)

    # 实例化模型
    model18 = resnet18(num_classes=10)
    print(model18.parameters().keys())
    model34 = resnet34(num_classes=10)
    print(model34.parameters().keys())

    # 前向传播
    output18 = model18(input_tensor)
    output34 = model34(input_tensor)

    # 输出形状检查
    assert output18.data.shape == (2, 10), f"resnet18 output shape mismatch: got {output18.data.shape}"
    assert output34.data.shape == (2, 10), f"resnet34 output shape mismatch: got {output34.data.shape}"

    print("ResNet18 and ResNet34 forward pass success.")
    print("ResNet18 output:", output18.data)
    print("ResNet34 output:", output34.data)

    # 测试反向传播是否报错
    loss18 = output18.sum()
    loss18.backward()  
    loss34 = output34.sum()
    loss34.backward() 

    print("Backward pass successful.")

if __name__ == "__main__":
    test_resnet_forward()
