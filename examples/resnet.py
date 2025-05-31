import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fudanai.models.resnet import resnet18
from fudanai.tensor import Tensor
from fudanai.losses.loss import CrossEntropyLoss  # 你需要实现或已有这个
from fudanai.optimizers.optimizer import SGD            # 你需要实现或已有这个

def generate_dummy_data(num_samples=30, num_classes=2, input_shape=(3, 32, 32)):
    X = np.random.randn(num_samples, *input_shape).astype(np.float32)
    y = np.random.randint(0, num_classes, size=(num_samples,))
    return X, y

def test_resnet_fit():
    # 模拟数据
    num_classes = 2
    X_data, y_data = generate_dummy_data(num_samples=8, num_classes=num_classes)
    
    # 转成 Tensor
    inputs = Tensor(X_data, requires_grad=True)
    targets = y_data  # 注意：这里是整数标签

    # 模型、损失、优化器
    model = resnet18(num_classes=num_classes, input_channels=3)
    criterion = CrossEntropyLoss()
    optimizer = SGD(model.parameters().values(), lr=0.01)

    # 简单训练 10 轮
    for epoch in range(100):
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred = np.argmax(outputs.data, axis=1)
        acc = (pred == targets).mean()

        print(f"Epoch {epoch+1} | Loss: {loss.data:.4f} | Acc: {acc:.2f}")

if __name__ == "__main__":
    test_resnet_fit()
