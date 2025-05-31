import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fudanai.tensor import Tensor
from fudanai.layers.linear import Linear
from fudanai.optimizers.optimizer import Adam
from fudanai.activations.activation import ReLU
from fudanai.losses.loss import CrossEntropyLoss
from fudanai.layers.base import Layer

class MLP(Layer):
    def __init__(self, input_size, hidden_size, output_size):
        self.fc1 = Linear(input_size, hidden_size)
        self.fc2 = Linear(hidden_size, hidden_size)
        self.fc3 = Linear(hidden_size, output_size)
        self.relu = ReLU()
        
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        x = self.relu(x)
        x = self.fc3(x)
        return x

def generate_data(n_samples=1000):
    # 生成多分类数据
    x = np.random.randn(n_samples, 4)
    # 使用简单的规则生成标签：根据x的符号组合
    y = np.zeros((n_samples, 3))
    for i in range(n_samples):
        if x[i, 0] > 0 and x[i, 1] > 0:
            y[i, 0] = 1
        elif x[i, 2] > 0 and x[i, 3] > 0:
            y[i, 1] = 1
        else:
            y[i, 2] = 1
    return x, y

def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)

def main():
    # 生成数据
    x_train, y_train = generate_data()
    
    # 创建模型
    model = MLP(4, 64, 3)  # 输入维度4，隐藏层64，输出维度3
    
    # 创建优化器
    optimizer = Adam(model.parameters().values(), lr=0.001)
    criterion = CrossEntropyLoss()
    
    # 训练循环
    n_epochs = 50
    batch_size = 32
    
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = len(x_train) // batch_size
        
        for i in range(n_batches):
            # 获取批次数据
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            x_batch = Tensor(x_train[start_idx:end_idx])
            y_batch = Tensor(y_train[start_idx:end_idx])
            
            # 前向传播
            logits = model.forward(x_batch)
            
            # 计算损失（交叉熵）
            loss = criterion(logits, y_batch)
            total_loss = total_loss + loss.data.item()
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        avg_loss = total_loss / n_batches
        if (epoch + 1) % 5 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Average Loss: {avg_loss:.4f}")
    
    # 测试模型
    x_test = np.array([
        [1, 1, -1, -1],  # 应该预测为类别0
        [-1, -1, 1, 1],  # 应该预测为类别1
        [1, -1, 1, -1],  # 应该预测为类别2
    ])
    x_test_tensor = Tensor(x_test)
    logits = model.forward(x_test_tensor)
    probs = softmax(logits.data)
    
    print("\n测试结果：")
    for i, (x, p) in enumerate(zip(x_test, probs)):
        pred_class = np.argmax(p)
        print(f"输入: {x}, 预测类别: {pred_class}, 预测概率: {p[pred_class]:.4f}")

if __name__ == "__main__":
    main() 