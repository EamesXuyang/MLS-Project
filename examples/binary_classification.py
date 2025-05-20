import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fudanai.tensor import Tensor
from fudanai.layers.linear import Linear
from fudanai.optimizers.optimizer import Adam
from fudanai.activations.activation import Sigmoid

def generate_data(n_samples=100):
    # 生成二分类数据
    x = np.random.randn(n_samples, 2)
    # 使用简单的决策边界：x1 + x2 > 0
    y = (x[:, 0] + x[:, 1] > 0).astype(np.float32).reshape(-1, 1)
    return x, y

def main():
    # 生成数据
    x_train, y_train = generate_data()
    
    # 创建模型
    model = Linear(2, 1)  # 输入维度2，输出维度1
    sigmoid = Sigmoid()
    
    # 创建优化器
    optimizer = Adam(model.parameters().values(), lr=0.01)
    
    # 训练循环
    n_epochs = 100
    batch_size = 10
    
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
            pred = sigmoid(logits)
            
            # 计算损失（二元交叉熵）
            loss = -(y_batch * pred.log() + (1 - y_batch) * (1 - pred).log()).sum() / batch_size
            total_loss += loss.data
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        avg_loss = total_loss / n_batches
        if (epoch + 1) % 10 == 0:
            print(f"Epoch {epoch + 1}/{n_epochs}, Average Loss: {avg_loss:.4f}")
    
    # 测试模型
    x_test = np.array([[1, 1], [-1, -1], [1, -1], [-1, 1]])
    x_test_tensor = Tensor(x_test)
    logits = model.forward(x_test_tensor)
    pred = sigmoid(logits)
    
    print("\n测试结果：")
    for i, (x, p) in enumerate(zip(x_test, pred.data)):
        print(f"输入: {x}, 预测概率: {p[0]:.4f}, 预测类别: {int(p[0] > 0.5)}")

if __name__ == "__main__":
    main() 