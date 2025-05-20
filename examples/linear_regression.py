import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fudanai.tensor import Tensor
from fudanai.layers.linear import Linear
from fudanai.optimizers.optimizer import SGD

def generate_data(n_samples=100):
    # 生成 y = 2x + 1 + noise 的数据
    x = np.random.randn(n_samples, 1)
    y = 2 * x + 1 + 0.1 * np.random.randn(n_samples, 1)
    return x, y

def main():
    # 生成数据
    x_train, y_train = generate_data(n_samples=1000)  # 增加样本数量
    
    # 创建模型
    model = Linear(1, 1)  # 输入维度1，输出维度1
    
    # 创建优化器
    optimizer = SGD(model.parameters().values(), lr=0.001)  # 降低学习率
    
    # 训练循环
    n_epochs = 200  # 增加训练轮数
    batch_size = 32  # 增加批次大小
    
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
            pred = model.forward(x_batch)
            
            # 计算损失（MSE）
            loss = ((pred - y_batch) ** 2).sum() / batch_size
            total_loss += loss.data
            
            # 反向传播
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        avg_loss = total_loss / n_batches
        if (epoch + 1) % 20 == 0:  # 每20轮打印一次
            print(f"Epoch {epoch + 1}/{n_epochs}, Average Loss: {avg_loss:.6f}")
    
    # 打印学习到的参数
    print("\n学习到的参数：")
    print(f"权重: {model.params['weight'].data[0][0]:.6f}")
    print(f"偏置: {model.params['bias'].data[0]:.6f}")
    
    # 打印真实参数
    print("\n真实参数：")
    print("权重: 2.000000")
    print("偏置: 1.000000")

if __name__ == "__main__":
    main() 