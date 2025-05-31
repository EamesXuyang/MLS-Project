import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
from fudanai.tensor import Tensor
from fudanai.layers.lstm import LSTM
from fudanai.layers.base import Layer
from fudanai.layers.linear import Linear
from fudanai.losses.loss import MSELoss
from fudanai.optimizers.optimizer import SGD  

class SimpleLSTMModel(Layer):
    def __init__(self, input_size, hidden_size, output_size):
        self.lstm = LSTM(input_size, hidden_size)
        self.fc = Linear(hidden_size, output_size)

    def __call__(self, x: Tensor) -> Tensor:
        output, (h_n, c_n) = self.lstm(x)
        # 只取最后一步的输出
        return self.fc(h_n)


def test_lstm_fit_sum():
    # 模拟任务：输入序列的和 -> 模拟成回归问题
    seq_len, batch_size, input_size = 5, 100, 1
    hidden_size, output_size = 8, 1

    model = SimpleLSTMModel(input_size, hidden_size, output_size)
    criterion = MSELoss()
    optimizer = SGD(model.parameters().values(), lr=0.1)

    x_np = np.random.rand(seq_len, batch_size, input_size).astype(np.float32)
    y_np = np.sum(x_np, axis=0, keepdims=False).reshape(batch_size, output_size)

    x = Tensor(x_np, requires_grad=True)
    y_true = Tensor(y_np)
    for epoch in range(300):
        # 前向
        y_pred = model(x)
        loss = criterion(y_pred, y_true)

        # 反向
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        if epoch % 20 == 0 or epoch == 299:
            print(f"Epoch {epoch}: Loss = {loss.data.item():.6f}")

    print("LSTM 拟合完成 ✔")


if __name__ == "__main__":
    test_lstm_fit_sum()
