import numpy as np
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
from fudanai.tensor import Tensor
from fudanai.layers.lstm import LSTM as LSTMMy
import torch
import torch.nn as nn

class LSTMCellTorch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.W_x = nn.Linear(input_size, 4 * hidden_size, bias=False)
        self.W_h = nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        self.hidden_size = hidden_size

    def forward(self, x, h_prev, c_prev):
        gates = self.W_x(x) + self.W_h(h_prev)
        i, f, g, o = gates.chunk(4, dim=-1)

        i = torch.sigmoid(i)
        f = torch.sigmoid(f)
        g = torch.tanh(g)
        o = torch.sigmoid(o)

        c = f * c_prev + i * g
        h = o * torch.tanh(c)
        return h, c


class LSTMTorch(nn.Module):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.cell = LSTMCellTorch(input_size, hidden_size)
        self.hidden_size = hidden_size

    def forward(self, x, h0=None, c0=None):
        seq_len, batch_size, _ = x.shape
        if h0 is None:
            h = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            h = h0
        if c0 is None:
            c = torch.zeros(batch_size, self.hidden_size, dtype=x.dtype, device=x.device)
        else:
            c = c0

        outputs = []
        for t in range(seq_len):
            h, c = self.cell(x[t], h, c)
            outputs.append(h.unsqueeze(0))

        return torch.cat(outputs, dim=0), (h, c)

def compare_lstm_output_and_grad():
    seq_len, batch_size, input_size, hidden_size = 10000, 2, 3, 4
    np.random.seed(0)
    torch.manual_seed(0)

    # Create input
    x_np = np.random.randn(seq_len, batch_size, input_size).astype(np.float32)

    # FudanAI input
    x_my = Tensor(x_np.copy(), requires_grad=True)
    lstm_my = LSTMMy(input_size, hidden_size)
    out_my, _ = lstm_my(x_my)
    loss_my = out_my.sum()
    loss_my.backward()
    grad_my = x_my.grad.data.copy()
    print(lstm_my.parameters().keys())

    # Torch input
    x_torch = torch.tensor(x_np.copy(), requires_grad=True)
    lstm_torch = LSTMTorch(input_size, hidden_size)
    with torch.no_grad():
        lstm_torch.cell.W_x.weight.copy_(torch.from_numpy(lstm_my.cell.W_x.params['weight'].data.T))
        lstm_torch.cell.W_h.weight.copy_(torch.from_numpy(lstm_my.cell.W_h.params['weight'].data.T))

    out_torch, _ = lstm_torch(x_torch)
    loss_torch = out_torch.sum()
    loss_torch.backward()
    grad_torch = x_torch.grad.detach().numpy()

    # Compare
    print("Output difference:", np.abs(out_my.data - out_torch.detach().numpy()).max())
    print("Input gradient difference:", np.abs(grad_my - grad_torch).max())
    same_sign = np.sign(grad_my) == np.sign(grad_torch)
    sign_match_ratio = np.sum(same_sign) / same_sign.size
    print("Sign match ratio:", sign_match_ratio)


if __name__ == "__main__":
    compare_lstm_output_and_grad()
