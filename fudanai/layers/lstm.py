from typing import Tuple, Optional
from .base import Layer
from .linear import Linear
from ..tensor import Tensor



class LSTMCell(Layer):
    def __init__(self, input_size: int, hidden_size: int):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.params = {}

        self.W_x = Linear(input_size, 4 * hidden_size, bias=False)
        self.W_h = Linear(hidden_size, 4 * hidden_size, bias=False)
    
    def forward(self, x: Tensor, h_prev: Tensor, c_prev: Tensor) -> Tuple[Tensor, Tensor]:
        gates = self.W_x(x) + self.W_h(h_prev)
        i, f, g, o = gates.chunk(4, axis=-1)

        i = i.sigmoid()
        f = f.sigmoid()
        g = g.tanh()
        o = o.sigmoid()

        c = f * c_prev + i * g
        h = o * c.tanh()

        return h, c
    
    def __call__(self, x: Tensor, h_prev: Tensor, c_prev: Tensor) -> Tuple[Tensor, Tensor]:
        return self.forward(x, h_prev, c_prev)

class LSTM(Layer):
    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.params = {}
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.cell = LSTMCell(input_size, hidden_size)

    def forward(self, x: Tensor, h_prev: Optional[Tensor] = None, c_prev: Optional[Tensor] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        sql_len, batch_size, _ = x.shape
        if h_prev is None:
            h = Tensor.zeros((batch_size, self.hidden_size), device=x.device)
        if c_prev is None:
            c = Tensor.zeros((batch_size, self.hidden_size), device=x.device)
        
        outputs = []

        for t in range(sql_len):
            h, c = self.cell(x[t], h, c)
            outputs.append(h.unsqueeze(0))
        
        return Tensor.concat(outputs, axis=0), (h, c)
