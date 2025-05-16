import numpy as np
from typing import Tuple, Optional
from .base import Layer
from ..tensor import Tensor

class LSTM(Layer):
    def __init__(self, input_size: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        
        # Initialize weights for input-to-hidden connections
        self.params["weight_ih"] = Tensor(
            np.random.randn(input_size, 4 * hidden_size) * np.sqrt(2.0 / input_size),
            requires_grad=True
        )
        
        # Initialize weights for hidden-to-hidden connections
        self.params["weight_hh"] = Tensor(
            np.random.randn(hidden_size, 4 * hidden_size) * np.sqrt(2.0 / hidden_size),
            requires_grad=True
        )
        
        if bias:
            self.params["bias_ih"] = Tensor(np.zeros(4 * hidden_size), requires_grad=True)
            self.params["bias_hh"] = Tensor(np.zeros(4 * hidden_size), requires_grad=True)
        else:
            self.params["bias_ih"] = None
            self.params["bias_hh"] = None
            
    def _sigmoid(self, x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))
        
    def forward(self, x: Tensor, init_states: Optional[Tuple[Tensor, Tensor]] = None) -> Tuple[Tensor, Tuple[Tensor, Tensor]]:
        batch_size, seq_len, _ = x.data.shape
        
        if init_states is None:
            h_t = Tensor(np.zeros((batch_size, self.hidden_size)))
            c_t = Tensor(np.zeros((batch_size, self.hidden_size)))
        else:
            h_t, c_t = init_states
            
        hidden_seq = []
        
        for t in range(seq_len):
            x_t = Tensor(x.data[:, t, :])
            
            # Calculate gates
            gates = (x_t @ self.params["weight_ih"]) + (h_t @ self.params["weight_hh"])
            
            if self.params["bias_ih"] is not None:
                gates = gates + self.params["bias_ih"]
            if self.params["bias_hh"] is not None:
                gates = gates + self.params["bias_hh"]
                
            # Split gates
            i_t, f_t, g_t, o_t = np.split(gates.data, 4, axis=1)
            
            # Apply activations
            i_t = self._sigmoid(i_t)
            f_t = self._sigmoid(f_t)
            g_t = np.tanh(g_t)
            o_t = self._sigmoid(o_t)
            
            # Update cell state
            c_t = Tensor(f_t * c_t.data + i_t * g_t)
            
            # Update hidden state
            h_t = Tensor(o_t * np.tanh(c_t.data))
            
            hidden_seq.append(h_t.data)
            
        hidden_seq = np.stack(hidden_seq, axis=1)
        
        result = Tensor(hidden_seq, requires_grad=x.requires_grad)
        
        if result.requires_grad:
            def _backward(grad):
                # Implement backward pass for LSTM
                # This is a simplified version and doesn't handle all gradients
                if x.requires_grad:
                    dx = grad @ self.params["weight_ih"].data.T
                    x.backward(dx)
                    
            result._grad_fn = _backward
            result.is_leaf = False
            
        return result, (h_t, c_t) 