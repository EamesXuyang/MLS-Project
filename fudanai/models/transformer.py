import random
import numpy as np
from typing import Optional

from ..activations.activation import ReLU
from ..layers.base import Layer
from ..layers.linear import Linear
from ..tensor import Tensor
import math

class PositionalEncoding(Layer):
    def __init__(self, d_model:int, max_len=5000):
        super().__init__()
        self.params = {}
        pe = Tensor.zeros((max_len, d_model))
        pos = Tensor([i for i in range(max_len)]).unsqueeze(1)
        div = Tensor([i * math.log(10000.0) / d_model for i in range(0, d_model, 2)]).exp()
        pe[:, 0::2] = (pos * div).sin()
        pe[:, 1::2] = (pos * div).cos()
        self.params['pe'] = pe.unsqueeze(0)

    def forward(self, x: Tensor):
        x = x + self.params['pe'][:, :x.shape[1]]
        return x

class LayerNorm(Layer):
    def __init__(self, d_model, eps=1e-5):
        super().__init__()
        self.eps = eps
        self.params = {}
        self.params['gamma'] = Tensor.ones((d_model,), requires_grad=True)
        self.params['beta'] = Tensor.zeros((d_model,), requires_grad=True)
    
    def forward(self, x: Tensor):
        mean = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        x_norm = (x - mean) / (var + self.eps).sqrt()
        return self.params['gamma'] * x_norm + self.params['beta']

class Embedding(Layer):
    def __init__(self, vocab_size, d_model):
        super().__init__()
        self.params = {}
        self.params['embedding'] = Tensor(np.random.randn(vocab_size, d_model), requires_grad=True)
        self.vocab_size = vocab_size
        self.d_model = d_model

    def forward(self, x: Tensor):
        return self.params['embedding'][x]
    
    
class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super().__init__()
        assert d_model % num_heads == 0
        self.d_k = d_model // num_heads
        self.num_heads = num_heads

        self.q_proj = Linear(d_model, d_model)
        self.k_proj = Linear(d_model, d_model)
        self.v_proj = Linear(d_model, d_model)
        self.o_proj = Linear(d_model, d_model)

    def forward(self, q: Tensor, k: Tensor, v: Tensor, mask: Optional[Tensor] = None):
        B, L, D = q.shape
        q = self.q_proj(q)
        k = self.k_proj(k)
        v = self.v_proj(v)

        q = q.reshape((B, -1, self.num_heads, self.d_k)).transpose((1, 2))
        k = k.reshape((B, -1, self.num_heads, self.d_k)).transpose((1, 2))
        v = v.reshape((B, -1, self.num_heads, self.d_k)).transpose((1, 2))

        scores = q @ k.transpose((-2, -1)) / Tensor(self.d_k).sqrt()
        if mask is not None:
            scores = scores.masked_fill(mask == 0, 1e-9)
        def softmax(y, axis=-1):
            return (y - y.max(axis=axis, keepdims=True)).exp() / (y - y.max(axis=axis, keepdims=True)).exp().sum(axis=axis, keepdims=True)
        att = softmax(scores)
        out = att @ v

        out = out.transpose((1, 2)).reshape((B, -1, self.num_heads * self.d_k))
        out = self.o_proj(out)
        return out
    
class FeedForward(Layer):
    def __init__(self, d_model, d_ff):
        super().__init__()
        self.linear1 = Linear(d_model, d_ff)
        self.relu = ReLU()
        self.linear2 = Linear(d_ff, d_model)

    def forward(self, x: Tensor):
        return self.linear2(self.relu(self.linear1(x)))
    

class EncoderLayer(Layer):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)

    def forward(self, x: Tensor, src_mask: Optional[Tensor] = None):
        x = self.norm1(x + self.self_attn(x, x, x, src_mask))
        x = self.norm2(x + self.ff(x))
        return x

class DecoderLayer(Layer):
    def __init__(self, d_model, num_heads, d_ff):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.enc_attn = MultiHeadAttention(d_model, num_heads)
        self.ff = FeedForward(d_model, d_ff)
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        self.norm3 = LayerNorm(d_model)

    def forward(self, x, enc_out: Tensor, tgt_amsk: Optional[Tensor] = None, memory_mask: Optional[Tensor]=None):
        x = self.norm1(x + self.self_attn(x, x, x, tgt_amsk))
        x = self.norm2(x + self.enc_attn(x, enc_out, enc_out, memory_mask))
        x = self.norm3(x + self.ff(x))
        return x

class Transformer(Layer):
    def __init__(self, src_vocab, tgt_vocab, d_model=512, num_heads=8, num_layers=6, d_ff=2048, max_len=100):
        super().__init__()
        self.src_embed = Embedding(src_vocab, d_model)
        self.tgt_embed = Embedding(tgt_vocab, d_model)
        self.pos_enc = PositionalEncoding(d_model, max_len)
        self.encoder = [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.decoder = [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]
        self.out = Linear(d_model, tgt_vocab)
    
    def make_tgt_mask(self, tgt: Tensor):
        size = tgt.shape[1]
        # TODO
        mask = Tensor(np.tril(np.ones((size, size))), device=tgt.device)
        return mask.unsqueeze(0).unsqueeze(1)

    def forward(self, src, tgt, src_mask):
        src = self.pos_enc(self.src_embed(src))
        tgt = self.pos_enc(self.tgt_embed(tgt))

        for layer in self.encoder:
            src = layer(src, src_mask)

        tgt_mask = self.make_tgt_mask(tgt)
        for layer in self.decoder:
            tgt = layer(tgt, src, tgt_mask, src_mask)

        out = self.out(tgt)
        return out
