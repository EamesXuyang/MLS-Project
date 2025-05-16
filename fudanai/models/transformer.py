import numpy as np
from typing import Optional
from ..layers.base import Layer
from ..layers.linear import Linear
from ..tensor import Tensor

class MultiHeadAttention(Layer):
    def __init__(self, embed_dim: int, num_heads: int, dropout: float = 0.0):
        super().__init__()
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        assert self.head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
        
        self.q_proj = Linear(embed_dim, embed_dim)
        self.k_proj = Linear(embed_dim, embed_dim)
        self.v_proj = Linear(embed_dim, embed_dim)
        self.out_proj = Linear(embed_dim, embed_dim)
        
        self.dropout = dropout
        
    def forward(
        self,
        query: Tensor,
        key: Tensor,
        value: Tensor,
        mask: Optional[Tensor] = None
    ) -> Tensor:
        batch_size = query.data.shape[0]
        
        # 线性变换
        q = self.q_proj(query)
        k = self.k_proj(key)
        v = self.v_proj(value)
        
        # 重塑为多头形式
        q = q.data.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        k = k.data.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        v = v.data.reshape(batch_size, -1, self.num_heads, self.head_dim).transpose(0, 2, 1, 3)
        
        # 注意力计算
        scores = np.matmul(q, k.transpose(0, 1, 3, 2)) / np.sqrt(self.head_dim)
        
        if mask is not None:
            scores = scores + mask.data.reshape(batch_size, 1, *mask.data.shape[1:])
            
        attn_weights = self._softmax(scores)
        
        if self.dropout > 0 and self.training:
            attn_weights = attn_weights * (np.random.random(attn_weights.shape) > self.dropout)
            attn_weights = attn_weights / (attn_weights.sum(axis=-1, keepdims=True) + 1e-8)
            
        attn_output = np.matmul(attn_weights, v)
        
        # 重塑回原始形状
        attn_output = attn_output.transpose(0, 2, 1, 3).reshape(batch_size, -1, self.embed_dim)
        
        # 最终线性变换
        output = self.out_proj(Tensor(attn_output))
        
        return output
        
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

class TransformerEncoderLayer(Layer):
    def __init__(
        self,
        d_model: int,
        nhead: int,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        self.self_attn = MultiHeadAttention(d_model, nhead, dropout)
        
        self.linear1 = Linear(d_model, dim_feedforward)
        self.linear2 = Linear(dim_feedforward, d_model)
        
        self.norm1 = LayerNorm(d_model)
        self.norm2 = LayerNorm(d_model)
        
        self.dropout = dropout
        self.activation = lambda x: np.maximum(0, x)  # ReLU
        
    def forward(self, src: Tensor, src_mask: Optional[Tensor] = None) -> Tensor:
        # 自注意力
        attn_output = self.self_attn(src, src, src, src_mask)
        
        # 第一个残差连接和层归一化
        src = src + self._dropout(attn_output.data)
        src = self.norm1(Tensor(src))
        
        # 前馈网络
        ff_output = self.linear2(Tensor(self.activation(self.linear1(src).data)))
        
        # 第二个残差连接和层归一化
        src = src + self._dropout(ff_output.data)
        src = self.norm2(Tensor(src))
        
        return src
        
    def _dropout(self, x: np.ndarray) -> np.ndarray:
        if self.dropout > 0 and self.training:
            mask = np.random.random(x.shape) > self.dropout
            return x * mask / (1 - self.dropout)
        return x

class LayerNorm(Layer):
    def __init__(self, normalized_shape: int, eps: float = 1e-5):
        super().__init__()
        self.eps = eps
        self.params["weight"] = Tensor(np.ones(normalized_shape), requires_grad=True)
        self.params["bias"] = Tensor(np.zeros(normalized_shape), requires_grad=True)
        
    def forward(self, x: Tensor) -> Tensor:
        mean = np.mean(x.data, axis=-1, keepdims=True)
        var = np.var(x.data, axis=-1, keepdims=True)
        
        x_norm = (x.data - mean) / np.sqrt(var + self.eps)
        
        return Tensor(
            self.params["weight"].data * x_norm + self.params["bias"].data,
            requires_grad=x.requires_grad
        )

class Transformer(Layer):
    def __init__(
        self,
        d_model: int = 512,
        nhead: int = 8,
        num_encoder_layers: int = 6,
        num_decoder_layers: int = 6,
        dim_feedforward: int = 2048,
        dropout: float = 0.1
    ):
        super().__init__()
        
        # 编码器
        self.encoder_layers = [
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_encoder_layers)
        ]
        
        # 解码器
        self.decoder_layers = [
            TransformerEncoderLayer(d_model, nhead, dim_feedforward, dropout)
            for _ in range(num_decoder_layers)
        ]
        
        self.d_model = d_model
        
    def forward(
        self,
        src: Tensor,
        tgt: Tensor,
        src_mask: Optional[Tensor] = None,
        tgt_mask: Optional[Tensor] = None
    ) -> Tensor:
        # 编码器前向传播
        memory = src
        for encoder_layer in self.encoder_layers:
            memory = encoder_layer(memory, src_mask)
            
        # 解码器前向传播
        output = tgt
        for decoder_layer in self.decoder_layers:
            output = decoder_layer(output, tgt_mask)
            
        return output
        
def create_transformer(
    src_vocab_size: int,
    tgt_vocab_size: int,
    d_model: int = 512,
    nhead: int = 8,
    num_encoder_layers: int = 6,
    num_decoder_layers: int = 6,
    dim_feedforward: int = 2048,
    dropout: float = 0.1
) -> Transformer:
    """创建一个完整的Transformer模型"""
    model = Transformer(
        d_model=d_model,
        nhead=nhead,
        num_encoder_layers=num_encoder_layers,
        num_decoder_layers=num_decoder_layers,
        dim_feedforward=dim_feedforward,
        dropout=dropout
    )
    
    # 添加词嵌入层
    model.src_embed = Linear(src_vocab_size, d_model)
    model.tgt_embed = Linear(tgt_vocab_size, d_model)
    
    # 添加位置编码
    model.pos_encoder = PositionalEncoding(d_model, dropout)
    
    return model

class PositionalEncoding(Layer):
    def __init__(self, d_model: int, dropout: float = 0.1, max_len: int = 5000):
        super().__init__()
        self.dropout = dropout
        
        position = np.arange(max_len)[:, np.newaxis]
        div_term = np.exp(np.arange(0, d_model, 2) * (-np.log(10000.0) / d_model))
        
        pe = np.zeros((max_len, d_model))
        pe[:, 0::2] = np.sin(position * div_term)
        pe[:, 1::2] = np.cos(position * div_term)
        
        self.register_buffer("pe", pe)
        
    def forward(self, x: Tensor) -> Tensor:
        x = x + self.pe[:x.data.shape[1]]
        return self._dropout(x)
        
    def _dropout(self, x: Tensor) -> Tensor:
        if self.dropout > 0 and self.training:
            mask = np.random.random(x.data.shape) > self.dropout
            return Tensor(x.data * mask / (1 - self.dropout))
        return x 