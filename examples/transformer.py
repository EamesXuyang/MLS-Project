import numpy as np
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import torch
from fudanai.tensor import Tensor
from fudanai.models.transformer import Transformer
from fudanai.losses.loss import CrossEntropyLoss

# 假设超参数
src_vocab_size = 50
tgt_vocab_size = 60
seq_len = 10
batch_size = 2
d_model = 32
num_heads = 4
num_layers = 2
d_ff = 64

# 构造模型
model = Transformer(src_vocab_size, tgt_vocab_size, d_model=d_model, num_heads=num_heads,
                    num_layers=num_layers, d_ff=d_ff, max_len=seq_len)

# 构造输入（src 和 tgt 是 int 类型的 token index）
src_tokens = np.random.randint(0, src_vocab_size, size=(batch_size, seq_len))
tgt_tokens = np.random.randint(0, tgt_vocab_size, size=(batch_size, seq_len))

# 构造 Tensor（需要 requires_grad=True 的仅是模型权重，因此输入不需要）
src = Tensor(src_tokens)
tgt = Tensor(tgt_tokens)

# 构造 mask：假设 src 中没有 padding，mask 全为 True（或 1）
src_mask = Tensor(np.ones((batch_size, 1, 1, seq_len)))

# 前向传播
output = model(src, tgt, src_mask)  # (batch, seq_len, tgt_vocab)

# 构造目标（通常是 tgt shifted by 1），这里我们用 tgt 自身做监督
target = Tensor(tgt_tokens)

# 损失函数：假设你有 CrossEntropyLoss（内部会自动 softmax）
criterion = CrossEntropyLoss()

# 将 output reshape 为 (batch * seq_len, vocab_size)，target 为 (batch * seq_len,)
logits = output.reshape((-1, tgt_vocab_size))
target = target.reshape(-1)

# 计算损失并反向传播
loss = criterion(logits, target)  # scalar Tensor
print("Loss:", loss.data)
loss.backward()

# 检查梯度是否传播
param_with_grad = [p for p in model.parameters().values() if p.requires_grad and p.grad is not None]
print(f"有梯度的参数数量: {len(param_with_grad)}")
assert len(param_with_grad) > 0, "模型中没有参数获得梯度，反向传播失败"

print("✅ 前向传播和反向传播验证通过！")
