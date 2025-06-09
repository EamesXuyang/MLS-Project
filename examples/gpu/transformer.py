import numpy as np
import os
import sys
import time
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))
import torch  # 仅用于验证时可选导入
from fudanai.tensor import Tensor
from fudanai.models.transformer import Transformer
from fudanai.losses.loss import CrossEntropyLoss
from fudanai.optimizers.optimizer import SGD 

# 超参数
src_vocab_size = 50
tgt_vocab_size = 60
seq_len = 10
batch_size = 2
d_model = 128
num_heads = 4
num_layers = 2
d_ff = 256
n_epochs = 100
learning_rate = 0.01

# 模型构造
model = Transformer(src_vocab_size, tgt_vocab_size, d_model=d_model, num_heads=num_heads,
                    num_layers=num_layers, d_ff=d_ff, max_len=seq_len)

model.to('cuda')
# 损失函数与优化器
criterion = CrossEntropyLoss()
optimizer = SGD(model.parameters().values(), lr=learning_rate)

# 构造随机输入数据
src_tokens = np.random.randint(0, src_vocab_size, size=(batch_size, seq_len))
tgt_tokens = np.random.randint(0, tgt_vocab_size, size=(batch_size, seq_len))

src = Tensor(src_tokens).to('cuda')
tgt = Tensor(tgt_tokens).to('cuda')

start_time = time.time()
for epoch in range(1, n_epochs + 1):
    src_mask = Tensor(np.ones((batch_size, 1, 1, seq_len))).to('cuda')  # 假设无 padding

    # 前向传播
    output = model(src, tgt, src_mask)  # (batch, seq_len, tgt_vocab)
    target = Tensor(tgt_tokens).to('cuda')

    # reshape
    logits = output.reshape((-1, tgt_vocab_size))
    target = target.reshape(-1)

    # 损失计算
    loss = criterion(logits, target)

    # 反向传播与参数更新
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # 输出
    if epoch % 10 == 0:
        print(f"[Epoch {epoch}] Loss: {loss.data}")
end_time = time.time()

print(f"多轮训练测试完成, 耗时: {end_time - start_time} 秒")
