import numpy as np
import torch
import torch.nn.functional as F
import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fudanai.tensor import Tensor


def test_operation(name, func_fudan, func_torch, input_data, atol=1e-5):
    print(f"\nTesting {name}")
    input_data = input_data.astype(np.float32)

    # ==== PyTorch Part ====
    x_torch = torch.tensor(input_data, dtype=torch.float32, requires_grad=True)
    y_torch = func_torch(x_torch)
    loss_torch = y_torch.sum()
    loss_torch.backward()
    expected_grad = x_torch.grad.detach().numpy()

    # ==== FudanAI Tensor Part ====
    x_fudan = Tensor(input_data, requires_grad=True, device="cuda")
    y_fudan = func_fudan(x_fudan)
    loss_fudan = y_fudan.sum()
    loss_fudan.backward()
    actual_grad = x_fudan.grad.data.get()


    if not np.allclose(actual_grad, expected_grad, atol=atol, equal_nan=True):
        print(f"Expected grad (PyTorch):\n{expected_grad}")
        print(f"Actual grad (FudanAI):\n{actual_grad}")
        print(f"{name} failed! Max error: {np.abs(actual_grad - expected_grad).max()}")
    else:
        print(f"{name} passed.")


if __name__ == "__main__":
    np.random.seed(42)
    shape = (3, 3)
    a = np.random.randn(*shape).astype(np.float32)
    b = np.random.randn(*shape).astype(np.float32)
    c = np.random.randn(shape[0]).astype(np.float32)

    tensor_b_fudan = Tensor(b, device="cuda")
    tensor_b_torch = torch.tensor(b, dtype=torch.float32)
    tensor_c_fudan = Tensor(c, device="cuda")
    tensor_c_torch = torch.tensor(c, dtype=torch.float32)

    # Basic operations

    test_operation("getitem",
                   lambda x: x[[0, 2, 0]],
                   lambda x: x[[0, 2, 0]],
                   a)
    
    test_operation("neg",
                   lambda x: -x,
                   lambda x: -x,
                   a)
    
    test_operation("add",
                   lambda x: x + tensor_b_fudan,
                   lambda x: x + tensor_b_torch,
                   a)

    test_operation("broadcast_add",
                   lambda x: x + tensor_c_fudan,
                   lambda x: x + tensor_c_torch,
                   a)
    
    test_operation("radd",
                   lambda x: tensor_b_fudan + x,
                   lambda x: tensor_b_torch + x,
                   a)

    test_operation("broadcast_radd",
                   lambda x: tensor_c_fudan + x,
                   lambda x: tensor_c_torch + x,
                   a)

    test_operation("sub",
                lambda x: x - tensor_b_fudan,
                lambda x: x - tensor_b_torch,
                a)
    
    test_operation("broadcast_sub",
                lambda x: x - tensor_c_fudan,
                lambda x: x - tensor_c_torch,
                a)

    test_operation("rsub",
                lambda x: tensor_b_fudan - x,
                lambda x: tensor_b_torch - x,
                a)
    
    test_operation("broadcast_rsub",
                lambda x: tensor_c_fudan - x,
                lambda x: tensor_c_torch - x,
                a)

    test_operation("mul",
                   lambda x: x * tensor_b_fudan,
                   lambda x: x * tensor_b_torch,
                   a)

    test_operation("broadcast_mul",
                   lambda x: x * tensor_c_fudan,
                   lambda x: x * tensor_c_torch,
                   a)

    test_operation("rmul",
                   lambda x: tensor_b_fudan * x,
                   lambda x: tensor_b_torch * x,
                   a)
    
    test_operation("broadcast_rmul",
                   lambda x: tensor_c_fudan * x,
                   lambda x: tensor_c_torch * x,
                   a)
    
    test_operation("truediv",
                   lambda x: x / tensor_b_fudan,
                   lambda x: x / tensor_b_torch,
                   a)

    test_operation("broadcast_truediv",
                   lambda x: x / tensor_c_fudan,
                   lambda x: x / tensor_c_torch,
                   a)
    
    test_operation("rtruediv",
                   lambda x: tensor_b_fudan / x,
                   lambda x: tensor_b_torch / x,
                   a)

    test_operation("broadcast_rtruediv",
                   lambda x: tensor_c_fudan / x,
                   lambda x: tensor_c_torch / x,
                   a)

    test_operation("pow",
                   lambda x: x ** tensor_b_fudan,
                   lambda x: x ** tensor_b_torch,
                   a)
    
    test_operation("broadcast_pow",
                   lambda x: x ** tensor_c_fudan,
                   lambda x: x ** tensor_c_torch,
                   a)
    
    test_operation("rpow",
                   lambda x: tensor_b_fudan ** x,
                   lambda x: tensor_b_torch ** x,
                   a)

    test_operation("broadcast_rpow",
                   lambda x: tensor_c_fudan ** x,
                   lambda x: tensor_c_torch ** x,
                   a)

    test_operation("matmul",
                   lambda x: x @ tensor_b_fudan.T,
                   lambda x: x @ tensor_b_torch.T,
                   a)
    
    test_operation("log",
                lambda x: x.log(),
                lambda x: x.log(),
                a)

    test_operation("exp",
                   lambda x: x.exp(),
                   lambda x: x.exp(),
                   a)

    test_operation("mean",
                lambda x: x.mean(),
                lambda x: x.mean(),
                a)

    test_operation("mean_dim0",
                lambda x: x.mean(axis=0),
                lambda x: x.mean(dim=0),
                a)

    test_operation("max",
                   lambda x: x.max(),
                   lambda x: x.max(),
                   a)

    test_operation("max_dim0",
                   lambda x: x.max(axis=0),
                   lambda x: x.max(dim=0)[0],
                   a)
    
    test_operation("sum",
                lambda x: x.sum(axis=0),
                lambda x: x.sum(dim=0),
                a)

    test_operation("reshape",
                   lambda x: x.reshape((1, 9)),
                   lambda x: x.reshape((1, 9)),
                   a)

    test_operation("flatten",
                   lambda x: x.flatten(),
                   lambda x: x.flatten(),
                   a)

    test_operation("squeeze",
                   lambda x: x.reshape((1, 3, 3)).squeeze(),
                   lambda x: x.reshape((1, 3, 3)).squeeze(),
                   a)

    test_operation("unsqueeze",
                   lambda x: x.unsqueeze(0),
                   lambda x: x.unsqueeze(0),
                   a)

    test_operation("transpose",
                   lambda x: x.transpose((1, 0)),
                   lambda x: x.transpose(1, 0),
                   a)

    test_operation("permute",
                   lambda x: x.reshape((1, 3, 3)).permute((2, 1, 0)),
                   lambda x: x.reshape((1, 3, 3)).permute(2, 1, 0),
                   a)

    # Complex operations

    test_operation("pow_sum_log",
                   lambda x: ((x * x).sum()).log(),
                   lambda x: ((x * x).sum()).log(),
                   a)
    
    test_operation("softmax",
                   lambda x: (x - x.max(axis=-1, keepdims=True)).exp() /
                   (x - x.max(axis=-1, keepdims=True)).exp().sum(axis=-1, keepdims=True),
                   lambda x: (x - x.max(axis=-1, keepdims=True)[0]).exp() /
                   (x - x.max(axis=-1, keepdims=True)[0]).exp().sum(axis=-1, keepdims=True),
                   a)

    test_operation("log_softmax",
                   lambda x: ((x - x.max(axis=-1, keepdims=True)).exp() /
                              (x - x.max(axis=-1, keepdims=True)).exp().sum(axis=-1, keepdims=True)).log(),
                   lambda x: ((x - x.max(axis=-1, keepdims=True)[0]).exp() /
                              (x - x.max(axis=-1, keepdims=True)[0]).exp().sum(axis=-1, keepdims=True)).log(),
                   a)
                
    def fudan_softmax_cross_entropy_loss(x):
        log_probs = ((x - x.max(axis=-1, keepdims=True)).exp() /
                              (x - x.max(axis=-1, keepdims=True)).exp().sum(axis=-1, keepdims=True)).log()
        label = Tensor([0, 1, 2], requires_grad=False)
        return -log_probs[[0, 1, 2], label].log().mean()
    
    def torch_softmax_cross_entropy_loss(x):
        log_probs = ((x - x.max(axis=-1, keepdims=True)[0]).exp() /
                              (x - x.max(axis=-1, keepdims=True)[0]).exp().sum(axis=-1, keepdims=True)).log()
        label = torch.tensor([0, 1, 2], dtype=torch.long)
        return -log_probs[[0, 1, 2], label].log().mean()

    test_operation("softmax_cross_entropy_loss",
                   fudan_softmax_cross_entropy_loss,
                   torch_softmax_cross_entropy_loss,
                   a)




    
