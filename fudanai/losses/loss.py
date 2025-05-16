import numpy as np
from ..tensor import Tensor

class Loss:
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError
        
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.forward(pred, target)

class MSELoss(Loss):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        result = Tensor(np.mean((pred.data - target.data) ** 2), requires_grad=pred.requires_grad)
        
        if result.requires_grad:
            def _backward(grad):
                if pred.requires_grad:
                    dx = 2 * (pred.data - target.data) * grad / pred.data.size
                    pred.backward(dx)
            result._grad_fn = _backward
            result.is_leaf = False
            
        return result

class CrossEntropyLoss(Loss):
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=1, keepdims=True)
        
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        batch_size = pred.data.shape[0]
        softmax_out = self._softmax(pred.data)
        
        # Convert one-hot target to class indices if necessary
        if len(target.data.shape) > 1:
            target_indices = np.argmax(target.data, axis=1)
        else:
            target_indices = target.data
            
        # Calculate cross entropy loss
        loss = -np.log(softmax_out[range(batch_size), target_indices.astype(int)]).mean()
        
        result = Tensor(loss, requires_grad=pred.requires_grad)
        
        if result.requires_grad:
            def _backward(grad):
                if pred.requires_grad:
                    dx = softmax_out.copy()
                    dx[range(batch_size), target_indices.astype(int)] -= 1
                    dx = dx / batch_size * grad
                    pred.backward(dx)
            result._grad_fn = _backward
            result.is_leaf = False
            
        return result 