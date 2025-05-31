import numpy as np
from ..tensor import Tensor

class Loss:
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        raise NotImplementedError
        
    def __call__(self, pred: Tensor, target: Tensor) -> Tensor:
        return self.forward(pred, target)

class MSELoss(Loss):
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        result = ((pred - target) ** 2).mean()
        
        return result

class CrossEntropyLoss(Loss):
    def _softmax(self, x: Tensor) -> Tensor:
        exp_x = (x - x.max(axis=1, keepdims=True)).exp()
        return exp_x / exp_x.sum(axis=1, keepdims=True)
        
    def forward(self, pred: Tensor, target: Tensor) -> Tensor:
        batch_size = pred.shape[0]
        softmax_out = self._softmax(pred)
        
        # Convert one-hot target to class indices if necessary
        if len(target.shape) > 1:
            target_indices = target.argmax(axis=1, keepdims=False)
        else:
            target_indices = target
            
        # Calculate cross entropy loss
        loss = (-(softmax_out[list(range(batch_size)), target_indices]).log()).mean()
            
        return loss