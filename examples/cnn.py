import numpy as np
import sys
import os
# 添加上一级目录到模块搜索路径中
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fudanai.tensor import Tensor
from fudanai.layers.conv import Conv2d
from fudanai.layers.linear import Linear
from fudanai.activations.activation import ReLU
from fudanai.losses.loss import CrossEntropyLoss
from fudanai.optimizers.optimizer import Adam
from fudanai.layers.base import Layer

class CNN(Layer):
    def __init__(self):
        self.conv1 = Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.conv2 = Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = Linear(64 * 7 * 7, 128)
        self.fc2 = Linear(128, 10)
        self.relu = ReLU()
        
    def forward(self, x: Tensor) -> Tensor:
        # First conv block
        x = self.conv1(x)
        x = self.relu(x)
        x = Tensor(x.data[:, :, ::2, ::2])  # Max pooling
        
        # Second conv block
        x = self.conv2(x)
        x = self.relu(x)
        x = Tensor(x.data[:, :, ::2, ::2])  # Max pooling
        
        # Flatten
        batch_size = x.data.shape[0]
        x = Tensor(x.data.reshape(batch_size, -1))
        
        # Fully connected layers
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        
        return x

def load_mnist():
    X_train = np.random.randn(100, 1, 28, 28)
    y_train = np.random.randint(0, 10, size=100)
    return X_train, y_train

def main():
    # Load data
    X_train, y_train = load_mnist()
    
    # Create model
    model = CNN()
    criterion = CrossEntropyLoss()
    
    # Collect all parameters
    params = model.parameters().values()
    optimizer = Adam(params, lr=0.001)
    
    # Training loop
    batch_size = 32
    n_epochs = 5
    
    for epoch in range(n_epochs):
        total_loss = 0
        n_batches = len(X_train) // batch_size
        
        for i in range(n_batches):
            # Get batch
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            X_batch = Tensor(X_train[start_idx:end_idx])
            y_batch = Tensor(y_train[start_idx:end_idx])
            
            # Forward pass
            pred = model.forward(X_batch)
            loss = criterion(pred, y_batch)
            total_loss += loss.data
            
            # Backward pass
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
        avg_loss = total_loss / n_batches
        print(f"Epoch {epoch + 1}/{n_epochs}, Average Loss: {avg_loss:.4f}")

if __name__ == "__main__":
    main() 