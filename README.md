# Custom PyTorch Like Framework

This project is a lightweight neural network framework focused on linear layers, implemented in **C++** with support for **CUDA** to enable GPU-based computations. It includes core components such as tensors, linear layers, optimizers, and loss functions. The framework uses a **computational graph** for efficient gradient computation and backpropagation. With Python bindings via **Pybind11**, users can integrate it seamlessly into Python workflows.

---

## Features
- **Tensor Operations**: Efficient operations for tensors on both CPU and GPU.
- **Fully Connected Layers**: Implements linear layers with weights and biases.
- **Optimizers**: Supports SGD, Adam, and RMSprop optimizers.
- **Loss Functions**: Includes MSE Loss and Cross Entropy Loss for training models.
- **Computational Graph**: Tracks operations dynamically for automatic gradient computation.
- **Python Bindings**: Seamless integration with Python via Pybind11.

---

## Documentation
For detailed information on individual components, refer to their documentation:
- [Tensor](src/tensor/tensor_README.md)
- [Linear Layer](src/layers/linear_README.md)
- [Cross Entropy Loss](src/loss/cross_entropy_loss_README.md)
- [MSE Loss](src/loss/mse_loss_README.md)
- [SGD Optimizer](src/optimizers/sgd_README.md)
- [Adam Optimizer](src/optimizers/adam_README.md)
- [RMSprop Optimizer](src/optimizers/rmsprop_README.md)

---

## Installation

### Prerequisites
- C++11 or higher
- Python 3.7+
- [Pybind11](https://github.com/pybind/pybind11)
- CUDA (for GPU support)

### Build Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/SIDD017/PyTorch-Framework.git
   cd yourproject
2. Build the project using CMake:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
3. Use the compiled Python module:
   ```python
   import custom_torch as ct

## Example Workflow
Here’s a basic example of defining a model, performing a forward pass, and optimizing it:
```python
import custom_torch
import numpy as np
from tqdm import tqdm
import os

# Utility functions for loading and preparing MNIST data
def tensor_to_numpy(tensor):
    if len(tensor.get_dims()) == 1:
        data = tensor.get_data_1d()
    elif len(tensor.get_dims()) == 2:
        data = tensor.get_data_2d()
    elif len(tensor.get_dims()) == 3:
        data = tensor.get_data_3d()
    return np.array(data)

def read_mnist_images(filename):
    with open(filename, 'rb') as f:
        f.read(16)  # Skip header
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8).astype(np.float32)
        return data.reshape(-1, 28 * 28) / 255.0

def read_mnist_labels(filename):
    with open(filename, 'rb') as f:
        f.read(8)  # Skip header
        buffer = f.read()
        return np.frombuffer(buffer, dtype=np.uint8)

def to_one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels]

def load_mnist_local(data_dir):
    train_images = read_mnist_images(os.path.join(data_dir, 'train-images-idx3-ubyte'))
    train_labels = read_mnist_labels(os.path.join(data_dir, 'train-labels-idx1-ubyte'))
    test_images = read_mnist_images(os.path.join(data_dir, 't10k-images-idx3-ubyte'))
    test_labels = read_mnist_labels(os.path.join(data_dir, 't10k-labels-idx1-ubyte'))
    return train_images, to_one_hot(train_labels), test_images, to_one_hot(test_labels)

# Define the MNIST Network
class MNISTNetwork:
    def __init__(self, device="cuda"):
        self.fc1 = custom_torch.Linear(784, 128, device, True)
        self.fc2 = custom_torch.Linear(128, 64, device, True)
        self.fc3 = custom_torch.Linear(64, 10, device, True)
        self.parameters = [
            self.fc1.weights, self.fc1.bias,
            self.fc2.weights, self.fc2.bias,
            self.fc3.weights, self.fc3.bias
        ]
    
    def forward(self, x):
        x = self.fc1.forward(x).relu()
        x = self.fc2.forward(x).relu()
        x = self.fc3.forward(x).exp()
        sums = x.sum(1).reciprocal()
        return sums * x

# Training function
def train(model, X_train, y_train, batch_size=32, epochs=10, learning_rate=0.001):
    optimizer = custom_torch.RMSprop(model.parameters, learning_rate)
    criterion = custom_torch.MSELoss()
    n_samples = X_train.shape[0]
    n_batches = n_samples // batch_size

    for epoch in range(epochs):
        total_loss, correct = 0, 0
        indices = np.random.permutation(n_samples)
        X_train, y_train = X_train[indices], y_train[indices]

        for i in tqdm(range(n_batches), desc=f'Epoch {epoch+1}/{epochs}'):
            start_idx, end_idx = i * batch_size, (i + 1) * batch_size
            batch_X = custom_torch.Tensor(X_train[start_idx:end_idx], model.device, False)
            batch_y = custom_torch.Tensor(y_train[start_idx:end_idx], model.device, False)

            output = model.forward(batch_X)
            loss = criterion(output, batch_y)
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            predictions = np.argmax(tensor_to_numpy(output), axis=1)
            targets = np.argmax(y_train[start_idx:end_idx], axis=1)
            correct += np.sum(predictions == targets)
            total_loss += tensor_to_numpy(loss)[0]

        print(f"Epoch {epoch+1}: Loss = {total_loss / n_batches:.4f}, Accuracy = {correct / n_samples:.4f}")

# Evaluation function
def evaluate(model, X_test, y_test, batch_size=32):
    n_samples = X_test.shape[0]
    n_batches = n_samples // batch_size
    correct = 0

    for i in range(n_batches):
        start_idx, end_idx = i * batch_size, (i + 1) * batch_size
        batch_X = custom_torch.Tensor(X_test[start_idx:end_idx], model.device, False)
        output = model.forward(batch_X)

        predictions = np.argmax(tensor_to_numpy(output), axis=1)
        targets = np.argmax(y_test[start_idx:end_idx], axis=1)
        correct += np.sum(predictions == targets)

    accuracy = correct / n_samples
    print(f"Test Accuracy: {accuracy:.4f}")
    return accuracy

# Main function
def main():
    data_dir = "datasets/mnist/MNIST/raw/"
    X_train, y_train, X_test, y_test = load_mnist_local(data_dir)

    model = MNISTNetwork(device="cuda")
    train(model, X_train, y_train, batch_size=32, epochs=10, learning_rate=0.001)
    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()


