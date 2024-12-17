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
2. Use the make.sh script or manually build the project using CMake:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
3. Use the compiled Python module:
   ```python
   import custom_torch as ct

---

## Testing

3 python scripts along with their required datasets have been provided to test the build:
- mnist__custom_torch.py: Trains and tests a classification model implemented using Linear layers, AAdam optimizer and CrossEntropyLoss functions on the MNIST datasets.
- regression_custom_torch.py: A simple linear reegression model to for testing a dataset with 100000 training and 20000 testing samples, each with 32 features (dataset randomly generated).
- boston_custom_torch.py: Small linear regression model to test on the Boston housing dataset from Kaggle.

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
2. Use the make.sh script or manually build the project using CMake:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
3. Use the compiled Python module:
   ```python
   import custom_torch as ct