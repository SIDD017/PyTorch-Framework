# Lightweight Neural Network Framework

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
- [Tensor](src/tensor/README.md)
- [Linear Layer](src/layers/README.md)
- [Cross Entropy Loss](src/losses/README.md)
- [MSE Loss](src/losses/README.md)
- [SGD Optimizer](src/optimizers/README.md)
- [Adam Optimizer](src/optimizers/README.md)
- [RMSprop Optimizer](src/optimizers/README.md)

---

## Installation

### Prerequisites
- C++17 or higher
- Python 3.7+
- [Pybind11](https://github.com/pybind/pybind11)
- CUDA (for GPU support)

### Build Instructions
1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yourproject.git
   cd yourproject
2. Build the project using CMake:
   ```bash
   mkdir build
   cd build
   cmake ..
   make
3. Use the compiled Python module:
   ```python
   import my_model_module as mm

## Example Workflow
Here’s a basic example of defining a model, performing a forward pass, and optimizing it:
```python
import my_model_module as mm

# Define a model
model = mm.Model()
model.add_layer(mm.Linear(4, 8, "cpu"))
model.add_layer(mm.Linear(8, 2, "cpu"))

# Define an optimizer
optimizer = mm.Adam(learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)

# Create input tensor
input_tensor = mm.Tensor([3, 4], "cpu", False)
input_tensor.randomize()

# Perform a forward pass
output = model.forward(input_tensor)

# Example gradients (for demonstration purposes)
gradients = [
    mm.Tensor.ones([4, 8], "cpu", False),
    mm.Tensor.ones([8, 2], "cpu", False)
]

# Get model parameters
parameters = model.get_parameters()
param_names = model.get_parameter_names()

# Perform optimization step
optimizer.step(gradients, parameters, param_names)

print("Model Output:", output.get_data())
