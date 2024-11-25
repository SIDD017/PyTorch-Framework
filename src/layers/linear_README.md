# Linear Layer

The `Linear` class implements a fully connected (dense) layer, essential for neural networks.

## Features
- Weight and bias initialization.
- Forward pass computation with activation functions.

## Public Methods

| Function Name          | Method Signature (Python)                        | Description                                         |
|-------------------------|-------------------------------------------------|-----------------------------------------------------|
| `Linear`               | `Linear(input_size: int, output_size: int, device: str, requires_grad: bool)` | Creates a fully connected layer.                   |
| `forward`              | `linear.forward(input: Tensor) -> Tensor`        | Performs the linear transformation \( y = xW + b \). |

## Example Usage in Python
```python
import custom_torch as ct

# Create a linear layer
linear = ct.Linear(784, 128, "cuda", True)

# Forward pass
input_tensor = ct.Tensor([32, 784], "cuda", False)  # Batch of 32 MNIST images
input_tensor.randomize()

output_tensor = linear.forward(input_tensor)
output_tensor = output_tensor.relu()