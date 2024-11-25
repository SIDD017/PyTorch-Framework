# Tensor Class

The `Tensor` class is the foundational data structure, providing capabilities for numerical computations and gradient-based optimization.

## Features
- Supports both CPU and GPU operations.
- Automatic gradient tracking for backpropagation.
- Essential mathematical operations, including matrix multiplication, element-wise operations, and activation functions.

## Public Methods

| Function Name          | Method Signature (Python)                       | Description                                         |
|-------------------------|------------------------------------------------|-----------------------------------------------------|
| `Tensor`               | `Tensor(dims: List[int], device: str, requires_grad: bool)` | Creates a tensor with specified dimensions, device, and gradient tracking. |
| `ones`                 | `Tensor.ones(dims: List[int], device: str, requires_grad: bool)` | Creates a tensor filled with ones.                 |
| `zeros`                | `Tensor.zeros(dims: List[int], device: str, requires_grad: bool)` | Creates a tensor filled with zeros.                |
| `rand`                 | `Tensor.rand(dims: List[int], device: str, requires_grad: bool)` | Creates a tensor with random values (uniform distribution). |
| `randn`                | `Tensor.randn(dims: List[int], device: str, requires_grad: bool)` | Creates a tensor with random values (normal distribution). |
| `transpose`            | `tensor.transpose() -> Tensor`                 | Returns the transposed tensor.                     |
| `matmul`               | `tensor.matmul(other: Tensor) -> Tensor`       | Performs matrix multiplication with another tensor. |
| `add`                  | `tensor.add(other: Tensor) -> Tensor`          | Adds another tensor element-wise.                  |
| `relu`                 | `tensor.relu() -> Tensor`                      | Applies the ReLU activation function.              |
| `exp`                  | `tensor.exp() -> Tensor`                       | Computes the exponential of each element.          |
| `sum`                  | `tensor.sum(dim: Optional[int]) -> Tensor`     | Computes the sum along a specific dimension.        |
| `reciprocal`           | `tensor.reciprocal() -> Tensor`                | Computes the reciprocal of each element.           |
| `zero_grad`            | `tensor.zero_grad()`                           | Resets the gradient to zero.                       |
| `backward`             | `tensor.backward()`                            | Performs backpropagation for tensors requiring gradients. |
| `freeDeviceMemory`     | `tensor.freeDeviceMemory()`                    | Frees allocated memory for GPU-based tensors.      |

## Example Usage in Python
```python
import custom_torch as ct

# Create a tensor on the GPU
a = ct.Tensor([3, 3], "cuda", True)
a.randomize()

# Perform matrix multiplication
b = a.matmul(a.transpose())

# Apply activation function
b = b.relu()
