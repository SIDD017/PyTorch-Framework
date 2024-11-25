# SGD Optimizer

The `SGD` class implements the Stochastic Gradient Descent optimization algorithm for updating model parameters.

## Features
- Supports a customizable learning rate.
- Simple and efficient for small-scale models or shallow networks.

## Public Methods

| Function Name          | Method Signature (Python)                        | Description                                         |
|-------------------------|-------------------------------------------------|-----------------------------------------------------|
| `SGD`                 | `SGD(parameters: List[Tensor], learning_rate: float)` | Initializes the optimizer with parameters and learning rate. |
| `step`                | `sgd.step()`                                     | Updates parameters using gradients.                |
| `zero_grad`           | `sgd.zero_grad()`                                | Resets gradients for all parameters.               |

## Example Usage in Python
```python
import custom_torch as ct

# Create SGD optimizer
optimizer = ct.SGD(parameters=model.parameters, learning_rate=0.01)

# Perform an optimization step
optimizer.step()

# Zero out gradients after the step
optimizer.zero_grad()