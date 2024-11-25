# Adam Optimizer

The `Adam` class implements the Adam optimization algorithm, which combines the benefits of momentum and RMSprop.

## Features
- Adaptive learning rate for each parameter.
- Handles sparse gradients and is effective for deep networks.

## Public Methods

| Function Name          | Method Signature (Python)                        | Description                                         |
|-------------------------|-------------------------------------------------|-----------------------------------------------------|
| `Adam`                | `Adam(parameters: List[Tensor], learning_rate: float, beta1: float = 0.9, beta2: float = 0.999, epsilon: float = 1e-8)` | Initializes the optimizer with parameters and hyperparameters. |
| `step`                | `adam.step()`                                     | Updates parameters using gradients.                |
| `zero_grad`           | `adam.zero_grad()`                                | Resets gradients for all parameters.               |

## Example Usage in Python
```python
import custom_torch as ct

# Create Adam optimizer
optimizer = ct.Adam(parameters=model.parameters, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8)

# Perform an optimization step
optimizer.step()

# Zero out gradients after the step
optimizer.zero_grad()