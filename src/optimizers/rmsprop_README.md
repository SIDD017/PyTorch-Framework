# RMSprop Optimizer

The `RMSprop` class implements the RMSprop optimization algorithm, which adapts the learning rate for each parameter using a moving average of squared gradients.

## Features
- Adaptive learning rate for better convergence.
- Effective for deep networks and sparse gradients.

## Public Methods

| Function Name          | Method Signature (Python)                        | Description                                         |
|-------------------------|-------------------------------------------------|-----------------------------------------------------|
| `RMSprop`              | `RMSprop(parameters: List[Tensor], learning_rate: float, alpha: float = 0.99, epsilon: float = 1e-8)` | Initializes the optimizer with parameters and hyperparameters. |
| `step`                 | `rmsprop.step()`                                | Updates parameters using gradients.                |
| `zero_grad`            | `rmsprop.zero_grad()`                           | Resets gradients for all parameters.               |

## Example Usage in Python
```python
import custom_torch as ct

# Create RMSprop optimizer
optimizer = ct.RMSprop(parameters=model.parameters, learning_rate=0.001, alpha=0.99, epsilon=1e-8)

# Perform an optimization step
optimizer.step()

# Zero out gradients after the step
optimizer.zero_grad()
