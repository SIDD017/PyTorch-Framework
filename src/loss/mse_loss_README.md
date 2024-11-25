# Mean Squared Error (MSE) Loss

The `MSELoss` class calculates the mean squared error, commonly used for regression tasks or probability predictions.

## Features
- Computes the average squared difference between predictions and targets.

## Public Methods

| Function Name          | Method Signature (Python)                         | Description                                         |
|-------------------------|--------------------------------------------------|-----------------------------------------------------|
| `MSELoss`              | `MSELoss()`                                      | Creates an MSE loss object.                        |
| `__call__`             | `mse_loss(predictions: Tensor, targets: Tensor) -> float` | Computes the loss between predictions and targets. |

## Example Usage in Python
```python
import custom_torch as ct

# Initialize loss function
loss_fn = ct.MSELoss()

# Compute loss
predictions = ct.Tensor([32, 10], "cuda", False)
targets = ct.Tensor([32, 10], "cuda", False)
predictions.randomize()
targets.randomize()

loss = loss_fn(predictions, targets)
loss.backward()