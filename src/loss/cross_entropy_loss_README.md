# Cross Entropy Loss

The `CrossEntropyLoss` class computes the loss for classification tasks by comparing predicted probabilities with true labels.

## Features
- Suitable for multi-class classification tasks.
- Computes loss for one-hot encoded or probability distribution targets.

## Public Methods

| Function Name          | Method Signature (Python)                         | Description                                         |
|-------------------------|--------------------------------------------------|-----------------------------------------------------|
| `CrossEntropyLoss`     | `CrossEntropyLoss()`                              | Creates a cross-entropy loss object.               |
| `__call__`             | `loss(predictions: Tensor, targets: Tensor) -> float` | Computes the loss between predictions and targets. |

## Example Usage in Python
```python
import custom_torch as ct

# Initialize Cross Entropy Loss
loss_fn = ct.CrossEntropyLoss()

# Predictions and targets
predictions = ct.Tensor([32, 10], "cuda", False)  # Batch size: 32, Classes: 10
targets = ct.Tensor([32, 10], "cuda", False)      # One-hot encoded targets
predictions.randomize()
targets.randomize()

# Compute the loss
loss = loss_fn(predictions, targets)

# Backpropagate
loss.backward()