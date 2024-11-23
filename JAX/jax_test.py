import jax
import jax.numpy as jnp
import numpy as np
from data_preprocess_jax import x_test, y_test

# Hyperparameters
WEIGHTS_PATH = "./jax_weights.npz"
INPUT_SIZE = 784
HIDDEN1 = 256
HIDDEN2 = 128
HIDDEN3 = 64
NUM_CLASSES = 10

# Define the forward pass (same as training)
def forward_pass(params, x):
    """Perform a forward pass through the network."""
    x = jnp.dot(x, params['W1']) + params['b1']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['W2']) + params['b2']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['W3']) + params['b3']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['W4']) + params['b4']
    return x

def accuracy(params, x, y):
    """Compute accuracy of the model."""
    logits = forward_pass(params, x)
    predictions = jnp.argmax(logits, axis=1)
    labels = jnp.argmax(y, axis=1)
    return jnp.mean(predictions == labels)

# Load saved weights
weights = np.load(WEIGHTS_PATH)
params = {k: jnp.array(v) for k, v in weights.items()}
print(f"Weights loaded from {WEIGHTS_PATH}")

# Evaluate the model
test_accuracy = accuracy(params, x_test, y_test)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")
