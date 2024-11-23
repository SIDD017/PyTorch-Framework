import jax
import jax.numpy as jnp
from jax import random, grad, jit
import optax  # Optax is the modern optimization library for JAX
from data_preprocess_jax import x_train, y_train
import numpy as np
import os

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
INPUT_SIZE = 784
HIDDEN1 = 256
HIDDEN2 = 128
HIDDEN3 = 64
NUM_CLASSES = 10
WEIGHTS_PATH = "./jax_weights.npz"

# Initialize random seed
key = random.PRNGKey(0)

# Define the network
def init_params():
    """Initialize parameters for a 4-layer neural network."""
    params = {
        'W1': random.normal(key, (INPUT_SIZE, HIDDEN1)) * 0.01,
        'b1': jnp.zeros(HIDDEN1),
        'W2': random.normal(key, (HIDDEN1, HIDDEN2)) * 0.01,
        'b2': jnp.zeros(HIDDEN2),
        'W3': random.normal(key, (HIDDEN2, HIDDEN3)) * 0.01,
        'b3': jnp.zeros(HIDDEN3),
        'W4': random.normal(key, (HIDDEN3, NUM_CLASSES)) * 0.01,
        'b4': jnp.zeros(NUM_CLASSES),
    }
    return params

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

def cross_entropy_loss(params, x, y):
    """Compute the loss using cross-entropy."""
    logits = forward_pass(params, x)
    log_probs = jax.nn.log_softmax(logits)
    return -jnp.mean(jnp.sum(y * log_probs, axis=1))

@jit
def update_params(params, opt_state, x, y):
    """Update parameters using gradients and optimizer."""
    loss, grads = jax.value_and_grad(cross_entropy_loss)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

# Initialize parameters
params = init_params()

# Initialize optimizer
optimizer = optax.adam(LEARNING_RATE)
opt_state = optimizer.init(params)

# Training loop
num_batches = x_train.shape[0] // BATCH_SIZE

for epoch in range(NUM_EPOCHS):
    epoch_loss = 0.0
    for i in range(num_batches):
        batch_x = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
        batch_y = y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

        # Update parameters
        params, opt_state, batch_loss = update_params(params, opt_state, batch_x, batch_y)
        epoch_loss += batch_loss

    print(f"Epoch {epoch + 1}/{NUM_EPOCHS}, Loss: {epoch_loss / num_batches:.4f}")

# Save weights
os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
np.savez(WEIGHTS_PATH, **{k: np.array(v) for k, v in params.items()})
print(f"Weights saved to {WEIGHTS_PATH}")
