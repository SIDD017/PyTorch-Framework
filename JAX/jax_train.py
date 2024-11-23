import jax
import jax.numpy as jnp
from jax import random, grad, jit
import optax
from data_preprocess_jax import x_train, y_train, x_test, y_test
import numpy as np
import os
import time
import csv

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
CSV_PATH = "../results/jax_metrics.csv"

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

def accuracy(params, x, y):
    """Compute accuracy of the model."""
    logits = forward_pass(params, x)
    predictions = jnp.argmax(logits, axis=1)
    labels = jnp.argmax(y, axis=1)
    return jnp.mean(predictions == labels)

@jit
def update_params(params, opt_state, x, y):
    """Update parameters using gradients and optimizer."""
    loss, grads = jax.value_and_grad(cross_entropy_loss)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state)
    new_params = optax.apply_updates(params, updates)
    return new_params, opt_state, loss

# Initialize parameters and optimizer
params = init_params()
optimizer = optax.adam(LEARNING_RATE)
opt_state = optimizer.init(params)

# CSV logging
with open(CSV_PATH, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Epoch", "Time (s)", "Train Loss", "Train Accuracy", "Test Accuracy"])  # CSV header

    # Training loop
    total_training_time = 0
    num_batches = x_train.shape[0] // BATCH_SIZE

    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        epoch_loss = 0.0
        correct_train = 0
        total_train = 0

        for i in range(num_batches):
            batch_x = x_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]
            batch_y = y_train[i * BATCH_SIZE:(i + 1) * BATCH_SIZE]

            # Update parameters
            params, opt_state, batch_loss = update_params(params, opt_state, batch_x, batch_y)
            epoch_loss += batch_loss

            # Training accuracy per batch
            logits = forward_pass(params, batch_x)
            correct_train += jnp.sum(jnp.argmax(logits, axis=1) == jnp.argmax(batch_y, axis=1))
            total_train += batch_x.shape[0]

        # Per-epoch metrics
        train_accuracy = correct_train / total_train
        train_loss = epoch_loss / num_batches
        test_accuracy = accuracy(params, x_test, y_test)
        epoch_time = time.time() - start_time
        total_training_time += epoch_time

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Time: {epoch_time:.2f}s, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Write metrics to CSV
        csv_writer.writerow([epoch + 1, epoch_time, train_loss, train_accuracy, test_accuracy])

# Save weights
os.makedirs(os.path.dirname(WEIGHTS_PATH), exist_ok=True)
np.savez(WEIGHTS_PATH, **{k: np.array(v) for k, v in params.items()})
print(f"Weights saved to {WEIGHTS_PATH}")
print(f"Training metrics saved to {CSV_PATH}")
print(f"Total Training Time: {total_training_time:.2f}s")
