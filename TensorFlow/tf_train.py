import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
import time
import csv
# Load preprocessed MNIST data
from data_preprocess_tf import x_train, y_train, x_test, y_test

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
WEIGHTS_PATH = "./tf_weights.weights.h5"
CSV_PATH = "../results/tf_metrics.csv"
INPUT_SIZE = 784  # 28x28
NUM_CLASSES = 10

# Define the model
model = models.Sequential([
    layers.Input(shape=(INPUT_SIZE,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(NUM_CLASSES, activation="softmax")
])
model.compile(optimizer=tf.keras.optimizers.Adam(LEARNING_RATE),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# CSV logging setup
with open(CSV_PATH, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Epoch", "Time (s)", "Train Loss", "Train Accuracy", "Test Accuracy"])  # CSV Header

    total_training_time = 0
    for epoch in range(NUM_EPOCHS):
        # Time tracking
        start_time = time.time()

        # Train the model for one epoch
        history = model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=1, verbose=1)

        # Evaluate on training data for metrics
        train_loss = history.history['loss'][0]
        train_accuracy = history.history['accuracy'][0]

        # Evaluate on test data for metrics
        test_loss, test_accuracy = model.evaluate(x_test, y_test, verbose=0)

        # Time taken for this epoch
        epoch_time = time.time() - start_time
        total_training_time += epoch_time

        # Log the epoch metrics
        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Time: {epoch_time:.2f}s, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Write metrics to CSV
        csv_writer.writerow([epoch + 1, epoch_time, train_loss, train_accuracy, test_accuracy])

# Save the model weights
model.save_weights(WEIGHTS_PATH)
print(f"Weights saved to {WEIGHTS_PATH}")
print(f"Training metrics saved to {CSV_PATH}")
print(f"Total Training Time: {total_training_time:.2f}s")
