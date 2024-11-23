import tensorflow as tf
from tensorflow.keras import layers, models
import numpy as np
# Load preprocessed MNIST data
from data_preprocess_tf import x_train, y_train

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
WEIGHTS_PATH = "./tf_weights.weights.h5"
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

# Train the model
model.fit(x_train, y_train, batch_size=BATCH_SIZE, epochs=NUM_EPOCHS)
model.save_weights(WEIGHTS_PATH)
print(f"Weights saved to {WEIGHTS_PATH}")
