import tensorflow as tf
from tensorflow.keras import layers, models
# Load preprocessed MNIST data
from data_preprocess_tf import x_test, y_test

# Hyperparameters
WEIGHTS_PATH = "./tf_weights.weights.h5"

# Define the model
model = models.Sequential([
    layers.Input(shape=(784,)),
    layers.Dense(256, activation="relu"),
    layers.Dense(128, activation="relu"),
    layers.Dense(64, activation="relu"),
    layers.Dense(10, activation="softmax")
])

# Compile the model before evaluation
model.compile(optimizer=tf.keras.optimizers.Adam(),
              loss="categorical_crossentropy",
              metrics=["accuracy"])

# Load weights
model.load_weights(WEIGHTS_PATH)
print(f"Weights loaded from {WEIGHTS_PATH}")

# Evaluate the model
loss, accuracy = model.evaluate(x_test, y_test)
print(f"Test Accuracy: {accuracy * 100:.2f}%")
