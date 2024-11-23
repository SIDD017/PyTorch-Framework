import numpy as np
import os
import struct

def load_mnist_images(file_path):
    """Load MNIST images from the IDX file."""
    with open(file_path, 'rb') as f:
        # Read header information
        magic, num_images, rows, cols = struct.unpack('>IIII', f.read(16))
        assert magic == 2051, "Invalid magic number for image file!"
        # Read the image data
        images = np.frombuffer(f.read(), dtype=np.uint8)
        images = images.reshape(num_images, rows * cols).astype(np.float32)
        return images / 255.0  # Normalize to [0, 1]

def load_mnist_labels(file_path):
    """Load MNIST labels from the IDX file."""
    with open(file_path, 'rb') as f:
        # Read header information
        magic, num_labels = struct.unpack('>II', f.read(8))
        assert magic == 2049, "Invalid magic number for label file!"
        # Read the label data
        labels = np.frombuffer(f.read(), dtype=np.uint8)
        return labels

# Paths to your MNIST files
DATA_PATH = "../data/MNIST/raw/"
TRAIN_IMAGES_PATH = os.path.join(DATA_PATH, "train-images-idx3-ubyte")
TRAIN_LABELS_PATH = os.path.join(DATA_PATH, "train-labels-idx1-ubyte")
TEST_IMAGES_PATH = os.path.join(DATA_PATH, "t10k-images-idx3-ubyte")
TEST_LABELS_PATH = os.path.join(DATA_PATH, "t10k-labels-idx1-ubyte")

# Load the dataset
x_train = load_mnist_images(TRAIN_IMAGES_PATH)
y_train = load_mnist_labels(TRAIN_LABELS_PATH)
x_test = load_mnist_images(TEST_IMAGES_PATH)
y_test = load_mnist_labels(TEST_LABELS_PATH)

# Convert labels to one-hot encoding for TensorFlow
y_train = np.eye(10)[y_train]  # One-hot encode
y_test = np.eye(10)[y_test]

print(f"Train images shape: {x_train.shape}, Train labels shape: {y_train.shape}")
print(f"Test images shape: {x_test.shape}, Test labels shape: {y_test.shape}")
