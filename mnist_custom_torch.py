import custom_torch
import numpy as np
from tqdm import tqdm
import gzip
import os

def tensor_to_numpy(tensor):
    if len(tensor.get_dims()) == 1:
        data = tensor.get_data_1d()
    elif len(tensor.get_dims()) == 2:
        data = tensor.get_data_2d()
    elif len(tensor.get_dims()) == 3:
        data = tensor.get_data_3d()
    if isinstance(data, np.ndarray):
        return data
    return np.array(data)

def read_mnist_images(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_images = int.from_bytes(f.read(4), 'big')
        num_rows = int.from_bytes(f.read(4), 'big')
        num_cols = int.from_bytes(f.read(4), 'big')
        
        buffer = f.read()
        data = np.frombuffer(buffer, dtype=np.uint8)
        data = data.reshape(num_images, num_rows * num_cols)
        return data.astype(np.float32) / 255.0

def read_mnist_labels(filename):
    with open(filename, 'rb') as f:
        magic = int.from_bytes(f.read(4), 'big')
        num_items = int.from_bytes(f.read(4), 'big')
        buffer = f.read()
        labels = np.frombuffer(buffer, dtype=np.uint8)
        return labels

def to_one_hot(labels, num_classes=10):
    return np.eye(num_classes)[labels]

def load_mnist_local(data_dir):
    train_images_path = os.path.join(data_dir, 'train-images-idx3-ubyte')
    train_labels_path = os.path.join(data_dir, 'train-labels-idx1-ubyte')
    test_images_path = os.path.join(data_dir, 't10k-images-idx3-ubyte')
    test_labels_path = os.path.join(data_dir, 't10k-labels-idx1-ubyte')
    print("Loading training images...")
    X_train = read_mnist_images(train_images_path)
    print("Loading training labels...")
    y_train = read_mnist_labels(train_labels_path)
    print("Loading test images...")
    X_test = read_mnist_images(test_images_path)
    print("Loading test labels...")
    y_test = read_mnist_labels(test_labels_path)
    y_train_one_hot = to_one_hot(y_train)
    y_test_one_hot = to_one_hot(y_test)
    
    return X_train, y_train_one_hot, X_test, y_test_one_hot

class MNISTNetwork:
    def __init__(self, device="cpu"):
        self.device = device
        # Input layer: 784 (28x28) -> 128
        self.fc1 = custom_torch.Linear(784, 128, device, True)
        # Hidden layer: 128 -> 64
        self.fc2 = custom_torch.Linear(128, 64, device, True)
        # Output layer: 64 -> 10 (number of classes)
        self.fc3 = custom_torch.Linear(64, 10, device, True)
        self.parameters = [
            self.fc1.weights, self.fc1.bias,
            self.fc2.weights, self.fc2.bias,
            self.fc3.weights, self.fc3.bias
        ]

    def forward(self, x):
        x = self.fc1.forward(x)
        x = x.relu()
        x = self.fc2.forward(x)
        x = x.relu()
        x = self.fc3.forward(x)
        return x

def train(model, X_train, y_train, batch_size=32, epochs=10, learning_rate=0.001):
    optimizer = custom_torch.Adam([p for p in model.parameters], learning_rate)
    criterion = custom_torch.MSELoss()
    
    n_samples = X_train.shape[0]
    n_batches = n_samples // batch_size
    
    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        indices = np.random.permutation(n_samples)
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        for i in tqdm(range(n_batches), desc=f'Epoch {epoch+1}/{epochs}'):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_X = X_train[start_idx:end_idx]
            batch_y = y_train[start_idx:end_idx]
            X_tensor = custom_torch.Tensor(batch_X, model.device, False)
            y_tensor = custom_torch.Tensor(batch_y, model.device, False)

            output = model.forward(X_tensor)

            loss = criterion(output, y_tensor)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            predictions = np.argmax(tensor_to_numpy(output).reshape(batch_size, -1), axis=1)
            targets = np.argmax(batch_y, axis=1)
            correct += np.sum(predictions == targets)
            total_loss += tensor_to_numpy(loss)[0]

            X_tensor.freeDeviceMemory()
            y_tensor.freeDeviceMemory()
        
        avg_loss = total_loss / n_batches
        accuracy = correct / n_samples
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}')

def evaluate(model, X_test, y_test, batch_size=32):
    n_samples = X_test.shape[0]
    n_batches = n_samples // batch_size
    correct = 0
    
    for i in range(n_batches):
        start_idx = i * batch_size
        end_idx = start_idx + batch_size

        batch_X = X_test[start_idx:end_idx]
        batch_y = y_test[start_idx:end_idx]

        X_tensor = custom_torch.Tensor(batch_X, model.device, False)

        output = model.forward(X_tensor)

        predictions = np.argmax(tensor_to_numpy(output).reshape(batch_size, -1), axis=1)
        targets = np.argmax(batch_y, axis=1)
        correct += np.sum(predictions == targets)
    
    accuracy = correct / n_samples
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

def main():
    data_dir = "datasets/mnist/MNIST/raw/"

    print("Loading MNIST dataset from local files...")
    X_train, y_train, X_test, y_test = load_mnist_local(data_dir)

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")

    print("Initializing model...")
    model = MNISTNetwork(device="cuda")

    print("Starting training...")
    train(model, X_train, y_train, batch_size=32, epochs=10, learning_rate=0.0001)

    print("Evaluating model...")
    evaluate(model, X_test, y_test)

if __name__ == "__main__":
    main()