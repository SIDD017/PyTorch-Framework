import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from tqdm import tqdm
import os
from torchvision import datasets, transforms

def load_mnist_pytorch(data_dir):
    """Load MNIST dataset using PyTorch's built-in datasets."""
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Lambda(lambda x: x.view(-1))
    ])
    
    print("Loading training dataset...")
    train_dataset = datasets.MNIST(root=data_dir, train=True,
                                 download=True, transform=transform)
    X_train = torch.stack([item[0] for item in train_dataset])
    y_train = torch.zeros((len(train_dataset), 10))
    y_train.scatter_(1, torch.tensor(train_dataset.targets).unsqueeze(1), 1)
    
    print("Loading test dataset...")
    test_dataset = datasets.MNIST(root=data_dir, train=False,
                                download=True, transform=transform)
    X_test = torch.stack([item[0] for item in test_dataset])
    y_test = torch.zeros((len(test_dataset), 10))
    y_test.scatter_(1, torch.tensor(test_dataset.targets).unsqueeze(1), 1)
    
    return X_train, y_train, X_test, y_test

class MNISTNetwork(nn.Module):
    def __init__(self, device="cpu"):
        super(MNISTNetwork, self).__init__()
        self.device = device
        # Input layer: 784 (28x28) -> 128
        self.fc1 = nn.Linear(784, 128)
        # Hidden layer: 128 -> 64
        self.fc2 = nn.Linear(128, 64)
        # Output layer: 64 -> 10 (number of classes)
        self.fc3 = nn.Linear(64, 10)
        self.to(device)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        x = self.fc3(x)
        return x

def train(model, X_train, y_train, batch_size=32, epochs=10, learning_rate=0.001):
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    criterion = nn.MSELoss()
    
    n_samples = X_train.shape[0]
    n_batches = n_samples // batch_size

    for epoch in range(epochs):
        total_loss = 0
        correct = 0
        indices = torch.randperm(n_samples)
        X_train = X_train[indices]
        y_train = y_train[indices]
        
        for i in tqdm(range(n_batches), desc=f'Epoch {epoch+1}/{epochs}'):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_X = X_train[start_idx:end_idx].to(model.device)
            batch_y = y_train[start_idx:end_idx].to(model.device)
            
            optimizer.zero_grad()
            output = model(batch_X)
            
            loss = criterion(output, batch_y)
            
            loss.backward()
            optimizer.step()
            
            predictions = torch.argmax(output, dim=1)
            targets = torch.argmax(batch_y, dim=1)
            correct += (predictions == targets).sum().item()
            total_loss += loss.item()
        
        avg_loss = total_loss / n_batches
        accuracy = correct / n_samples
        print(f'Epoch {epoch+1}: Loss = {avg_loss:.4f}, Accuracy = {accuracy:.4f}')

def evaluate(model, X_test, y_test, batch_size=32):
    n_samples = X_test.shape[0]
    n_batches = n_samples // batch_size
    correct = 0
    
    model.eval()
    with torch.no_grad():
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            batch_X = X_test[start_idx:end_idx].to(model.device)
            batch_y = y_test[start_idx:end_idx].to(model.device)
            
            output = model(batch_X)
            
            predictions = torch.argmax(output, dim=1)
            targets = torch.argmax(batch_y, dim=1)
            correct += (predictions == targets).sum().item()
    
    accuracy = correct / n_samples
    print(f'Test Accuracy: {accuracy:.4f}')
    return accuracy

def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    data_dir = "../datasets/mnist"
    
    print("Loading MNIST dataset...")
    X_train, y_train, X_test, y_test = load_mnist_pytorch(data_dir)

    print(X_train)

    print(f"Training data shape: {X_train.shape}")
    print(f"Training labels shape: {y_train.shape}")
    print(f"Test data shape: {X_test.shape}")
    print(f"Test labels shape: {y_test.shape}")
    
    print("Initializing model...")
    model = MNISTNetwork(device=device)
    
    print("Starting training...")
    train(model, X_train, y_train, batch_size=32, epochs=10, learning_rate=0.001)
    
    print("Evaluating model...")
    evaluate(model, X_test, y_test)
    
    torch.save(model.state_dict(), 'mnist_model.pth')

if __name__ == "__main__":
    main()
