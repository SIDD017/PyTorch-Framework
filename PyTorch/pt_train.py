import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
import time
import csv

# Hyperparameters
BATCH_SIZE = 64
LEARNING_RATE = 0.001
NUM_EPOCHS = 5
INPUT_SIZE = 784  # 28x28
NUM_CLASSES = 10
DATA_PATH = "../data"
WEIGHTS_PATH = "./pt_weights.pth"
CSV_PATH = "../results/pt_metrics.csv"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the neural network
class LinearNN(nn.Module):
    def __init__(self):
        super(LinearNN, self).__init__()
        self.fc1 = nn.Linear(INPUT_SIZE, 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc3 = nn.Linear(128, 64)
        self.fc4 = nn.Linear(64, NUM_CLASSES)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)  # Flatten the input
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)  # Logits for classification
        return x

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
train_dataset = datasets.MNIST(root=DATA_PATH, train=True, transform=transform, download=False)
test_dataset = datasets.MNIST(root=DATA_PATH, train=False, transform=transform, download=False)
train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Function to evaluate accuracy
def evaluate_accuracy(model, loader):
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data, targets in loader:
            data, targets = data.to(DEVICE), targets.to(DEVICE)
            outputs = model(data)
            _, predicted = torch.max(outputs, 1)
            total += targets.size(0)
            correct += (predicted == targets).sum().item()
    return correct / total

# Model, loss, and optimizer
model = LinearNN().to(DEVICE)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)

# CSV logging
with open(CSV_PATH, mode="w", newline="") as csv_file:
    csv_writer = csv.writer(csv_file)
    csv_writer.writerow(["Epoch", "Time (s)", "Train Loss", "Train Accuracy", "Test Accuracy"])  # CSV header

    # Training loop
    total_training_time = 0
    for epoch in range(NUM_EPOCHS):
        start_time = time.time()
        model.train()
        total_loss = 0
        correct_train = 0
        total_train = 0

        for batch_idx, (data, targets) in enumerate(train_loader):
            data, targets = data.to(DEVICE), targets.to(DEVICE)

            optimizer.zero_grad()
            outputs = model(data)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            _, predicted = torch.max(outputs, 1)
            total_train += targets.size(0)
            correct_train += (predicted == targets).sum().item()

        # Per-epoch metrics
        train_loss = total_loss / len(train_loader)
        train_accuracy = correct_train / total_train
        test_accuracy = evaluate_accuracy(model, test_loader)
        epoch_time = time.time() - start_time
        total_training_time += epoch_time

        print(f"Epoch [{epoch + 1}/{NUM_EPOCHS}], Time: {epoch_time:.2f}s, "
              f"Train Loss: {train_loss:.4f}, Train Accuracy: {train_accuracy:.4f}, Test Accuracy: {test_accuracy:.4f}")

        # Save metrics to CSV
        csv_writer.writerow([epoch + 1, epoch_time, train_loss, train_accuracy, test_accuracy])

# Save weights
torch.save(model.state_dict(), WEIGHTS_PATH)
print(f"Weights saved to {WEIGHTS_PATH}")
print(f"Training metrics saved to {CSV_PATH}")
print(f"Total Training Time: {total_training_time:.2f}s")
