import torch
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Hyperparameters
BATCH_SIZE = 64
INPUT_SIZE = 784
NUM_CLASSES = 10
DATA_PATH = "../data"
WEIGHTS_PATH = "./pt_weights.pth"

# Device configuration
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Define the neural network (same as in training)
class LinearNN(torch.nn.Module):
    def __init__(self):
        super(LinearNN, self).__init__()
        self.fc1 = torch.nn.Linear(INPUT_SIZE, 256)
        self.fc2 = torch.nn.Linear(256, 128)
        self.fc3 = torch.nn.Linear(128, 64)
        self.fc4 = torch.nn.Linear(64, NUM_CLASSES)
        self.relu = torch.nn.ReLU()

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        x = self.fc4(x)
        return x

# Data preparation
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
test_dataset = datasets.MNIST(root=DATA_PATH, train=False, transform=transform, download=False)
test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)

# Load the model
model = LinearNN().to(DEVICE)
model.load_state_dict(torch.load(WEIGHTS_PATH))
model.eval()
print(f"Weights loaded from {WEIGHTS_PATH}")

# Testing loop
correct = 0
total = 0
with torch.no_grad():
    for data, targets in test_loader:
        data, targets = data.to(DEVICE), targets.to(DEVICE)
        outputs = model(data)
        _, predicted = torch.max(outputs, 1)
        total += targets.size(0)
        correct += (predicted == targets).sum().item()

print(f"Test Accuracy: {100 * correct / total:.2f}%")
