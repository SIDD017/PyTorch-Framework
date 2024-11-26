import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Load the dataset from the CSV file
file_path = "../datasets/HousingData.csv"

# Use numpy.genfromtxt to handle missing values
data = np.genfromtxt(file_path, delimiter=",", skip_header=1, missing_values="NA", filling_values=0)

# Separate features (X) and target (y)
X = data[:, :-1]  # All columns except the last are features
y = data[:, -1]   # Last column is the target

# Normalize features (manual normalization)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X = (X - X_mean) / X_std

# Normalize target
y_mean = y.mean()
y_std = y.std()
y = (y - y_mean) / y_std

# Convert to PyTorch tensors
X = torch.tensor(X, dtype=torch.float32)
y = torch.tensor(y, dtype=torch.float32).view(-1, 1)  # Make target a column vector

# Split data into training and testing sets (80% train, 20% test)
n_samples = X.shape[0]
n_train = int(0.8 * n_samples)

X_train, X_test = X[:n_train], X[n_train:]
y_train, y_test = y[:n_train], y[n_train:]

# Define the linear regression model
class LinearRegressionModel(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, 1)  # Single linear layer for regression

    def forward(self, x):
        return self.linear(x)

# Initialize the model, loss function, and optimizer
input_dim = X.shape[1]
model = LinearRegressionModel(input_dim)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.Adam(model.parameters(), lr=0.01)  # Adam optimizer

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    model.train()
    
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)
    
    # Backward pass and optimization
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    
    # Print loss every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f'Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}')

# Evaluate the model on the test set
model.eval()
with torch.no_grad():
    test_predictions = model(X_test)
    test_loss = criterion(test_predictions, y_test)
    print(f'\nTest Loss: {test_loss.item():.4f}')
