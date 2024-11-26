import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

# Load dataset from file
data = np.load("../datasets/linear_regression_data.npz")
X_train = torch.tensor(data['X_train'], dtype=torch.float32)
y_train = torch.tensor(data['y_train'], dtype=torch.float32).view(-1, 1)  # Reshape for compatibility
X_test = torch.tensor(data['X_test'], dtype=torch.float32)
y_test = torch.tensor(data['y_test'], dtype=torch.float32).view(-1, 1)

# Define the linear regression model
class LinearRegression(nn.Module):
    def __init__(self, input_dim):
        super(LinearRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return self.linear(x)

# Model, loss, and optimizer
input_dim = X_train.shape[1]
model = LinearRegression(input_dim)
criterion = nn.MSELoss()  # Mean Squared Error Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
batch_size = 1024

for epoch in range(num_epochs):
    permutation = torch.randperm(X_train.size(0))
    epoch_loss = 0.0

    for i in range(0, X_train.size(0), batch_size):
        indices = permutation[i:i+batch_size]
        batch_X, batch_y = X_train[indices], y_train[indices]

        # Forward pass
        outputs = model(batch_X)
        loss = criterion(outputs, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Print average loss for the epoch
    print(f"Epoch [{epoch+1}/{num_epochs}], Loss: {epoch_loss / (X_train.size(0) // batch_size):.4f}")

# Model evaluation
with torch.no_grad():
    predictions = model(X_test)
    mse = criterion(predictions, y_test).item()
    print(f"Mean Squared Error on Test Data: {mse:.4f}")

# Optional: Visualize the weights (for insight)
print("Learned weights:", model.linear.weight.data.numpy())
print("Learned bias:", model.linear.bias.data.numpy())
