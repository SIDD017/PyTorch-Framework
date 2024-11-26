import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np

# Generate synthetic data for binary classification
np.random.seed(42)
num_samples = 1000
num_features = 2

# Generate random features
X = np.random.randn(num_samples, num_features).astype(np.float32)
# Generate labels (0 or 1) based on a linear function of X
true_weights = np.array([2.0, -1.0])
true_bias = -0.5
y = (X @ true_weights + true_bias > 0).astype(np.float32)

# Convert to PyTorch tensors
X_tensor = torch.tensor(X)
y_tensor = torch.tensor(y).view(-1, 1)  # Reshaping to make y a column vector

# Define Logistic Regression model (single Linear layer + Sigmoid)
class LogisticRegressionModel(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(LogisticRegressionModel, self).__init__()
        self.linear = nn.Linear(input_dim, output_dim)
    
    def forward(self, x):
        return torch.sigmoid(self.linear(x))

# Instantiate the model
input_dim = X.shape[1]
output_dim = 1
model = LogisticRegressionModel(input_dim, output_dim)

# Define Binary Cross-Entropy loss function
criterion = nn.BCELoss()

# Define optimizer
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Training loop
num_epochs = 100
batch_size = 64

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i in range(0, X.shape[0], batch_size):
        # Get batch data
        batch_X = X_tensor[i:i + batch_size]
        batch_y = y_tensor[i:i + batch_size]

        # Forward pass
        predictions = model(batch_X)
        loss = criterion(predictions, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.item()

    # Print average loss for the epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / (X.shape[0] // batch_size):.4f}")

# Evaluation on test data (optional)
with torch.no_grad():
    predictions = model(X_tensor)
    predictions = (predictions > 0.5).float()
    accuracy = (predictions.flatten() == y_tensor.flatten()).float().mean()
    print(f"Accuracy: {accuracy.item() * 100:.2f}%")
