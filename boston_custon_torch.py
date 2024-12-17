import numpy as np
from custom_torch import Tensor, Linear, Adam, MSELoss, SGD

# Load the dataset
file_path = "./datasets/HousingData.csv"
data = np.genfromtxt(file_path, delimiter=",", skip_header=1, missing_values="NA", filling_values=0)

# Separate features (X) and target (y)
X = data[:, :-1]  # All columns except the last are features
y = data[:, -1]   # Last column is the target

# Normalize features and target
X_mean, X_std = X.mean(axis=0), X.std(axis=0)
X = (X - X_mean) / X_std

y_mean, y_std = y.mean(), y.std()
y = (y - y_mean) / y_std

# Convert to Tensor objects
X_tensor = Tensor(X.tolist(), "cuda", False)
y_tensor = Tensor(y.reshape(-1, 1).tolist(), "cuda", False)

# Split data into training and testing sets (80% train, 20% test)
n_samples = len(X)
n_train = int(0.8 * n_samples)

X_train = Tensor(X[:n_train].tolist(), "cuda", False)
X_test = Tensor(X[n_train:].tolist(), "cuda", False)
y_train = Tensor(y[:n_train].reshape(-1, 1).tolist(), "cuda", False)
y_test = Tensor(y[n_train:].reshape(-1, 1).tolist(), "cuda", False)

# Define the model
input_dim = X.shape[1]
model = Linear(input_dim, 1, "cuda", True)

# Define loss and optimizer
criterion = MSELoss()
optimizer = SGD([model.weights, model.bias], 0.01)

# Training loop
num_epochs = 500
for epoch in range(num_epochs):
    predictions = model.forward(X_train)
    loss = criterion(predictions, y_train)

    # Backward pass
    loss.backward()

    # Update weights
    optimizer.step()
    optimizer.zero_grad()

    # Print loss every 50 epochs
    if (epoch + 1) % 50 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.get_data()[0]:.4f}")

# Evaluate the model on the test set
predictions = model.forward(X_test)
test_loss = criterion(predictions, y_test)
print(f"\nTest Loss: {test_loss.get_data()[0]:.4f}")