import custom_torch as ct
import numpy as np

# Generate synthetic data for binary classification
np.random.seed(42)
num_samples = 1000
num_features = 2

# Generate random features
X = np.random.randn(num_samples, num_features)
# Generate labels (0 or 1) based on a linear function of X
true_weights = np.array([2.0, -1.0])
true_bias = -0.5
y = (X @ true_weights + true_bias > 0).astype(np.float32)

# Convert to Tensor objects
X_tensor = ct.Tensor(X.tolist(), "cpu", False)
# y_tensor = ct.Tensor(y.tolist(), "cpu", False)

# Initialize Logistic Regression model (single Linear layer + Sigmoid)
input_dim = X.shape[1]
output_dim = 1
linear_model = ct.Linear(input_dim, output_dim, "cpu", True)

# Loss function and optimizer
criterion = ct.BinaryCrossEntropyLoss()
optimizer = ct.SGD([linear_model.weights, linear_model.bias], 0.01)

# Training loop
num_epochs = 100
batch_size = 64

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i in range(0, X.shape[0], batch_size):
        # Get batch data
        batch_X = X[i:i + batch_size]
        batch_y = y[i:i + batch_size]

        batch_X = ct.Tensor(batch_X.tolist(), "cpu", False)
        batch_y = ct.Tensor(batch_y.tolist(), "cpu", False)

        # Forward pass
        predictions = linear_model.forward(batch_X)
        loss = criterion(predictions, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.get_data()[0]

    # Print average loss for the epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / (X.shape[0] // batch_size):.4f}")

# Evaluation on test data (optional)
predictions = linear_model.forward(X_tensor)
predictions = (predictions.get_data() > 0.5).astype(int)
accuracy = (predictions.flatten() == y).mean()
print(f"Accuracy: {accuracy * 100:.2f}%")
