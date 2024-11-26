import numpy as np
import custom_torch as ct

# Load dataset from file
data = np.load("./datasets/linear_regression_data.npz")
X_train = data['X_train']
y_train = data['y_train'].reshape(-1, 1)  # Reshape for compatibility
X_test = data['X_test']
y_test = data['y_test'].reshape(-1, 1)

# Convert numpy arrays to custom Tensor objects
X_test_tensor = ct.Tensor(X_test.tolist(), "cpu", False)
y_test_tensor = ct.Tensor(y_test.tolist(), "cpu", False)

# Initialize the linear model
input_dim = X_train.shape[1]
output_dim = 1
model = ct.Linear(input_dim, output_dim, "cpu", True)

# Loss function and optimizer
criterion = ct.MSELoss()
optimizer = ct.SGD([model.weights, model.bias], 0.01)

# Training loop
num_epochs = 100
batch_size = 1024

for epoch in range(num_epochs):
    epoch_loss = 0.0
    for i in range(0, X_train.shape[0], batch_size):
        # Get batch data
        batch_X = X_train[i:i + batch_size]
        batch_y = y_train[i:i + batch_size]

        batch_X = ct.Tensor(batch_X.tolist(), "cpu", False)
        batch_y = ct.Tensor(batch_y.tolist(), "cpu", False)

        # Forward pass
        predictions = model.forward(batch_X)
        loss = criterion(predictions, batch_y)

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        epoch_loss += loss.get_data()[0]

    # Print average loss for the epoch
    print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss / (X_train.shape[0] // batch_size):.4f}")

# Evaluation on test set
predictions = model.forward(X_test_tensor)
test_loss = criterion(predictions, y_test_tensor).get_data()[0]
print(f"Mean Squared Error on Test Data: {test_loss:.4f}")

# Optional: Visualize the weights and bias
print("Learned weights:", model.weights.get_data())
print("Learned bias:", model.bias.get_data())
