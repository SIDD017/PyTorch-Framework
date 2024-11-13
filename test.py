import apna_torch

# Sample data (inputs and outputs)
X = torch.tensor([[1.0], [2.0], [3.0], [4.0]], dtype=torch.float32)
Y = torch.tensor([[2.0], [4.0], [6.0], [8.0]], dtype=torch.float32)

# Initialize weights and bias
w = torch.tensor(0.0, requires_grad=True)
b = torch.tensor(0.0, requires_grad=True)

# Define learning rate
learning_rate = 0.01
epochs = 100

# Training loop
for epoch in range(epochs):
    # Forward pass: Compute predicted y
    y_pred = X * w + b
    
    # Compute loss (Mean Squared Error)
    loss = ((y_pred - Y) ** 2).mean()
    
    # Backward pass
    loss.backward()
    
    # Update weights and bias
    with torch.no_grad():
        w -= learning_rate * w.grad
        b -= learning_rate * b.grad
    
    # Zero the gradients after updating
    w.grad.zero_()
    b.grad.zero_()

    if (epoch + 1) % 10 == 0:
        print(f'Epoch [{epoch+1}/{epochs}], Loss: {loss.item():.4f}')

print(f'Trained Weight: {w.item()}, Bias: {b.item()}')
