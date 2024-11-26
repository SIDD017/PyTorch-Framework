import numpy as np

# Parameters for dataset generation
num_train_samples = 100000  # Number of training samples
num_test_samples = 20000   # Number of testing samples
num_features = 32          # Number of input features
true_weights = np.random.uniform(-5, 5, num_features)  # Random true weights
true_bias = np.random.uniform(-10, 10)                 # Random true bias
noise_std = 0.5                                       # Standard deviation of noise

# Function to generate data
def generate_linear_data(num_samples, num_features, weights, bias, noise_std):
    X = np.random.randn(num_samples, num_features)  # Random input features
    noise = np.random.randn(num_samples) * noise_std  # Gaussian noise
    y = X @ weights + bias + noise                   # Linear relation with noise
    return X, y

# Generate training data
X_train, y_train = generate_linear_data(num_train_samples, num_features, true_weights, true_bias, noise_std)

# Generate testing data
X_test, y_test = generate_linear_data(num_test_samples, num_features, true_weights, true_bias, noise_std)

# Save data to files
np.savez("linear_regression_data.npz", 
         X_train=X_train, y_train=y_train, 
         X_test=X_test, y_test=y_test)

print("Dataset saved to 'linear_regression_data.npz'")
