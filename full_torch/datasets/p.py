import numpy as np

data = np.load("linear_regression_data.npz")
X_train, y_train = data['X_train'], data['y_train']
X_test, y_test = data['X_test'], data['y_test']

print(X_train)
print(y_train)