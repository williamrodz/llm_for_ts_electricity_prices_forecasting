import numpy as np
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

# Example data
X = np.linspace(0, 10, 100).reshape(-1, 1)
print("X:\n", X)
y = np.sin(X).ravel() + 0.1 * np.random.randn(100)
print("y:\n", y)

# Define the squared exponential (RBF) kernel with a given length_scale
length_scale = 1.0
kernel = RBF(length_scale=length_scale)

# Create the Gaussian Process Regressor with the RBF kernel
gp = GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10)

# Fit the model to the data
gp.fit(X, y)

# Make predictions
X_test = np.linspace(0, 10, 100).reshape(-1, 1)
y_pred, y_std = gp.predict(X_test, return_std=True)

# Plotting the results
import matplotlib.pyplot as plt

plt.figure()
plt.plot(X, y, 'r.', markersize=10, label='Observations')
plt.plot(X_test, y_pred, 'b-', label='Prediction')
plt.fill_between(X_test.ravel(), y_pred - y_std, y_pred + y_std, alpha=0.2, color='k')
plt.xlabel('Input')
plt.ylabel('Output')
plt.legend()
plt.show()