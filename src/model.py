import numpy as np

def sigmoid(z): return 1.0 / (1.0 + np.exp(-np.clip(z, -35, 35)))
def predict_proba(X, theta): return sigmoid(X @ theta[:-1] + theta[-1])
def train(X, y, theta, epochs, lr):
    p = theta.copy()
    for _ in range(epochs):
        err = predict_proba(X, p) - y
        p[:-1] -= lr * (X.T @ err / len(y) + 0.001 * p[:-1])
        p[-1] -= lr * err.mean()
    return p
