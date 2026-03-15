import numpy as np

def _sigmoid(z):
    """Numerically stable sigmoid implementation."""
    return np.where(z >= 0, 1/(1+np.exp(-z)), np.exp(z)/(1+np.exp(z)))

def train_logistic_regression(X, y, lr=0.1, steps=1000):
    """
    Train logistic regression via gradient descent.
    Return (w, b).
    """
    # Write code here
    X = np.array(X)
    X = np.c_[np.ones(X.shape[0]),X]
    y = np.array(y)
    N = X.shape[0]
    w = np.zeros(X.shape[1])
    for _ in range(steps):
        grad = X.T @ (-y + _sigmoid(X @ w))
        w -= lr * 1/N * grad
    
    return w[1:],w[0]