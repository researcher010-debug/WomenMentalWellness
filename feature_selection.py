
import numpy as np
import pandas as pd
from scipy.optimize import minimize

SEED = 42
HSIC_LAMBDA = 0.01
MAX_SAMPLES = 1000

def centered_kernel(x):
    x = np.asarray(x, dtype=np.float64).reshape(-1, 1)
    distances = (x - x.T) ** 2
    positive = distances[distances > 0]
    gamma = 1.0 / (np.median(positive) + 1e-10) if len(positive) else 1.0
    kernel = np.exp(-gamma * distances)
    return kernel - kernel.mean(axis=0, keepdims=True) - kernel.mean(axis=1, keepdims=True) + kernel.mean()

def hsic_lasso(X, y, n_features=None, regularization=HSIC_LAMBDA, max_samples=MAX_SAMPLES):
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y)
    if X.ndim != 2 or X.shape[1] == 0:
        raise ValueError("HSIC-LASSO requires a non-empty feature matrix.")
    if len(X) != len(y):
        raise ValueError("X and y must contain the same number of samples.")
    if not np.isfinite(X).all():
        raise ValueError("Feature matrix contains NaN or infinite values.")
    if len(X) > max_samples:
        rng = np.random.RandomState(SEED)
        indices = np.sort(rng.choice(len(X), max_samples, replace=False))
        X, y = X[indices], y[indices]
    n, p = X.shape
    target_kernel = (y[:, None] == y[None, :]).astype(np.float64)
    target_kernel -= target_kernel.mean(axis=0, keepdims=True)
    target_kernel -= target_kernel.mean(axis=1, keepdims=True)
    target_kernel += target_kernel.mean()
    kernels = np.asarray([centered_kernel(X[:, i]).ravel() for i in range(p)])
    target = target_kernel.ravel()
    Q = kernels @ kernels.T / (n * n)
    b = kernels @ target / (n * n)
    Q += np.eye(p) * 1e-8
    objective = lambda alpha: 0.5 * alpha @ Q @ alpha - b @ alpha + regularization * alpha.sum()
    gradient = lambda alpha: Q @ alpha - b + regularization
    result = minimize(objective, np.zeros(p), jac=gradient, bounds=[(0, None)] * p, method="L-BFGS-B")
    scores = np.maximum(result.x, 0)
    if n_features is None:
        n_features = min(p, max(1, int(np.sqrt(p))))
    n_features = max(1, min(int(n_features), p))
    selected = np.argsort(scores)[::-1][:n_features]
    return selected, scores

def select_features(X, y, n_features=None, regularization=HSIC_LAMBDA):
    if not isinstance(X, pd.DataFrame):
        X = pd.DataFrame(X)
    selected_indices, scores = hsic_lasso(X.values, y, n_features, regularization)
    ranking = pd.DataFrame({"feature": X.columns, "hsic_score": scores})
    ranking = ranking.sort_values("hsic_score", ascending=False).reset_index(drop=True)
    selected_features = X.columns[selected_indices].tolist()
    return X[selected_features].copy(), selected_features, ranking
