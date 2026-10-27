"""Six course estimators. Losses are evaluated at the returned weights."""

import numpy as np


def _mse(y, tx, w):
    return np.mean((y - tx @ w) ** 2) / 2


def _sigmoid(z):
    return np.exp(-np.logaddexp(0, -z))


def _logloss(y, tx, w):
    z = tx @ w
    return np.mean(np.logaddexp(0, z) - y * z)


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Minimize half mean squared error using full batch gradient descent."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        w -= gamma * (tx.T @ (tx @ w - y)) / len(y)
    return w, _mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Sample one uniformly random observation per gradient step."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        i = np.random.randint(len(y))
        w -= gamma * tx[i] * (tx[i] @ w - y[i])
    return w, _mse(y, tx, w)


def least_squares(y, tx):
    """Solve normal equations; use their pseudoinverse if singular."""
    gram, rhs = tx.T @ tx, tx.T @ y
    try:
        w = np.linalg.solve(gram, rhs)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(gram) @ rhs
    return w, _mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Minimize MSE + lambda_ * ||w||^2; return unpenalized MSE."""
    if lambda_ == 0:
        return least_squares(y, tx)
    w = np.linalg.solve(
        tx.T @ tx + 2 * len(y) * lambda_ * np.eye(tx.shape[1]), tx.T @ y
    )
    return w, _mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Fit binary labels in {0, 1} with mean logistic loss."""
    return reg_logistic_regression(y, tx, 0, initial_w, max_iters, gamma)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Fit mean logistic loss + lambda_ * ||w||^2 with full batch GD."""
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        grad = tx.T @ (_sigmoid(tx @ w) - y) / len(y)
        w -= gamma * (grad + 2 * lambda_ * w)
    return w, _logloss(y, tx, w)
