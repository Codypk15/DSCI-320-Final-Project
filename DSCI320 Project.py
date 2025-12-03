# Install dependencies as needed:
# pip install kagglehub[pandas-datasets]
import kagglehub
from kagglehub import KaggleDatasetAdapter
import pandas as pd

# Set the path to the file you'd like to load
file_path = "nba_team_stats_00_to_23.csv"

# Load the latest version
df = kagglehub.load_dataset(
  KaggleDatasetAdapter.PANDAS,
  "mharvnek/nba-team-stats-00-to-18",
  file_path,
  # Provide any additional arguments like 
  # sql_query or pandas_kwargs. See the 
  # documenation for more information:
  # https://github.com/Kaggle/kagglehub/blob/main/README.md#kaggledatasetadapterpandas
)

df.set_index(['Team','season'], inplace=True)


features = ['assists', 'turnovers', 'rebounds', 'field_goal_percentage', 'three_point_percentage']
df = df.dropna(subset=features + ["win_percentage"])
x = df[features].values
y = df["win_percentage"].values

print(df.head())

import numpy as np

X = x  # just rename for clarity
y = y
lam = 10.0  # regularization parameter

# ============= basic helpers =============

def ridge_objective(beta, X, y, lam):
    # L(beta) = ||y - X beta||^2 + lam ||beta||^2
    r = y - X @ beta
    return r @ r + lam * (beta @ beta)

def ridge_gradient(beta, X, y, lam):
    # ∇L = -2 X^T (y - X beta) + 2 lam beta
    r = y - X @ beta
    return -2 * X.T @ r + 2 * lam * beta

# ============= 1. Steepest Descent =============

def steepest_descent_ridge(X, y, lam, alpha=1e-6, max_iter=5000, tol=1e-6):
    beta = np.zeros(X.shape[1])
    for _ in range(max_iter):
        grad = ridge_gradient(beta, X, y, lam)
        if np.linalg.norm(grad) < tol:
            break
        beta = beta - alpha * grad
    return beta

# ============= 2. Conjugate Gradient =============
# Solve (X^T X + lam I) beta = X^T y

def conjugate_gradient_ridge(X, y, lam, max_iter=1000, tol=1e-6):
    n_features = X.shape[1]
    A = X.T @ X + lam * np.eye(n_features)
    b = X.T @ y

    beta = np.zeros(n_features)
    r = b - A @ beta          # residual
    d = r.copy()              # direction
    delta_new = r @ r

    for _ in range(max_iter):
        if np.sqrt(delta_new) < tol:
            break
        Ad = A @ d
        alpha = delta_new / (d @ Ad)
        beta = beta + alpha * d
        r = r - alpha * Ad
        delta_old = delta_new
        delta_new = r @ r
        if delta_new < tol**2:
            break
        beta = beta  # just to see it clearly; no-op
        d = r + (delta_new / delta_old) * d

    return beta

# ============= 3. Newton’s Method =============

def newton_ridge(X, y, lam, max_iter=50, tol=1e-10):
    n_features = X.shape[1]
    H = 2 * (X.T @ X + lam * np.eye(n_features))  # Hessian
    H_inv = np.linalg.inv(H)

    beta = np.zeros(n_features)
    for _ in range(max_iter):
        grad = ridge_gradient(beta, X, y, lam)
        if np.linalg.norm(grad) < tol:
            break
        beta = beta - H_inv @ grad
    return beta

# ============= 4. Closed-form solution =============

def closed_form_ridge(X, y, lam):
    n_features = X.shape[1]
    A = X.T @ X + lam * np.eye(n_features)
    b = X.T @ y
    return np.linalg.solve(A, b)

# ============= run all methods and compare =============

beta_sd = steepest_descent_ridge(X, y, lam)
beta_cg = conjugate_gradient_ridge(X, y, lam)
beta_newton = newton_ridge(X, y, lam)
beta_cf = closed_form_ridge(X, y, lam)

print("Objective values (lower is better):")
print("Steepest Descent:", ridge_objective(beta_sd, X, y, lam))
print("Conjugate Grad.:", ridge_objective(beta_cg, X, y, lam))
print("Newton         :", ridge_objective(beta_newton, X, y, lam))
print("Closed-form    :", ridge_objective(beta_cf, X, y, lam))
