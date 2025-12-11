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

def steepest_descent_ridge(X, y, lam, alpha=1e-7, max_iter=2000, tol=1e-8):
    import time
    beta = np.zeros(X.shape[1])
    objectives = []
    start_time = time.time()
    
    for iteration in range(max_iter):
        grad = ridge_gradient(beta, X, y, lam)
        objectives.append(ridge_objective(beta, X, y, lam))
        if np.linalg.norm(grad) < tol:
            break
        beta = beta - alpha * grad
    
    elapsed_time = time.time() - start_time
    return beta, np.array(objectives), elapsed_time, iteration + 1

# ============= 2. Conjugate Gradient =============
# Solve (X^T X + lam I) beta = X^T y

def conjugate_gradient_ridge(X, y, lam, max_iter=1000, tol=1e-6):
    import time
    n_features = X.shape[1]
    A = X.T @ X + lam * np.eye(n_features)
    b = X.T @ y
    
    start_time = time.time()
    beta = np.zeros(n_features)
    r = b - A @ beta          # residual
    d = r.copy()              # direction
    delta_new = r @ r
    objectives = []

    for iteration in range(max_iter):
        objectives.append(ridge_objective(beta, X, y, lam))
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
        d = r + (delta_new / delta_old) * d

    elapsed_time = time.time() - start_time
    return beta, np.array(objectives), elapsed_time, iteration + 1

# ============= 3. Newton's Method =============

def newton_ridge(X, y, lam, max_iter=50, tol=1e-10):
    import time
    n_features = X.shape[1]
    H = 2 * (X.T @ X + lam * np.eye(n_features))  # Hessian
    start_time = time.time()
    H_inv = np.linalg.inv(H)

    beta = np.zeros(n_features)
    objectives = []
    
    for iteration in range(max_iter):
        grad = ridge_gradient(beta, X, y, lam)
        objectives.append(ridge_objective(beta, X, y, lam))
        if np.linalg.norm(grad) < tol:
            break
        beta = beta - H_inv @ grad
    
    elapsed_time = time.time() - start_time
    return beta, np.array(objectives), elapsed_time, iteration + 1

# ============= 4. Closed-form solution =============

def closed_form_ridge(X, y, lam):
    import time
    n_features = X.shape[1]
    start_time = time.time()
    A = X.T @ X + lam * np.eye(n_features)
    b = X.T @ y
    beta = np.linalg.solve(A, b)
    elapsed_time = time.time() - start_time
    
    # Single objective value (no iterations)
    objectives = np.array([ridge_objective(beta, X, y, lam)])
    return beta, objectives, elapsed_time, 1

# ============= run all methods and compare =============

beta_sd, obj_sd, time_sd, iter_sd = steepest_descent_ridge(X, y, lam)
beta_cg, obj_cg, time_cg, iter_cg = conjugate_gradient_ridge(X, y, lam)
beta_newton, obj_newton, time_newton, iter_newton = newton_ridge(X, y, lam)
beta_cf, obj_cf, time_cf, iter_cf = closed_form_ridge(X, y, lam)

print("="*70)
print("RIDGE REGRESSION OPTIMIZATION METHODS COMPARISON")
print("="*70)
print("\n1. OBJECTIVE VALUES (Lower is Better)")
print("-" * 70)
print(f"Steepest Descent: {ridge_objective(beta_sd, X, y, lam):.6f}")
print(f"Conjugate Grad..: {ridge_objective(beta_cg, X, y, lam):.6f}")
print(f"Newton Method...: {ridge_objective(beta_newton, X, y, lam):.6f}")
print(f"Closed-form.....: {ridge_objective(beta_cf, X, y, lam):.6f}")

print("\n2. COMPUTATIONAL COST & EFFICIENCY")
print("-" * 70)
print(f"{'Method':<20} {'Time (ms)':<15} {'Iterations':<15} {'Time/Iter (μs)':<15}")
print("-" * 70)
print(f"{'Steepest Descent':<20} {time_sd*1000:<15.4f} {iter_sd:<15} {(time_sd*1e6/iter_sd):<15.2f}")
print(f"{'Conjugate Grad..':<20} {time_cg*1000:<15.4f} {iter_cg:<15} {(time_cg*1e6/iter_cg):<15.2f}")
print(f"{'Newton Method':<20} {time_newton*1000:<15.4f} {iter_newton:<15} {(time_newton*1e6/iter_newton):<15.2f}")
print(f"{'Closed-form':<20} {time_cf*1000:<15.4f} {iter_cf:<15} {(time_cf*1e6/iter_cf):<15.2f}")

print("\n3. CONVERGENCE RATE SUMMARY")
print("-" * 70)
print(f"{'Method':<20} {'Iterations':<15} {'Convergence Type':<35}")
print("-" * 70)
print(f"{'Steepest Descent':<20} {iter_sd:<15} {'Slow (linear)':<35}")
print(f"{'Conjugate Grad..':<20} {iter_cg:<15} {'Fast (super-linear)':<35}")
print(f"{'Newton Method':<20} {iter_newton:<15} {'Very Fast (quadratic)':<35}")
print(f"{'Closed-form':<20} {iter_cf:<15} {'Direct (no iterations)':<35}")

print("\n4. OVERALL EFFICIENCY RANKING")
print("-" * 70)
efficiency_scores = [
    ("Closed-form", time_cf),
    ("Newton Method", time_newton),
    ("Conjugate Grad.", time_cg),
    ("Steepest Descent", time_sd)
]
efficiency_scores.sort(key=lambda x: x[1])
for rank, (method, exec_time) in enumerate(efficiency_scores, 1):
    print(f"{rank}. {method:<20} ({exec_time*1000:.4f} ms)")

# ============= VISUALIZATIONS =============

import matplotlib.pyplot as plt

# ========== CONVERGENCE PLOTS (STANDALONE) ==========

# Plot 1: Loss vs Iteration (Linear Scale) - STANDALONE
fig1, ax1 = plt.subplots(figsize=(12, 7))
ax1.plot(obj_sd, 'o-', label='Steepest Descent', linewidth=3, markersize=5, alpha=0.8, color='#FF6B6B')
ax1.plot(obj_cg, 's-', label='Conjugate Gradient', linewidth=3, markersize=5, alpha=0.8, color='#4ECDC4')
ax1.plot(obj_newton, '^-', label="Newton's Method", linewidth=3, markersize=5, alpha=0.8, color='#45B7D1')
ax1.axhline(y=obj_cf[0], color='#96CEB4', linestyle='--', label='Closed-form (Optimal)', linewidth=3)
ax1.set_xlabel('Iteration', fontsize=13, fontweight='bold')
ax1.set_ylabel('Loss (Objective Value)', fontsize=13, fontweight='bold')
ax1.set_title('Convergence Rate: Loss vs Iteration (Linear Scale)', fontsize=14, fontweight='bold', pad=20)
ax1.legend(fontsize=11, loc='upper right', framealpha=0.95)
ax1.grid(True, alpha=0.4, linestyle='--', linewidth=0.8)
ax1.set_facecolor('#f9f9f9')
plt.tight_layout()
plt.savefig('convergence_linear.png', dpi=300, bbox_inches='tight')
print("\n✓ Standalone Convergence Plot (Linear) saved as 'convergence_linear.png'")
plt.show()

# Plot 2: Log Scale Convergence - STANDALONE
fig2, ax2 = plt.subplots(figsize=(12, 7))
optimal_loss = obj_cf[0]

# Filter out non-positive values for log scale
obj_sd_filtered = np.maximum(obj_sd - optimal_loss, 1e-10)
obj_cg_filtered = np.maximum(obj_cg - optimal_loss, 1e-10)
obj_newton_filtered = np.maximum(obj_newton - optimal_loss, 1e-10)

ax2.semilogy(obj_sd_filtered, 'o-', label='Steepest Descent', linewidth=3, markersize=5, alpha=0.8, color='#FF6B6B')
ax2.semilogy(obj_cg_filtered, 's-', label='Conjugate Gradient', linewidth=3, markersize=5, alpha=0.8, color='#4ECDC4')
ax2.semilogy(obj_newton_filtered, '^-', label="Newton's Method", linewidth=3, markersize=5, alpha=0.8, color='#45B7D1')
ax2.set_xlabel('Iteration', fontsize=13, fontweight='bold')
ax2.set_ylabel('Distance from Optimal (Log Scale)', fontsize=13, fontweight='bold')
ax2.set_title('Convergence Rate: Error Reduction (Log Scale)', fontsize=14, fontweight='bold', pad=20)
ax2.legend(fontsize=11, loc='upper right', framealpha=0.95)
ax2.grid(True, alpha=0.4, linestyle='--', linewidth=0.8, which='both')
ax2.set_facecolor('#f9f9f9')
plt.tight_layout()
plt.savefig('convergence_log.png', dpi=300, bbox_inches='tight')
print("✓ Standalone Convergence Plot (Log Scale) saved as 'convergence_log.png'")
plt.show()

# ========== COMPUTATIONAL COST & EFFICIENCY PLOTS ==========

fig3, axes = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle('Computational Cost & Efficiency Analysis', fontsize=14, fontweight='bold', y=1.02)

# Plot 3: Computational Cost Comparison
ax3 = axes[0]
methods = ['Steepest\nDescent', 'Conjugate\nGradient', "Newton's\nMethod", 'Closed-form']
times = [time_sd*1000, time_cg*1000, time_newton*1000, time_cf*1000]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']
bars = ax3.bar(methods, times, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
ax3.set_title('Computational Cost Comparison', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
ax3.set_facecolor('#f9f9f9')
for bar, time_val in zip(bars, times):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{time_val:.4f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Plot 4: Iterations per Second Efficiency
ax4 = axes[1]
iterations = [iter_sd, iter_cg, iter_newton, iter_cf]
efficiency = [iter_val / exec_time if exec_time > 0 else 0 for iter_val, exec_time in zip(iterations, [time_sd, time_cg, time_newton, time_cf])]
methods_short = ['SD', 'CG', 'NM', 'CF']
bars2 = ax4.barh(methods_short, efficiency, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Iterations per Second', fontsize=12, fontweight='bold')
ax4.set_title('Iteration Efficiency (Higher = Better)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x', linestyle='--')
ax4.set_facecolor('#f9f9f9')
for bar, eff in zip(bars2, efficiency):
    width = bar.get_width()
    ax4.text(width, bar.get_y() + bar.get_height()/2.,
            f'{eff:.0f}/s', ha='left', va='center', fontsize=10, fontweight='bold', 
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.8, edgecolor='gray'))

plt.tight_layout()
plt.savefig('computational_efficiency.png', dpi=300, bbox_inches='tight')
print("✓ Computational Cost & Efficiency Plot saved as 'computational_efficiency.png'")
plt.show()

print("="*70)
print("✓ All visualizations generated successfully!")
print("  - convergence_linear.png (Standalone)")
print("  - convergence_log.png (Standalone)")
print("  - computational_efficiency.png")
print("="*70)
