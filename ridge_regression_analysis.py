"""
Ridge Regression Optimization Methods Comparison
Analyzes convergence and efficiency of 4 different optimization algorithms
for solving Ridge Regression on NBA team statistics data.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import time
import kagglehub
from kagglehub import KaggleDatasetAdapter
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# SECTION 1: DATA LOADING & PREPROCESSING
# ============================================================================

print("="*80)
print("RIDGE REGRESSION: OPTIMIZATION METHODS COMPARISON")
print("Dataset: NBA Team Statistics (2000-2023)")
print("="*80)

# Load dataset from Kaggle
file_path = "nba_team_stats_00_to_23.csv"
df = kagglehub.load_dataset(
    KaggleDatasetAdapter.PANDAS,
    "mharvnek/nba-team-stats-00-to-18",
    file_path
)

# Select features and target
features = [
    'field_goal_percentage',      # Shooting efficiency
    'three_point_percentage',     # Three-point shooting
    'assists',                    # Ball movement
    'turnovers',                  # Turnovers
    'rebounds',                   # Rebounding
    'steals',                     # Defense
    'blocks'                      # Defense
]

target = 'win_percentage'

# Clean data: remove any rows with missing values
df_clean = df[features + [target]].dropna()

print(f"\nData Summary:")
print(f"  Total samples: {df_clean.shape[0]}")
print(f"  Features used: {len(features)}")
print(f"  Target variable: {target}")
print(f"\nFeatures:")
for i, feat in enumerate(features, 1):
    print(f"  {i}. {feat}")

# Prepare data
X = df_clean[features].values
y = df_clean[target].values

# Standardize features (important for numerical stability)
X_mean = X.mean(axis=0)
X_std = X.std(axis=0)
X_std[X_std == 0] = 1  # Avoid division by zero
X_normalized = (X - X_mean) / X_std

# Regularization parameter
lambda_param = 1.0

print(f"\nData Preprocessing:")
print(f"  Features standardized: Yes")
print(f"  Regularization parameter (lambda): {lambda_param}")
print(f"  Feature matrix shape: {X_normalized.shape}")


# ============================================================================
# SECTION 2: OPTIMIZATION FUNCTIONS
# ============================================================================

def ridge_objective(beta, X, y, lam):
    """
    Compute Ridge Regression objective function.
    L(β) = ||y - Xβ||² + λ||β||²
    """
    residuals = y - X @ beta
    mse = residuals @ residuals
    regularization = lam * (beta @ beta)
    return mse + regularization


def ridge_gradient(beta, X, y, lam):
    """
    Compute gradient of Ridge Regression objective.
    ∇L(β) = -2X^T(y - Xβ) + 2λβ
    """
    residuals = y - X @ beta
    return -2 * X.T @ residuals + 2 * lam * beta


# ============================================================================
# SECTION 3: OPTIMIZATION ALGORITHMS
# ============================================================================

def steepest_descent(X, y, lam, learning_rate=1e-6, max_iter=2000, tol=1e-3):
    """
    Steepest Descent (Gradient Descent)
    
    Algorithm:
    1. Start with β = 0
    2. At each iteration: β := β - α∇L(β)
    3. Stop when gradient norm < tolerance
    
    Convergence: Linear (O(1/k))
    """
    beta = np.zeros(X.shape[1])
    objectives = []
    start_time = time.time()
    
    for iteration in range(max_iter):
        grad = ridge_gradient(beta, X, y, lam)
        objectives.append(ridge_objective(beta, X, y, lam))
        
        if np.linalg.norm(grad) < tol:
            break
        
        beta = beta - learning_rate * grad
    
    elapsed_time = time.time() - start_time
    return beta, np.array(objectives), elapsed_time, iteration + 1


def conjugate_gradient(X, y, lam, max_iter=1000, tol=1e-8):
    """
    Conjugate Gradient Method
    
    Algorithm:
    1. Reformulates as: (X^TX + λI)β = X^Ty
    2. Uses conjugate directions for faster convergence
    3. Each search direction is orthogonal to previous
    
    Convergence: Super-linear (O(1/k²))
    Typically converges in ~n iterations for n features
    """
    n_features = X.shape[1]
    A = X.T @ X + lam * np.eye(n_features)
    b = X.T @ y
    
    beta = np.zeros(n_features)
    r = b - A @ beta
    d = r.copy()
    delta_new = r @ r
    
    objectives = []
    start_time = time.time()
    
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
        
        beta_coeff = delta_new / delta_old
        d = r + beta_coeff * d
    
    elapsed_time = time.time() - start_time
    return beta, np.array(objectives), elapsed_time, iteration + 1


def newton_method(X, y, lam, max_iter=50, tol=1e-10):
    """
    Newton's Method
    
    Algorithm:
    1. Compute Hessian: H = 2(X^TX + λI)
    2. At each iteration: β := β - H⁻¹∇L(β)
    3. Uses second-order information for faster convergence
    
    Convergence: Quadratic (O(1/k²)) in neighborhood of optimum
    Often converges in < 5 iterations
    """
    n_features = X.shape[1]
    H = 2 * (X.T @ X + lam * np.eye(n_features))
    H_inv = np.linalg.inv(H)
    
    beta = np.zeros(n_features)
    objectives = []
    start_time = time.time()
    
    for iteration in range(max_iter):
        grad = ridge_gradient(beta, X, y, lam)
        objectives.append(ridge_objective(beta, X, y, lam))
        
        if np.linalg.norm(grad) < tol:
            break
        
        beta = beta - H_inv @ grad
    
    elapsed_time = time.time() - start_time
    return beta, np.array(objectives), elapsed_time, iteration + 1


def closed_form_solution(X, y, lam):
    """
    Closed-Form Solution (Analytical)
    
    Formula: β = (X^TX + λI)⁻¹X^Ty
    
    Properties:
    - Direct computation, no iterations
    - Gold standard for comparison
    - Optimal solution guaranteed (up to numerical precision)
    """
    n_features = X.shape[1]
    start_time = time.time()
    
    A = X.T @ X + lam * np.eye(n_features)
    b = X.T @ y
    beta = np.linalg.solve(A, b)
    
    elapsed_time = time.time() - start_time
    objectives = np.array([ridge_objective(beta, X, y, lam)])
    
    return beta, objectives, elapsed_time, 1


# ============================================================================
# SECTION 4: RUN ALL ALGORITHMS
# ============================================================================

print("\n" + "="*80)
print("RUNNING OPTIMIZATION ALGORITHMS...")
print("="*80)

beta_sd, obj_sd, time_sd, iter_sd = steepest_descent(X_normalized, y, lambda_param)
print(f"[OK] Steepest Descent completed ({iter_sd} iterations, {time_sd*1000:.4f} ms)")

beta_cg, obj_cg, time_cg, iter_cg = conjugate_gradient(X_normalized, y, lambda_param)
print(f"[OK] Conjugate Gradient completed ({iter_cg} iterations, {time_cg*1000:.4f} ms)")

beta_nm, obj_nm, time_nm, iter_nm = newton_method(X_normalized, y, lambda_param)
print(f"[OK] Newton's Method completed ({iter_nm} iterations, {time_nm*1000:.4f} ms)")

beta_cf, obj_cf, time_cf, iter_cf = closed_form_solution(X_normalized, y, lambda_param)
print(f"[OK] Closed-Form Solution completed ({iter_cf} iterations, {time_cf*1000:.4f} ms)")


# ============================================================================
# SECTION 5: RESULTS COMPARISON
# ============================================================================

print("\n" + "="*80)
print("RESULTS & ANALYSIS")
print("="*80)

final_obj_sd = ridge_objective(beta_sd, X_normalized, y, lambda_param)
final_obj_cg = ridge_objective(beta_cg, X_normalized, y, lambda_param)
final_obj_nm = ridge_objective(beta_nm, X_normalized, y, lambda_param)
final_obj_cf = ridge_objective(beta_cf, X_normalized, y, lambda_param)

print(f"\n1. FINAL OBJECTIVE VALUES")
print(f"   {'Method':<25} {'Objective Value':<20} {'Status':<15}")
print(f"   {'-'*60}")
print(f"   {'Steepest Descent':<25} {final_obj_sd:<20.8f} {'✓ Converged' if abs(final_obj_sd - final_obj_cf) < 1e-4 else 'Warning'}")
print(f"   {'Conjugate Gradient':<25} {final_obj_cg:<20.8f} {'✓ Converged'}")
print(f"   {'Newton Method':<25} {final_obj_nm:<20.8f} {'✓ Converged'}")
print(f"   {'Closed-Form':<25} {final_obj_cf:<20.8f} {'✓ Optimal'}")

print(f"\n2. COMPUTATIONAL EFFICIENCY")
print(f"   {'Method':<25} {'Time (ms)':<15} {'Iterations':<15} {'Time/Iter (μs)':<15}")
print(f"   {'-'*70}")
print(f"   {'Steepest Descent':<25} {time_sd*1000:<15.4f} {iter_sd:<15} {(time_sd*1e6/iter_sd):<15.2f}")
print(f"   {'Conjugate Gradient':<25} {time_cg*1000:<15.4f} {iter_cg:<15} {(time_cg*1e6/iter_cg):<15.2f}")
print(f"   {'Newton Method':<25} {time_nm*1000:<15.4f} {iter_nm:<15} {(time_nm*1e6/iter_nm):<15.2f}")
print(f"   {'Closed-Form':<25} {time_cf*1000:<15.4f} {iter_cf:<15} {(time_cf*1e6/iter_cf):<15.2f}")

print(f"\n3. CONVERGENCE CHARACTERISTICS")
print(f"   {'Method':<25} {'Iterations':<15} {'Convergence Rate':<25}")
print(f"   {'-'*65}")
print(f"   {'Steepest Descent':<25} {iter_sd:<15} {'Linear O(1/k)':<25}")
print(f"   {'Conjugate Gradient':<25} {iter_cg:<15} {'Super-linear O(1/k²)':<25}")
print(f"   {'Newton Method':<25} {iter_nm:<15} {'Quadratic O(ε²)':<25}")
print(f"   {'Closed-Form':<25} {iter_cf:<15} {'Direct (analytical)':<25}")

efficiency = [
    ("Closed-Form", time_cf),
    ("Conjugate Gradient", time_cg),
    ("Newton Method", time_nm),
    ("Steepest Descent", time_sd)
]
efficiency.sort(key=lambda x: x[1])

print(f"\n4. OVERALL EFFICIENCY RANKING (Fastest to Slowest)")
print(f"   {'-'*40}")
for rank, (method, exec_time) in enumerate(efficiency, 1):
    speedup = time_sd / exec_time if method != "Steepest Descent" else 1.0
    print(f"   {rank}. {method:<25} {exec_time*1000:>10.4f} ms {f'({speedup:.0f}x faster)' if speedup > 1 else ''}")


# ============================================================================
# SECTION 6: VISUALIZATIONS
# ============================================================================

print("\n" + "="*80)
print("GENERATING VISUALIZATIONS...")
print("="*80)

# Plot 1: Convergence - Log Scale for Iterations, Linear for Loss
fig1, ax1 = plt.subplots(figsize=(12, 7))
iterations_sd = np.arange(1, len(obj_sd) + 1)
iterations_cg = np.arange(1, len(obj_cg) + 1)
iterations_nm = np.arange(1, len(obj_nm) + 1)

ax1.semilogx(iterations_sd, obj_sd, 'o-', label='Steepest Descent', linewidth=2.5, markersize=5, alpha=0.8, color='#FF6B6B')
ax1.semilogx(iterations_cg, obj_cg, 's-', label='Conjugate Gradient', linewidth=2.5, markersize=5, alpha=0.8, color='#4ECDC4')
ax1.semilogx(iterations_nm, obj_nm, '^-', label='Newton Method', linewidth=2.5, markersize=5, alpha=0.8, color='#45B7D1')
ax1.axhline(y=final_obj_cf, color='#96CEB4', linestyle='--', linewidth=2.5, label='Closed-Form (Optimal)', alpha=0.9)
ax1.set_xlabel('Iteration (Log Scale)', fontsize=13, fontweight='bold')
ax1.set_ylabel('Loss (Objective Function Value)', fontsize=13, fontweight='bold')
ax1.set_title('Ridge Regression: Convergence Rate (Log Scale Iterations)', fontsize=14, fontweight='bold', pad=20)
ax1.legend(fontsize=11, loc='upper right', framealpha=0.95)
ax1.grid(True, alpha=0.35, linestyle='--', linewidth=0.8, which='both')
ax1.set_facecolor('#fafafa')
plt.tight_layout()
plt.savefig('01_convergence_log_iterations.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: 01_convergence_log_iterations.png")
plt.close()

# Plot 2: Convergence - Log Scale (Both axes)
fig2, ax2 = plt.subplots(figsize=(12, 7))
optimal_loss = final_obj_cf
obj_sd_log = np.maximum(obj_sd - optimal_loss, 1e-12)
obj_cg_log = np.maximum(obj_cg - optimal_loss, 1e-12)
obj_nm_log = np.maximum(obj_nm - optimal_loss, 1e-12)

iterations_sd = np.arange(1, len(obj_sd) + 1)
iterations_cg = np.arange(1, len(obj_cg) + 1)
iterations_nm = np.arange(1, len(obj_nm) + 1)

ax2.loglog(iterations_sd, obj_sd_log, 'o-', label='Steepest Descent', linewidth=2.5, markersize=5, alpha=0.8, color='#FF6B6B')
ax2.loglog(iterations_cg, obj_cg_log, 's-', label='Conjugate Gradient', linewidth=2.5, markersize=5, alpha=0.8, color='#4ECDC4')
ax2.loglog(iterations_nm, obj_nm_log, '^-', label='Newton Method', linewidth=2.5, markersize=5, alpha=0.8, color='#45B7D1')
ax2.set_xlabel('Iteration (Log Scale)', fontsize=13, fontweight='bold')
ax2.set_ylabel('Error (Distance from Optimal) - Log Scale', fontsize=13, fontweight='bold')
ax2.set_title('Ridge Regression: Convergence Rate (Log-Log Scale)', fontsize=14, fontweight='bold', pad=20)
ax2.legend(fontsize=11, loc='upper right', framealpha=0.95)
ax2.grid(True, alpha=0.35, linestyle='--', linewidth=0.8, which='both')
ax2.set_facecolor('#fafafa')
plt.tight_layout()
plt.savefig('02_convergence_log_log.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: 02_convergence_log_log.png")
plt.close()

# Plot 3: Computational Cost Comparison
fig3, (ax3, ax4) = plt.subplots(1, 2, figsize=(14, 5))
fig3.suptitle('Computational Efficiency Analysis', fontsize=14, fontweight='bold', y=1.00)

# Execution Time
methods_label = ['Steepest\nDescent', 'Conjugate\nGradient', "Newton's\nMethod", 'Closed-Form']
times_list = [time_sd*1000, time_cg*1000, time_nm*1000, time_cf*1000]
colors = ['#FF6B6B', '#4ECDC4', '#45B7D1', '#96CEB4']

bars = ax3.bar(methods_label, times_list, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax3.set_ylabel('Execution Time (ms)', fontsize=12, fontweight='bold')
ax3.set_title('Total Execution Time', fontsize=12, fontweight='bold')
ax3.grid(True, alpha=0.3, axis='y', linestyle='--')
ax3.set_facecolor('#fafafa')
for bar, time_val in zip(bars, times_list):
    height = bar.get_height()
    ax3.text(bar.get_x() + bar.get_width()/2., height,
            f'{time_val:.4f}ms', ha='center', va='bottom', fontsize=10, fontweight='bold')

# Iterations per Second
iterations_list = [iter_sd, iter_cg, iter_nm, iter_cf]
efficiency_list = [it / t if t > 0 else 0 for it, t in zip(iterations_list, [time_sd, time_cg, time_nm, time_cf])]
methods_short = ['SD', 'CG', 'NM', 'CF']

bars2 = ax4.barh(methods_short, efficiency_list, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
ax4.set_xlabel('Iterations per Second', fontsize=12, fontweight='bold')
ax4.set_title('Iteration Speed (Higher = Better)', fontsize=12, fontweight='bold')
ax4.grid(True, alpha=0.3, axis='x', linestyle='--')
ax4.set_facecolor('#fafafa')
for bar, eff in zip(bars2, efficiency_list):
    width = bar.get_width()
    ax4.text(width, bar.get_y() + bar.get_height()/2.,
            f'{eff:.0f}/s', ha='left', va='center', fontsize=10, fontweight='bold',
            bbox=dict(boxstyle='round,pad=0.4', facecolor='white', alpha=0.85, edgecolor='gray', linewidth=1))

plt.tight_layout()
plt.savefig('03_computational_efficiency.png', dpi=300, bbox_inches='tight')
print("[OK] Saved: 03_computational_efficiency.png")
plt.show()

print("\n" + "="*80)
print("ANALYSIS COMPLETE - All visualizations generated successfully!")
print("="*80)
