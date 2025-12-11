# Ridge Regression Optimization Methods Analysis

## Project Overview

This project compares **four different optimization algorithms** for solving Ridge Regression on real-world NBA team statistics data. The analysis demonstrates how different mathematical approaches solve the same problem with dramatically different performance characteristics.

**Dataset:** NBA Team Statistics (2000-2023)  
**Total Samples:** 716 team-seasons  
**Features:** 7 key basketball metrics  
**Target:** Win percentage prediction  

---

## Table of Contents

1. [Quick Summary](#quick-summary)
2. [The Problem: Ridge Regression](#the-problem-ridge-regression)
3. [Optimization Algorithms](#optimization-algorithms)
4. [Results & Analysis](#results--analysis)
5. [Key Insights](#key-insights)
6. [Visualizations](#visualizations)
7. [Technical Details](#technical-details)

---

## Quick Summary

| Method | Time | Iterations | Speed | Convergence |
|--------|------|-----------|-------|------------|
| **Newton's Method** | 0.166 ms | 2 | 676x faster | Quadratic |
| **Closed-Form** | 0.218 ms | 1 | 514x faster | Direct |
| **Conjugate Gradient** | 0.476 ms | 7 | 236x faster | Super-linear |
| **Steepest Descent** | 112.3 ms | 2000 | Baseline | Linear |

**Winner:** Newton's Method achieves convergence in just **2 iterations** and runs **676× faster** than Steepest Descent.

---

## The Problem: Ridge Regression

### What is Ridge Regression?

Ridge Regression is a linear regression variant that adds regularization to prevent overfitting:

$$L(\beta) = \|y - X\beta\|^2 + \lambda\|\beta\|^2$$

Where:
- **X** = Feature matrix (716 × 7) - normalized basketball statistics
- **y** = Target vector - win percentages
- **β** = Coefficients we want to find
- **λ** = Regularization parameter (λ = 1.0) - controls complexity

### Goal

Find the coefficient vector **β** that minimizes the loss function. This gives us a linear model that:
- Fits the training data well
- Generalizes to unseen data
- Has smooth, reasonable coefficients

### Features Used

1. **field_goal_percentage** - Overall shooting efficiency
2. **three_point_percentage** - Three-point shooting ability
3. **assists** - Ball movement and playmaking
4. **turnovers** - Ball security and team discipline
5. **rebounds** - Possession control
6. **steals** - Perimeter defense
7. **blocks** - Interior defense

---

## Optimization Algorithms

### 1. Steepest Descent (Gradient Descent)

**Mathematical Approach:**
- Uses only first-order information (gradient)
- Always moves in the steepest downhill direction

$$\beta_{k+1} = \beta_k - \alpha \nabla L(\beta_k)$$

Where:
- α = learning rate (step size)
- ∇L = gradient of loss function

**Characteristics:**
- ✓ Simple to understand and implement
- ✓ Works for any differentiable function
- ✗ Slow convergence (linear rate)
- ✗ Requires careful tuning of learning rate

**Performance:**
- **Iterations:** 2000
- **Convergence Type:** Linear (O(1/k)) - error reduces proportionally with iterations
- **Speed:** 112.33 ms (baseline)

**When to use:**
- Very large datasets where memory is limited
- Online learning scenarios
- When speed per iteration matters more than total time

---

### 2. Conjugate Gradient Method

**Mathematical Approach:**
- Reformulates the problem as a linear system: $(X^TX + \lambda I)\beta = X^Ty$
- Uses conjugate directions - each new direction is orthogonal to all previous ones
- Each iteration provides significant progress toward the solution

$$\begin{align}
\text{Find:} \quad & \beta \\
\text{such that:} \quad & A\beta = b \\
\text{where:} \quad & A = X^TX + \lambda I, \quad b = X^Ty
\end{align}$$

**Characteristics:**
- ✓ Much faster than gradient descent
- ✓ No hyperparameter tuning needed
- ✓ Guaranteed convergence in n iterations for n-dimensional problem
- ✗ More complex to implement

**Performance:**
- **Iterations:** 7
- **Convergence Type:** Super-linear (O(1/k²)) - error reduces quadratically
- **Speed:** 0.476 ms (236× faster than Steepest Descent)
- **Iterations per second:** 14,722

**When to use:**
- Medium-sized problems (thousands of features)
- When memory allows storing a few vectors
- Good balance of speed and simplicity

---

### 3. Newton's Method

**Mathematical Approach:**
- Uses second-order information (Hessian matrix)
- Approximates the loss function with a quadratic model
- Jumps directly to the minimum of the approximation

$$\beta_{k+1} = \beta_k - H^{-1} \nabla L(\beta_k)$$

Where:
- H = Hessian matrix (2nd derivatives) = 2(X^TX + λI)
- H^{-1} = Inverse of Hessian

**Characteristics:**
- ✓ Fastest convergence (quadratic - O(ε²))
- ✓ Few iterations needed
- ✗ Computing Hessian inverse is expensive: O(n³) time
- ✗ May not work if Hessian is singular

**Performance:**
- **Iterations:** 2
- **Convergence Type:** Quadratic (O(ε²)) - error squares each iteration
- **Speed:** 0.166 ms (676× faster than Steepest Descent)
- **Iterations per second:** 12,019

**When to use:**
- When number of features is moderate (< 10,000)
- When total runtime is critical
- When you can afford O(n³) computation for Hessian

---

### 4. Closed-Form Solution (Analytical)

**Mathematical Approach:**
- Directly solves the normal equations using algebra
- No iteration needed - compute the answer directly

$$\beta = (X^TX + \lambda I)^{-1} X^Ty$$

**Characteristics:**
- ✓ Guaranteed optimal solution (up to numerical precision)
- ✓ No iterations, no approximation
- ✓ Well-studied numerical methods available
- ✗ Same O(n³) cost as Newton's method
- ✗ Not applicable to non-convex problems

**Performance:**
- **Iterations:** 1
- **Convergence Type:** Direct (analytical) - no iteration
- **Speed:** 0.218 ms (514× faster than Steepest Descent)
- **Gold Standard:** All other methods should converge to this value

**When to use:**
- Benchmark/reference solution
- Final check for convergence of iterative methods
- Small to medium problems where O(n³) is acceptable

---

## Results & Analysis

### 1. Final Objective Values

All methods converge to the same optimal solution (within numerical precision):

```
Steepest Descent:    186.36 (Warning: did not fully converge)
Conjugate Gradient:  185.94 ✓
Newton Method:       185.94 ✓
Closed-Form:         185.94 ✓ (Optimal)
```

**Interpretation:** Three out of four methods found the true optimal value. Steepest Descent needs more iterations due to its slow linear convergence.

### 2. Computational Efficiency

| Method | Execution Time | Iterations | Time per Iteration |
|--------|-----------------|------------|-------------------|
| Steepest Descent | 112.33 ms | 2000 | 56.16 μs |
| Conjugate Gradient | 0.48 ms | 7 | 67.98 μs |
| Newton Method | 0.17 ms | 2 | 83.09 μs |
| Closed-Form | 0.22 ms | 1 | 218.39 μs |

**Key Observation:** Newton's Method is fastest overall despite taking longer per iteration, because it needs so few iterations!

### 3. Convergence Behavior

The three plots (see Visualizations section) show:

1. **Linear Scale Plot:** Shows absolute loss values
   - Steepest Descent: Gradual, slow improvement
   - Conjugate Gradient: Rapid early progress
   - Newton Method: Dramatic drop to optimal in 2 steps

2. **Log Scale Plot:** Shows convergence rate clearly
   - Steepest Descent: Shallow linear decline (linear convergence)
   - Conjugate Gradient: Progressively steeper decline (super-linear)
   - Newton Method: Vertical drops (quadratic convergence)

3. **Efficiency Plot:** Direct comparison
   - Execution time bar chart
   - Iterations per second comparison

---

## Key Insights

### 1. **Iteration Count ≠ Total Time**

Steepest Descent performs 2000 iterations but takes 112 ms.  
Newton's Method performs 2 iterations and takes 0.17 ms.

This is because Newton's method does more expensive per-iteration computation (matrix inversion), but the benefit of needing so few iterations vastly outweighs this cost.

### 2. **Convergence Rate Matters**

- **Linear (Steepest Descent):** Error reduction: 0.5×, 0.25×, 0.125× ...
- **Super-linear (Conjugate Gradient):** Error reduction: 0.5×, 0.1×, 0.001× ...
- **Quadratic (Newton):** Error reduction: 0.5×, 0.25×, 0.0625× ...

Quadratic convergence means Newton's method hits the solution exponentially faster.

### 3. **Problem Structure Matters**

Conjugate Gradient is specifically designed for quadratic problems (which Ridge Regression is), giving it super-linear convergence. This makes it ideal for this type of problem.

### 4. **Trade-offs**

| Criterion | Winner |
|-----------|--------|
| Fastest total time | Newton's Method |
| Most reliable (fewest tuning) | Conjugate Gradient |
| Simplest to code | Steepest Descent |
| Most memory efficient | Steepest Descent |
| Best for large datasets | Steepest Descent or Conjugate Gradient |

---

## Visualizations

### Plot 1: Convergence Rate (Linear Scale)
**File:** `01_convergence_linear.png`

Shows the loss function value at each iteration for all methods. 

**What to observe:**
- Steepest Descent slowly climbs toward the solution
- Conjugate Gradient and Newton's methods rapidly reach optimality
- The horizontal dashed line shows the true optimal value

**Interpretation:** Visually demonstrates why Newton's method is superior - it reaches the goal in just 2 iterations while Steepest Descent needs 2000.

---

### Plot 2: Convergence Rate (Log Scale)
**File:** `02_convergence_log.png`

Shows distance from optimal solution on logarithmic scale.

**What to observe:**
- Log scale makes convergence rates visible as slopes
- Steepest Descent: shallow slope (slow improvement)
- Conjugate Gradient: accelerating slope (super-linear)
- Newton's Method: nearly vertical (quadratic - dramatic jumps)

**Interpretation:** The slope becomes steeper with better convergence rates. Newton's nearly vertical line shows exponential improvement.

---

### Plot 3: Computational Efficiency
**File:** `03_computational_efficiency.png`

Two sub-plots comparing execution time and iteration speed.

**Left subplot (Execution Time):**
- Newton's Method: ~0.17 ms
- Closed-Form: ~0.22 ms
- Conjugate Gradient: ~0.48 ms
- Steepest Descent: ~112 ms

**Right subplot (Iterations per Second):**
- Shows how many iterations each method completes per second
- Newton's Method: ~12,000 iterations/second
- Despite being slower per iteration, the overall time is best

---

## Technical Details

### Data Preprocessing

1. **Feature Standardization:**
   - Each feature was normalized to zero mean and unit variance
   - Formula: $x_{normalized} = \frac{x - \mu}{\sigma}$
   - Why: Ensures features on similar scales; improves numerical stability

2. **Feature Selection:**
   - Started with 29 available columns
   - Selected 7 features with clear basketball interpretation
   - Removed missing values (none found in this dataset)

### Algorithm Parameters

| Algorithm | Key Parameters | Values Used |
|-----------|-----------------|-------------|
| Steepest Descent | Learning rate (α) | 1e-6 |
| | Max iterations | 2000 |
| | Tolerance | 1e-8 |
| Conjugate Gradient | Max iterations | 1000 |
| | Tolerance | 1e-8 |
| Newton's Method | Max iterations | 50 |
| | Tolerance | 1e-10 |
| Closed-Form | - | Direct computation |

### Implementation Notes

1. **Numerical Stability:** 
   - Used feature normalization to keep matrix condition numbers reasonable
   - Used `np.linalg.solve()` instead of explicit matrix inversion for numerical robustness

2. **Convergence Criteria:**
   - All iterative methods check if gradient norm falls below tolerance
   - Newton's method uses stricter tolerance due to quadratic convergence

3. **Timing Measurements:**
   - Used `time.time()` for wall-clock measurements
   - Includes all computation including objective function evaluation

---

## Mathematical Foundations

### The Ridge Regression Problem

We want to find **β** that minimizes:

$$L(\beta) = \frac{1}{2}\|y - X\beta\|^2 + \frac{\lambda}{2}\|\beta\|^2$$

This is a **convex optimization problem** - has a unique global minimum.

### Gradient

The gradient is:
$$\nabla L(\beta) = -X^T(y - X\beta) + \lambda\beta = -X^Ty + (X^TX + \lambda I)\beta$$

### Hessian

The Hessian (matrix of second derivatives) is:
$$H = X^TX + \lambda I$$

This is **positive definite** (all eigenvalues > 0), which guarantees:
- The function is strictly convex
- Any local minimum is the global minimum
- Newton's method converges quadratically

### Normal Equations

Setting gradient to zero:
$$-X^Ty + (X^TX + \lambda I)\beta = 0$$
$$(X^TX + \lambda I)\beta = X^Ty$$

This linear system's solution is the optimal **β**.

---

## How to Use This Code

### Running the Analysis

```bash
python ridge_regression_analysis.py
```

This will:
1. Load the NBA dataset from Kaggle
2. Preprocess and normalize the features
3. Run all 4 optimization algorithms
4. Print detailed results and comparisons
5. Generate 3 visualization PNG files

### Output Files

```
01_convergence_linear.png       # Convergence on linear scale
02_convergence_log.png          # Convergence on log scale  
03_computational_efficiency.png  # Cost and speed comparison
README.md                        # This documentation
```

### Modifying the Analysis

**To change features:**
Edit the `features` list in Section 1.

**To change regularization:**
Modify `lambda_param = 1.0` (higher = more regularization)

**To change learning rate (Steepest Descent):**
In `steepest_descent()` function, modify `learning_rate=1e-6`

**To change convergence tolerance:**
Modify the `tol` parameters in each function

---

## References & Further Reading

### Optimization Methods
- Nocedal & Wright. "Numerical Optimization" (2006) - Comprehensive reference
- Boyd & Vandenberghe. "Convex Optimization" (2004) - Theoretical foundations
- Conjugate Gradient: Hestenes & Stiefel (1952) - Original paper

### Ridge Regression
- Tikhonov & Arsenin. "Solutions of Ill-posed Problems" (1977)
- Hastie, Tibshirani, & Friedman. "Elements of Statistical Learning" (2009)

### Numerical Methods
- Golub & Pereyra. "Differentiation of pseudoinverses and nonlinear least squares problems whose variables separate"

---

## Summary

This analysis demonstrates that **choosing the right optimization algorithm matters significantly**. For Ridge Regression:

- **Steepest Descent:** 112 ms, 2000 iterations
- **Conjugate Gradient:** 0.48 ms, 7 iterations  
- **Newton's Method:** 0.17 ms, 2 iterations ⭐
- **Closed-Form:** 0.22 ms, 1 iteration (reference)

Newton's Method achieves **676× speedup** over Steepest Descent by using second-order information and exploiting the problem's convex structure. For Ridge Regression specifically, the Conjugate Gradient method offers an excellent balance of speed and simplicity.

---

**Project Completed:** December 2025  
**Dataset:** NBA Team Statistics (2000-2023), 716 samples  
**Framework:** NumPy, Pandas, Matplotlib  
