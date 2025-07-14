# Unconstrained Optimization Methods

A comprehensive Jupyter notebook implementation of various unconstrained optimization algorithms using PyTorch for automatic differentiation and numerical optimization.

## 📋 Table of Contents

- [Overview](#overview)
- [Optimization Methods Implemented](#optimization-methods-implemented)
- [Requirements](#requirements)
- [Installation](#installation)
- [Usage](#usage)
- [Method Details](#method-details)
- [Examples](#examples)
- [Mathematical Background](#mathematical-background)
- [Features](#features)

## 🎯 Overview

This notebook provides implementations of fundamental unconstrained optimization algorithms commonly used in machine learning, engineering, and mathematical optimization. All methods are implemented using PyTorch to leverage automatic differentiation for gradient and Hessian computations.

## 🚀 Optimization Methods Implemented

### 1. **Newton-Raphson Method**
- **Type**: Second-order optimization method
- **Features**: 
  - Quadratic convergence near optimal points
  - Handles singular Hessian matrices using Moore-Penrose pseudoinverse
  - Regularization support for ill-conditioned problems
- **Best for**: Functions with well-behaved second derivatives

### 2. **Steepest Descent (Gradient Descent)**
- **Type**: First-order optimization method
- **Features**:
  - Optimal step size computation using line search
  - Adam optimizer for step size optimization
  - Convergence monitoring
- **Best for**: Convex functions and initial exploration

### 3. **Conjugate Gradient Method**
- **Type**: First-order method with second-order convergence properties
- **Features**:
  - Polak-Ribière formula for conjugate direction computation
  - Analytical step size calculation for quadratic functions
  - Memory-efficient for large-scale problems
- **Best for**: Quadratic and near-quadratic functions

### 4. **Fibonacci Search Method**
- **Type**: Line search method for single-variable optimization
- **Features**:
  - Golden ratio-based interval reduction
  - Optimal reduction ratio analysis
  - Visualization of convergence behavior
- **Best for**: Single-variable functions and line search subroutines

### 5. **Quasi-Newton (DFP) Method**
- **Type**: Quasi-Newton method using Davidon-Fletcher-Powell formula
- **Features**:
  - Approximates Hessian inverse using gradient information
  - Superlinear convergence
  - 3D visualization of optimization path
- **Best for**: Functions where Hessian computation is expensive

## 📦 Requirements

```python
torch >= 1.9.0
matplotlib >= 3.3.0
pandas >= 1.3.0
numpy >= 1.21.0
```

## 🛠️ Installation

1. **Clone or download the notebook**
2. **Install dependencies**:
   ```bash
   pip install torch matplotlib pandas numpy
   ```
3. **Run the first cell** to install matplotlib (if needed):
   ```python
   %pip install -U matplotlib
   ```

## 💻 Usage

### Basic Example - Newton-Raphson Method

```python
import torch

# Define your objective function
def objective_function(v):
    x, y, z = v
    return 2*x**2 + 2*x*y + z + 3*y**2

# Set starting point
start_point = torch.tensor([1, -2, 3], dtype=float, requires_grad=True)

# Optimize
gradient, optimal_point = newton_raphson(
    fn=objective_function,
    start_point=start_point,
    epsilon=1e-6,
    iterations=100
)

print(f"Optimal point: {optimal_point}")
```

### Using Steepest Descent

```python
optimal_point = steepest_descent(
    fn=objective_function,
    starting_point=torch.tensor([1, -2, 3], dtype=float, requires_grad=True),
    tolerance=1e-5,
    iterations=100
)
```

## 🔍 Method Details

### Newton-Raphson with Moore-Penrose Alternative

The Newton-Raphson implementation includes a robust handling of singular Hessian matrices:

```python
# Standard Newton step when Hessian is invertible
if torch.det(hessian_matrix) != 0:
    step_size = -torch.matmul(torch.inverse(hessian_matrix), gradient)
else:
    # Moore-Penrose pseudoinverse for singular cases
    step_size = -torch.matmul(torch.pinverse(hessian_matrix), gradient)
```

**Moore-Penrose Pseudoinverse Benefits:**
- **Handles rank-deficient Hessians**: When the Hessian matrix is singular or nearly singular
- **Numerical stability**: More robust than regularization in some cases
- **Automatic rank detection**: PyTorch's `pinverse` automatically handles the rank computation
- **Graceful degradation**: Provides the least-squares solution when exact solutions don't exist

### Convergence Criteria

All methods implement multiple convergence criteria:
- **Gradient norm**: `||∇f(x)|| < ε`
- **Function value change**: `|f(x_{k+1}) - f(x_k)| < tolerance`
- **Point displacement**: `||x_{k+1} - x_k|| < tolerance`

## 📊 Examples

### 1. Rosenbrock Function (Quasi-Newton)
```python
def rosenbrock(v):
    x, y = v
    return 100 * (y - x**2)**2 + (1 - x)**2

optimal_point = dfp_quasi_newton(rosenbrock, torch.tensor([-1.2, 1.0]))
```

### 2. Fibonacci Search for Single Variable
```python
def polynomial(x):
    return x**5 - 5*x**3 - 20*x + 5

result = fib_search(polynomial, xl=-2, xr=2, n=25)
```

### 3. Conjugate Gradient for Quadratic Functions
```python
def quadratic(v):
    x, y = v
    return (3/2)*x**2 + (1/2)*y**2 - x*y + 2*x

optimal_point = conjugate_gradient_method(quadratic)
```

## 📐 Mathematical Background

### Newton-Raphson Update Rule
```
x_{k+1} = x_k - H^{-1}_f(x_k) ∇f(x_k)
```

### Steepest Descent Update Rule
```
x_{k+1} = x_k - α_k ∇f(x_k)
```

### Conjugate Gradient Update Rule
```
x_{k+1} = x_k + α_k d_k
d_{k+1} = -∇f(x_{k+1}) + β_k d_k
```

### DFP Hessian Approximation Update
```
B_{k+1} = B_k + (y_k y_k^T)/(y_k^T s_k) - (B_k s_k s_k^T B_k)/(s_k^T B_k s_k)
```

## ✨ Features

- **Automatic Differentiation**: Uses PyTorch's autograd for exact gradients and Hessians
- **Robust Error Handling**: Graceful handling of singular matrices and numerical issues
- **Visualization**: 3D plots and convergence analysis
- **Flexible Interface**: Easy to adapt for different objective functions
- **Educational Focus**: Clear implementations with detailed logging
- **Performance Monitoring**: Iteration tracking and convergence metrics

## 🎓 Educational Value

This notebook is designed for:
- **Optimization course students** learning fundamental algorithms
- **Researchers** needing reference implementations
- **Engineers** requiring robust optimization tools
- **Data scientists** understanding ML optimization foundations

## 🔧 Customization

Each method can be customized with:
- **Convergence tolerances**
- **Maximum iterations**
- **Starting points**
- **Regularization parameters**
- **Step size strategies**

## 📈 Performance Considerations

- **Newton-Raphson**: Fast convergence but expensive Hessian computation
- **Steepest Descent**: Slow but reliable, good for ill-conditioned problems
- **Conjugate Gradient**: Good balance of speed and robustness
- **Quasi-Newton**: Approximates Newton with lower computational cost

## 🤝 Contributing

Feel free to extend this notebook with:
- Additional optimization methods (BFGS, L-BFGS, etc.)
- More sophisticated line search strategies
- Constraint handling methods
- Performance benchmarking

## 📚 References

1. Nocedal, J., & Wright, S. J. (2006). Numerical optimization
2. Boyd, S., & Vandenberghe, L. (2004). Convex optimization
3. Bertsekas, D. P. (2016). Nonlinear programming

---

**Author**: Gourav K Sinha
**Last Updated**: 2025  
**License**: Educational Use
