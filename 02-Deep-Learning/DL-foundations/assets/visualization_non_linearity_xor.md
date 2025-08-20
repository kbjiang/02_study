
# Why Non-Linearity is Needed in Neural Networks

## XOR Problem: A Classic Example

The XOR function outputs 1 when inputs differ and 0 when they’re the same:

| x₁ | x₂ | XOR(x₁, x₂) |
|----|----|-------------|
| 0  | 0  | 0           |
| 0  | 1  | 1           |
| 1  | 0  | 1           |
| 1  | 1  | 0           |

This function **cannot** be separated by a straight line — it requires a **non-linear** decision boundary.

## Visualization

We generate an XOR-like dataset and apply:

1. **No transformation** — Original space
2. **Linear transformation** — Stretches/rotates the space
3. **Non-linear transformation (tanh)** — Warps the space

```python
import matplotlib.pyplot as plt
import numpy as np

# Generate XOR-like dataset
np.random.seed(0)
n = 100
x1 = np.random.randn(n, 2) * 0.2 + [0, 0]
x2 = np.random.randn(n, 2) * 0.2 + [0, 1]
x3 = np.random.randn(n, 2) * 0.2 + [1, 0]
x4 = np.random.randn(n, 2) * 0.2 + [1, 1]

X = np.vstack([x1, x4, x2, x3])
Y = np.array([0]*n + [0]*n + [1]*n + [1]*n)

# Simple linear transformation
W_linear = np.array([[1, 2], [-2, 1]])
b_linear = np.array([0.5, -0.5])
X_linear = X @ W_linear.T + b_linear

# Non-linear transformation using tanh
X_nonlinear = np.tanh(X_linear)

# Plotting
fig, axs = plt.subplots(1, 3, figsize=(15, 4))

# Original XOR-like distribution
axs[0].scatter(X[Y==0][:,0], X[Y==0][:,1], label="Class 0")
axs[0].scatter(X[Y==1][:,0], X[Y==1][:,1], label="Class 1")
axs[0].set_title("Original (XOR-like) Data")
axs[0].legend()
axs[0].axis('equal')

# After Linear Transformation
axs[1].scatter(X_linear[Y==0][:,0], X_linear[Y==0][:,1], label="Class 0")
axs[1].scatter(X_linear[Y==1][:,0], X_linear[Y==1][:,1], label="Class 1")
axs[1].set_title("After Linear Transformation")
axs[1].legend()
axs[1].axis('equal')

# After Non-linear Transformation (tanh)
axs[2].scatter(X_nonlinear[Y==0][:,0], X_nonlinear[Y==0][:,1], label="Class 0")
axs[2].scatter(X_nonlinear[Y==1][:,0], X_nonlinear[Y==1][:,1], label="Class 1")
axs[2].set_title("After Non-linear Transformation (tanh)")
axs[2].legend()
axs[2].axis('equal')

plt.tight_layout()
plt.show()
```

## Key Takeaway

- **Linear transformations**, even into higher dimensions, do not introduce curves or new decision boundaries.
- **Non-linear functions** like `tanh` or `ReLU` enable the network to warp space, allowing it to model complex boundaries and solve problems like XOR.
