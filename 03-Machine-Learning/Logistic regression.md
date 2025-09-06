# Orientation and steepness

## 1. Perfectly vs. Almost Linearly Separable Data
- **Dataset A (perfectly separable):**
  - Logistic regression can keep increasing the magnitude of coefficients.
  - This makes the decision boundary sharper and the sigmoid curve steeper.
  - Without regularization, the maximum likelihood solution is *not finite* (coefficients → ∞).
  - With regularization (e.g., L2), coefficients are capped but still larger compared to noisy data.

- **Dataset B (almost separable):**
  - A few points fall on the "wrong" side of the boundary.
  - Logistic regression finds finite coefficients that balance classification performance across all points.
  - Coefficients are smaller in magnitude, leading to a smoother sigmoid transition.

## 2. Role of Coefficients
For logistic regression:
$P(y=1|x) = \sigma(w^\top x + b), \quad \sigma(z) = \frac{1}{1 + e^{-z}}$
- **Direction of \(w\):** Defines the orientation of the separating hyperplane.
- **Magnitude of \(w\):** Controls the steepness of the sigmoid transition across the boundary.

At the decision boundary $w^\top x + b = 0$:
$\nabla_x P(y=1|x) = 0.25 \, w$
- Larger \(\|w\|\) → steeper probability change (more confident boundary).
- Smaller \(\|w\|\) → smoother probability change (less confident boundary).

## 3. Key Takeaways
- Perfectly separable data pushes logistic regression coefficients toward *infinity* unless regularized.
- Noisy or nearly separable data leads to finite, smaller coefficients.
- The *line’s orientation* stays similar, but the *steepness of classification* (driven by coefficient magnitude) differs.

---
# Appendix: Python Demonstrations

## A. Comparing Coefficients (Dataset A vs. Dataset B)
```python
import numpy as np
from sklearn.linear_model import LogisticRegression

# Seed for reproducibility
np.random.seed(42)

# --- Dataset A: perfectly separable ---
X_A = np.vstack([
    np.random.randn(20, 2) + np.array([2, 2]),   # class 1
    np.random.randn(20, 2) + np.array([-2, -2])  # class 0
])
y_A = np.array([1]*20 + [0]*20)

# --- Dataset B: almost separable, add noise ---
X_B = X_A.copy()
y_B = y_A.copy()
flip_idx = np.random.choice(len(y_B), size=3, replace=False)
y_B[flip_idx] = 1 - y_B[flip_idx]

# Train logistic regression with L2 regularization
clf_A = LogisticRegression(solver="lbfgs").fit(X_A, y_A)
clf_B = LogisticRegression(solver="lbfgs").fit(X_B, y_B)

coeffs_A = np.hstack([clf_A.coef_.flatten(), clf_A.intercept_])
coeffs_B = np.hstack([clf_B.coef_.flatten(), clf_B.intercept_])

# similar direction, but `coeffs_A` will be larger in magnitude
print("Dataset A coefficients:", coeffs_A)
print("Dataset B coefficients:", coeffs_B)
