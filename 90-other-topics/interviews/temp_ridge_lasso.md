# Geometry of Feature Rotation and Ridge vs Lasso

## 1. Rotation / Change of Basis in Feature Space

We start with two original features:

- \(x_1\)
- \(x_2\)

and define new features:

- \(x_3 = x_1 + x_2\)
- \(x_4 = x_1 - x_2\)

Geometrically, this is a **45° rotation (plus scaling)** of the original feature space. The transformation simply changes the coordinate system — it does *not* create new information.

In vector form:

\[
egin{bmatrix}x_3\x_4\end{bmatrix}
=
egin{bmatrix}1 & 1\1 & -1\end{bmatrix}
egin{bmatrix}x_1\x_2\end{bmatrix}
\]

After normalization by \(1/\sqrt{2}\), this becomes an orthonormal rotation.

---

## 2. Geometry of Ridge vs Lasso

The figure below shows two classic geometric ideas in coefficient space:

- **Gray ellipses**: contours of equal loss (e.g., RSS)
- **Blue circle**: Ridge (L2) constraint
- **Red diamond**: Lasso (L1) constraint

![Rotation and Ridge vs Lasso Geometry](geometry_ellipses_loss_touch_l2_circle_vs_l1_diamond.png)

### Ridge (L2 Regularization)

Ridge regression solves:

- Minimize RSS subject to \(\|eta\|_2 \le t\)

The feasible region is a **circle**. Because the circle is smooth, the loss contours usually touch it away from the axes.

**Result:** coefficients are shrunk but rarely exactly zero.

---

### Lasso (L1 Regularization)

Lasso solves:

- Minimize RSS subject to \(\|eta\|_1 \le s\)

The feasible region is a **diamond** with sharp corners aligned to the axes.

**Result:** the optimal solution often occurs at a corner → one or more coefficients are exactly zero (sparsity).

---

## 3. Why Rotation Matters

- **OLS** is invariant to linear reparameterization: predictions do not change.
- **Ridge** is rotationally invariant: L2 norm is round.
- **Lasso is *not* rotationally invariant**: rotating the feature space changes where the diamond’s corners lie.

This explains why feature engineering (e.g., using sums/differences or PCA directions) can dramatically affect Lasso solutions.

---

## 4. Practical Intuition

- If signal lies along \(x_1 pprox x_2\), then \(x_3\) captures most information.
- Lasso will often keep \(x_3\) and drop \(x_4\).
- Ridge will keep both but shrink them.

---

**One-line takeaway:**

> Lasso prefers axis-aligned sparsity, while ridge prefers smooth shrinkage — and rotating the feature space changes what “axis-aligned” means.
