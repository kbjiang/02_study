# Kernel Functions, Kernel Matrices, and Polynomial Mapping

### Core Definitions

* **Kernel Function ($k$):** A function that computes the inner product of data points in a high-dimensional feature space without explicitly mapping them there:
  $$k(x_i, x_j) = \langle \Phi(x_i), \Phi(x_j) \rangle$$
* **Kernel Matrix ($K$):** An $n \times n$ symmetric, positive semi-definite matrix storing pairwise similarities for a dataset of $n$ points:
  $$K_{ij} = k(x_i, x_j)$$

---

### Numeric Example

#### 1. Setup
* **Data Points:** $x_1 = [1, 2]^T$, $x_2 = [3, 4]^T$
* **Kernel Function:** Quadratic polynomial kernel $k(x_i, x_j) = (x_i^T x_j + 1)^2$

#### 2. Resulting Kernel Matrix
Evaluating $k(x_i, x_j)$ for all pairs yields:

$$K = \begin{bmatrix} 36 & 144 \\ 144 & 676 \end{bmatrix}$$

---

### Explicit High-Dimensional Mapping

#### 1. Feature Map
Expanding the kernel algebraically reveals the explicit 6-dimensional feature map $\Phi(x)$ for a 2D input $x = [a_1, a_2]^T$:

$$\Phi(x) = \begin{bmatrix} a_1^2 & a_2^2 & \sqrt{2}a_1a_2 & \sqrt{2}a_1 & \sqrt{2}a_2 & 1 \end{bmatrix}^T$$

#### 2. Mapped Data Points
Applying $\Phi(x)$ projects the original 2D points into 6D space:

$$\Phi(x_1) = \begin{bmatrix} 1 & 4 & 2\sqrt{2} & \sqrt{2} & 2\sqrt{2} & 1 \end{bmatrix}^T$$

$$\Phi(x_2) = \begin{bmatrix} 9 & 16 & 12\sqrt{2} & 3\sqrt{2} & 4\sqrt{2} & 1 \end{bmatrix}^T$$

#### 3. Verification
Computing the standard dot product directly in this 6D space confirms the matrix entry:

$$\langle \Phi(x_1), \Phi(x_2) \rangle = 9 + 64 + 48 + 6 + 16 + 1 = 144$$

This matches $K_{12}$ exactly, demonstrating how the kernel function bypasses explicit high-dimensional projection.
