## 1. Core Idea
Principal Component Analysis (PCA) is a linear dimensionality reduction technique that finds new orthogonal axes which capture the **maximum variance** in the data.
Goals of PCA:
- Reduce dimensionality
- Remove redundancy from correlated features
- Preserve as much information (variance) as possible

Each **principal component (PC)**:
- Is a linear combination of original features
- Is orthogonal to all previous components
- Explains the maximum remaining variance
---
## 2. Data Setup (Matrix Form)
Let:
- $X \in \mathbb{R}^{n \times d}$ be the data matrix
    - $n$: number of samples
    - $d$: number of features
Assume the data is **mean-centered**:
$$  
\sum_{i=1}^n X_{i,j} = 0  
$$
The sample covariance matrix is:
$$  
\Sigma = \frac{1}{n} X^T X \in \mathbb{R}^{d \times d}  
$$
---
## 3. PCA as an Optimization Problem
We want to find a direction $w$ such that the projected data has **maximum variance**.
### Projection
$$  
z = X w  
$$
### Variance of the projection
$$  
\text{Var}(z) = \frac{1}{n} |Xw|^2  
$$
$$  
= \frac{1}{n} w^T X^T X w  
$$
$$  
= w^T \Sigma w  
$$
---
## 4. Constrained Maximization Problem
We solve:
$$  
\max_w \quad w^T \Sigma w  
$$
subject to:
$$  
|w|^2 = 1  
$$
This constraint prevents the trivial solution $w \to \infty$.

---
## 5. Solving Using Lagrange Multipliers
Define the Lagrangian:
$$  
\mathcal{L}(w, \lambda) =  
w^T \Sigma w - \lambda (w^T w - 1)  
$$
Take derivative with respect to $w$:
$$  
\nabla_w \mathcal{L}  
= 2\Sigma w - 2\lambda w = 0  
$$
$$  
\Rightarrow \Sigma w = \lambda w  
$$
This is an **eigenvalue problem**.
So:
- $w$ is an eigenvector of $\Sigma$
- $\lambda$ is the corresponding eigenvalue

---
## 6. Interpretation

| Quantity             | Meaning             |
| -------------------- | ------------------- |
| Eigenvector $w$      | Principal direction |
| Eigenvalue $\lambda$ | Variance explained  |
Eigenvalues are ordered:
$$  
\lambda_1 \ge \lambda_2 \ge \cdots  
$$
- **First PC**: eigenvector with largest eigenvalue
- **Second PC**: next largest, orthogonal to first
- And so on

Each subsequent component captures the **maximum remaining variance** under the orthogonality constraint.