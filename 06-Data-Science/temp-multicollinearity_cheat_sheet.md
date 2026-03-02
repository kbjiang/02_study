# Multicollinearity & Highly Correlated Features — Cheat Sheet

## What it is
- **Multicollinearity**: predictors are correlated with each other
- **Perfect multicollinearity**: one feature is an exact linear combination of others
- **High (imperfect) correlation**: very common in real data

---

## Practical problems it causes
- Coefficients become **unstable**
- Signs and magnitudes may flip
- **Standard errors inflate**
- Individual p-values become unreliable

**Key point:**  
Predictions are often still good — this is mainly a **stability & interpretability** issue.

---

## Perfect multicollinearity (hard failure)
- No unique coefficient solution
- Software may drop variables or fail

**What to do**
- Drop redundant features
- Combine features (sum, ratio, index)
- Use domain knowledge to keep one

**Perfect collinearity** (x2=x1x_2 = x_1x2​=x1​):
- Column space is 1D
- Infinitely many solutions: β1+β2=c\beta_1 + \beta_2 = cβ1​+β2​=c
- OLS has **no unique solution**
---

## ==High but imperfect correlation (soft problem)==

**Near collinearity** (x2≈x1x_2 \approx x_1x2​≈x1​):
- Projection is still well-defined
- Loss surface has a **flat valley**
- Small noise → large swings in coefficients
- Predictions remain stable; coefficients become unstable

### If your goal is **prediction**
- Use **regularization**
  - Ridge (best default)
  - Elastic Net if sparsity helps
- Cross-validate and move on

### If your goal is **interpretation / inference**
- Drop one of the correlated features
- Combine them into a single variable
- Use PCA / factors if component-level interpretation is acceptable

---

## Feature engineering best practices
- Center and scale features
- Center before creating:
  - Polynomial terms
	  - Polynomial features are a way to _linearize nonlinear dependence_ by expanding the feature space.
  - Interaction terms
- Prefer splines over high-degree polynomials

### detailed explanation 
1. For Polynomial terms
	1. When you have uncentered $x$ and create $x^2$, they're highly correlated: If $x = [1, 2, 3, 4, 5]$, then $x^2 = [1, 4, 9, 16, 25]$. The correlation between $x$ and $x^2$ is **0.98** — almost perfect multicollinearity. Let $x_c = x - \bar{x}$, so $x_c = [-2, -1, 0, 1, 2]$ and $x_c^2 = [4, 1, 0, 1, 4]$. Now the correlation between $x_c$ and $x_c^2$ is **0** — orthogonal!
2. For Interaction Terms
	For interaction terms ($x_1 \cdot x_2$): Centering both variables reduces their correlation with the product term because:
$$\text{Cov}(x_{1c}, x_{1c} \cdot x_{2c}) = E[x_{1c}^2 \cdot x_{2c}] \approx 0$$

---

## Mental model
- Multicollinearity ≠ bias
- Multicollinearity = **variance + instability**
- Fix it only if it hurts what you care about

---

## Interview-ready one-liner
> Multicollinearity doesn’t bias predictions, but it inflates variance and makes coefficients unstable. In practice, we use regularization for prediction and feature selection or combination for interpretability.


### How L2 Regularization Mitigates Multicollinearity (Core Intuition)
**Mathematically**
- Ridge regression solves 
  $$
  \hat{\beta}_{\text{ridge}} = (X^\top X + \lambda I)^{-1} X^\top y
  $$
- Adding $\lambda I$ inflates small eigenvalues of $X^\top X$, making the matrix well-conditioned and the inverse stable.
- This *keeps all features* but prevents coefficient blow‑up when predictors are nearly linearly dependent.
**Correlated Feature Intuition**
- With multicollinearity, the loss surface has *flat directions*: many coefficient combinations fit the data equally well.
- OLS can move freely along these directions, producing large, unstable, opposite‑signed coefficients.
- L2 regularization penalizes coefficient magnitude and selects the **minimum‑norm solution**, effectively sharing weight across correlated features.
**Geometric View**
- OLS: elongated elliptical contours → extreme solutions along correlated axes.
- Ridge: circular L2 constraint