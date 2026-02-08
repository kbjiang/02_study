# Understanding OLS Output Statistics

## F-Statistic

### What It Tests

The F-statistic tests the **overall significance of the regression model**.

Null hypothesis:
$$H_0: \beta_1 = \beta_2 = \cdots = \beta_k = 0$$

vs. the alternative that **at least one coefficient is non-zero**.

### Formula

$$F = \frac{\text{Explained Variance} / k}{\text{Residual Variance} / (n - k - 1)}$$

### Intuition

**F is a signal-to-noise ratio** — it compares how much variance the model explains vs. how much unexplained noise remains.

| Scenario | F value | Meaning |
|----------|---------|---------|
| Model captures strong signal | High F | Explained variance >> residual noise |
| Model barely helps | F ≈ 1 | Explained ≈ unexplained (could be random) |
| Model is useless | F < 1 | You're explaining less than random chance |

---

## t-Statistic

### What It Tests

The t-statistic tests whether a **single coefficient is significantly different from zero**.

$$t = \frac{\hat{\beta} - 0}{\text{SE}(\hat{\beta})} = \frac{\text{estimated effect}}{\text{uncertainty in estimate}}$$

### Intuition

t asks: **"Is this individual predictor's effect real, or could it be zero?"**

| Scenario | t value | Meaning |
|----------|---------|---------|
| Strong clear effect | \|t\| > 2 | Coefficient is far from zero relative to uncertainty |
| Weak/uncertain effect | \|t\| ≈ 0-1 | Effect size is small compared to noise |
| No effect | t ≈ 0 | Estimated coefficient is basically zero |

---

## F vs t Comparison

| Statistic | Tests | Question |
|-----------|-------|----------|
| **t** | One coefficient = 0? | "Is *this* predictor useful?" |
| **F** | All coefficients = 0? | "Is *any* predictor useful?" |

---

## Special Case: F = t² When k = 1

For simple linear regression with one predictor, F equals t².

### Proof

**Setup**: $y = \beta_0 + \beta_1 x + \epsilon$

Key definitions:
- $S_{xx} = \sum(x_i - \bar{x})^2$
- $S_{xy} = \sum(x_i - \bar{x})(y_i - \bar{y})$
- $\hat{\beta}_1 = S_{xy}/S_{xx}$
- $\text{SE}(\hat{\beta}_1) = \sqrt{\text{MSE}/S_{xx}}$

**Compute t²:**
$$t^2 = \frac{\hat{\beta}_1^2}{\text{SE}(\hat{\beta}_1)^2} = \frac{\hat{\beta}_1^2 \cdot S_{xx}}{\text{MSE}} = \frac{S_{xy}^2}{S_{xx} \cdot \text{MSE}}$$

**Compute F:**
$$\text{SSR} = \hat{\beta}_1^2 \cdot S_{xx} = \frac{S_{xy}^2}{S_{xx}}$$

$$F = \frac{\text{SSR}}{\text{MSE}} = \frac{S_{xy}^2}{S_{xx} \cdot \text{MSE}}$$

**Conclusion:** Both equal $\frac{S_{xy}^2}{S_{xx} \cdot \text{MSE}}$, therefore $F = t^2$.

---

## Degrees of Freedom: A Deep Dive

### Basic Definitions

**DF Model:**
$$\text{DF Model} = k$$

Where $k$ = number of predictors (excluding intercept). This is how many parameters (slopes) the model estimates.

**DF Residuals:**
$$\text{DF Residuals} = n - k - 1$$

Where:
- $n$ = sample size
- $k$ = number of predictors
- $1$ = for the intercept

This is how many independent data points remain after fitting the model.

### Why "Residuals"?

Residuals are the leftover errors: $e_i = y_i - \hat{y}_i$

DF Residuals is used to estimate the **residual variance** (MSE):
$$\text{MSE} = \frac{\sum e_i^2}{n - k - 1}$$

---

### Why n - k - 1, Not n? A Concrete Example

#### Simple Case: Sample Variance (n - 1)

Suppose you have 3 numbers: 2, 5, 8

**Mean**: $\bar{x} = 5$

The deviations from mean must sum to zero:
- $(2-5) + (5-5) + (8-5) = -3 + 0 + 3 = 0$ ✓

If I tell you two deviations are -3 and 0, you **already know** the third must be +3.

→ Only 2 of the 3 deviations are "free" → **DF = n - 1 = 2**

#### Regression Case (n - k - 1)

Say you fit: $y = \beta_0 + \beta_1 x$

With 5 data points:

| i | x | y | ŷ | residual (e) |
|---|---|---|---|--------------|
| 1 | 1 | 3 | 2.8 | 0.2 |
| 2 | 2 | 4 | 3.6 | 0.4 |
| 3 | 3 | 5 | 4.4 | 0.6 |
| 4 | 4 | 4 | 5.2 | -1.2 |
| 5 | 5 | 6 | 6.0 | **?** |

**Constraint 1** (from intercept): Residuals sum to zero
$$\sum e_i = 0$$
$$0.2 + 0.4 + 0.6 + (-1.2) + e_5 = 0 \implies e_5 = 0$$

**Constraint 2** (from slope): Residuals are uncorrelated with x
$$\sum x_i \cdot e_i = 0$$

These 2 constraints mean once you know 3 residuals, the other 2 are determined.

→ **DF Residuals = 5 - 1 - 1 = 3**

#### General Pattern

| What you estimate | Constraints imposed | DF lost |
|-------------------|---------------------|---------|
| Intercept $\beta_0$ | $\sum e_i = 0$ | 1 |
| Slope $\beta_1$ | $\sum x_1 e_i = 0$ | 1 |
| Slope $\beta_2$ | $\sum x_2 e_i = 0$ | 1 |
| ... | ... | ... |

Each parameter estimated = one linear constraint on residuals = one less free residual.

**Total**: $n - (k + 1) = n - k - 1$ free residuals to estimate variance.

#### Why This Matters

Using $n$ instead of $n-k-1$ would **underestimate** the true variance — you'd be dividing by too large a number, making your estimates look more precise than they really are.

---

### Why Are Residuals Uncorrelated with X?

This isn't an assumption — it's a **mathematical consequence of how OLS works**.

#### OLS Minimizes Squared Errors

OLS finds $\hat{\beta}_0, \hat{\beta}_1$ by minimizing:

$$\text{SSE} = \sum (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2$$

Taking the derivative with respect to $\hat{\beta}_1$ and setting to zero:

$$\frac{\partial \text{SSE}}{\partial \hat{\beta}_1} = -2 \sum x_i (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i) = 0$$

Since $e_i = y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i$:

$$\sum x_i \cdot e_i = 0$$

#### Intuition

If residuals were correlated with x, **OLS would adjust the slope to exploit that correlation**.

Think of it this way:
- Positive correlation ($\sum x_i e_i > 0$): High x values have positive errors → slope is too shallow → OLS would increase $\hat{\beta}_1$
- Negative correlation ($\sum x_i e_i < 0$): High x values have negative errors → slope is too steep → OLS would decrease $\hat{\beta}_1$

OLS **keeps adjusting until** there's no remaining correlation to exploit.

#### Visual Example

```
Before optimal fit:              After optimal fit:
(residuals correlated with x)    (residuals uncorrelated)

    •                                •  
      •  ← positive errors             • ← random errors
        •    at high x                   •
──────────────                   ────────•─────
  •                                •       
    •  ← negative errors             •   
      •    at low x                    •
```

The line rotates until errors are randomly distributed across all x values.

#### Same Logic for Intercept

$$\frac{\partial \text{SSE}}{\partial \hat{\beta}_0} = -2 \sum (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i) = 0$$

This gives: $\sum e_i = 0$

The intercept adjusts until errors average to zero.

#### Summary: OLS Optimality Conditions

| Condition | Mathematical | Why |
|-----------|--------------|-----|
| $\sum e_i = 0$ | First-order condition for $\beta_0$ | Intercept centers the line |
| $\sum x_i e_i = 0$ | First-order condition for $\beta_1$ | Slope removes x-error correlation |

These are **optimality conditions**, not assumptions. OLS guarantees them by construction.

---

### Geometric View of Degrees of Freedom ^geometric-view

#### Two Ways to View the Data

There are **two dual geometric views** of regression data:

**View 1: Row Space (Each observation is a point)**

Each row = one data point in **p-dimensional feature space** (where p = number of features).

```
        x₂
        |       • obs 3
        |   • obs 1
        |           • obs 2
        └───────────── x₁
```

- n points in $\mathbb{R}^p$
- Used for: clustering, PCA, visualization
- **Not** the view for understanding DF

**View 2: Column Space (Each variable is a vector)**

Each column = one vector in **n-dimensional observation space**.

```
        obs₃
        |       
        |   → x₁ (predictor vector)
        |       → y (response vector)  
        └───────────── obs₁
              ↗
            obs₂
```

- p+1 vectors in $\mathbb{R}^n$
- Used for: OLS geometry, understanding DF (see [Degrees of Freedom, Actually Explained - The Geometry of Statistics](https://youtu.be/VDlnuO96p58?list=TLPQMDMwMjIwMjbJ-Nrl2sTgmg))
- Each vector has n components (one per observation)

#### Why Column Space for OLS?

In OLS, we're asking: *"What linear combination of predictor columns best approximates the response column?"*

$$\hat{\mathbf{y}} = \beta_0 \mathbf{1} + \beta_1 \mathbf{x}_1 + \cdots + \beta_k \mathbf{x}_k$$

This is a **linear combination of column vectors**, so we work in $\mathbb{R}^n$.

#### Concrete Example

3 observations, 2 features:

| obs | $x_1$ | $x_2$ | $y$ |
|-----|-------|-------|-----|
| 1   | 2     | 5     | 10  |
| 2   | 3     | 1     | 8   |
| 3   | 7     | 4     | 15  |

**Row view**: 3 points in $\mathbb{R}^2$
$$\text{obs}_1 = (2, 5), \quad \text{obs}_2 = (3, 1), \quad \text{obs}_3 = (7, 4)$$

**Column view**: 3 vectors in $\mathbb{R}^3$
$$\mathbf{x}_1 = \begin{pmatrix} 2 \\ 3 \\ 7 \end{pmatrix}, \quad \mathbf{x}_2 = \begin{pmatrix} 5 \\ 1 \\ 4 \end{pmatrix}, \quad \mathbf{y} = \begin{pmatrix} 10 \\ 8 \\ 15 \end{pmatrix}$$

OLS projects $\mathbf{y}$ onto the plane spanned by $\mathbf{1}, \mathbf{x}_1, \mathbf{x}_2$ in $\mathbb{R}^3$.

#### The Projection

The columns of $\mathbf{X}$ (including the intercept column of 1s) span a **(k+1)-dimensional subspace** of $\mathbb{R}^n$.

```
           y (true response)
          /|
         / |
        /  | e (residual)
       /   |
      /    |
     ●─────● ŷ (projection onto column space)
    
    ←───────────────→
     Column space of X
     (k+1 dimensions)
```

OLS finds the point $\hat{\mathbf{y}}$ in this subspace **closest to $\mathbf{y}$**.

The residual vector $\mathbf{e}$ is **perpendicular to the column space of X**:
- $\mathbf{e} \perp \mathbf{1}$ → $\sum e_i = 0$ (perpendicular to intercept column)
- $\mathbf{e} \perp \mathbf{x}_j$ → $\sum x_{ij} e_i = 0$ (perpendicular to each predictor)

#### Degrees of Freedom as Dimensions

| Space | Dimension | Meaning |
|-------|-----------|---------|
| Full data space | n | All possible residual directions |
| Column space of X | k + 1 | Directions "used up" by model |
| **Orthogonal complement** | **n - k - 1** | Directions residuals can vary freely |

**DF Residuals = dimension of the space residuals live in = n - k - 1**

#### Visual: 2D Analogy

Imagine $n = 3$ observations and $k = 1$ predictor (plus intercept, so k+1 = 2 columns).

```
        z
        |   • y (3D point)
        |  /|
        | / | e (residual)
        |/  |
        ●───● ŷ (on the plane)
       /
      /
     ────────── y
    /    Plane = column space of X
   x     (2 dimensions)
```

- $\mathbf{y}$ lives in 3D space ($\mathbb{R}^3$)
- Column space of X is a 2D plane
- $\hat{\mathbf{y}}$ = projection of $\mathbf{y}$ onto this plane
- $\mathbf{e}$ = perpendicular distance from plane

The residual can only point in **1 direction** (perpendicular to the 2D plane).

→ **DF Residuals = 3 - 1 - 1 = 1 dimension**

#### Why Perpendicularity?

OLS minimizes $\|\mathbf{e}\|^2 = \|\mathbf{y} - \hat{\mathbf{y}}\|^2$.

The shortest distance from a point to a subspace is always **perpendicular**.

If $\mathbf{e}$ had any component parallel to the column space, you could reduce error by moving $\hat{\mathbf{y}}$ in that direction — contradicting optimality.

#### Summary: Geometric vs Algebraic

| Concept | Algebraic | Geometric |
|---------|-----------|-----------|
| $\sum e_i = 0$ | Constraint equation | $\mathbf{e} \perp \mathbf{1}$ |
| $\sum x_j e_i = 0$ | Constraint equation | $\mathbf{e} \perp \mathbf{x}_j$ |
| DF Residuals = n-k-1 | Free residual values | Dimension of orthogonal complement |
| OLS solution | Normal equations | Projection onto column space |

| View | Space | Dimension | Used for |
|------|-------|-----------|----------|
| Row (observations as points) | $\mathbb{R}^p$ | p features | Clustering, PCA, visualization |
| Column (variables as vectors) | $\mathbb{R}^n$ | n observations | OLS geometry, degrees of freedom |

The constraints aren't arbitrary — they're a consequence of projecting onto a lower-dimensional subspace. Each basis vector of that subspace removes one dimension from where residuals can point.

---

## Quick Reference

| Metric | Formula | Tests |
|--------|---------|-------|
| F-statistic | MSR / MSE | All coefficients = 0? |
| t-statistic | β̂ / SE(β̂) | Single coefficient = 0? |
| DF Model | k | # of predictors |
| DF Residuals | n - k - 1 | # free observations |
