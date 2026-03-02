# Statistical Tests in A/B Testing and Proportion Problems

This note summarizes key points from our discussion on hypothesis testing for binomial (conversion-type) metrics and the relationships among common classical tests.

---
## Q: Two samples of size 50 follow Binomial distribution. How to test if they have same success rate? What if the size is 10?
## 1. Why the t-test is *not* appropriate for binomial problems

In binomial or Bernoulli settings (e.g., conversions, clicks, success/failure):

- Each observation is **binary**, not continuous.
- The data-generating model is **Binomial/Bernoulli**, not Normal.
- The variance is **not a free parameter**. It is fully determined by the mean:

$$
\mathrm{Var}(\hat p) = \frac{p(1-p)}{n}
$$

Although the true success probability $p$ is unknown, once $p$ is estimated (by $\hat p$ or a pooled estimate under the null), the variance is automatically determined.

A t-test is designed for situations where:
- The variance is an **independent, unknown parameter** (e.g., continuous outcomes).
- Estimating that variance introduces extra uncertainty, which is why the **t distribution** is used.
    - This is also why $\text{DOF}=n-1$, because deviations need to obey $\sum(X_i - \bar{X})=0$

For binomial data, this logic does not apply.

**Conclusion:** use **z-tests (or exact tests)**, not t-tests, for binomial/proportion problems.

---

## 2. When sample size is very small: Fisher’s Exact Test vs z-test

For A/B tests with binary outcomes and **small sample sizes or rare events**:

- The normal approximation behind z-tests can break down.
- Expected cell counts in the 2×2 contingency table may be very small (often < 5).

### Fisher’s Exact Test [[temp-fisher_exact_test]]

- Operates on a **2×2 contingency table** of counts.
- Tests the null hypothesis that **treatment and outcome are independent** (equivalently, odds ratio = 1).
- Conditions on the **row and column totals**.
- Computes the **exact probability** of the observed table (and more extreme tables) using the **hypergeometric distribution**.
- Does **not** rely on CLT or large-sample approximations.

**Rule of thumb:**
- Large traffic A/B tests → z-test for proportions
- Small pilots / rare conversions → **Fisher’s exact test**

---

## 3. How the tests relate conceptually (hierarchy)

All of the following tests are manifestations of the same likelihood-based inference framework. They differ mainly in:
- the data type,
- the number of groups,
- and whether variance is a free parameter.

### z-test vs t-test

- Both test **linear contrasts** of a single parameter (mean or proportion).
- **z-test**: variance is known or determined by the model (e.g., proportions).
- **t-test**: variance is an unknown *free* parameter and must be estimated.

As sample size grows:

$$
t \rightarrow z
$$

### t-test vs ANOVA

- **ANOVA is a generalization of the t-test** to more than two groups.
- Instead of asking whether two means are equal, ANOVA asks whether **all group means are equal**.

Key identity:

$$
F = t^2 \quad 	\text{(when there are exactly two groups)}
$$

So, a two-sample t-test is a special case of ANOVA.

### z-test and chi-square (χ²)

- The square of a z-statistic follows a chi-square distribution with 1 degree of freedom:

$$
z^2 \sim \chi^2_1
$$

- Two-sample proportion tests can be written either as **z-tests** or as **χ² tests** on a 2×2 table.

---

## 4. Big-picture equivalences and summary tables

### Conceptual equivalence map

| Situation | Equivalent tests |
|---------|------------------|
| One-sample proportion | z-test ↔ χ²(1) |
| Two-sample proportion | z-test ↔ χ²(1) |
| Two-group mean | t-test ↔ F(1, df) |
| Multi-group mean | ANOVA ↔ likelihood-ratio χ² |
| GLM coefficient | Wald z ↔ LR χ² ↔ Score χ² |

### Summary comparison

| Test | Typical use | Distribution | Core relationship |
|-----|------------|--------------|------------------|
| z-test | Means / proportions | Normal | Score/Wald test |
| t-test | Means (unknown variance) | Student-t | z with variance estimation |
| ANOVA | Multiple means | F | Generalized t-test |
| χ² test | Counts / independence | Chi-square | z² or likelihood distance |

---

## Key takeaways

- Don’t use t-tests for binomial data: the variance is model-determined, not free.
- For very small samples or rare events in A/B tests, Fisher’s exact test is preferable to z-tests.
- z-test, t-test, ANOVA, and χ² are deeply connected; many are equivalent under transformations or in special cases.
