# Statistical Nuances of Small Sample Sizes: The 5-Run Accuracy Example

When evaluating machine learning classifiers with a limited number of runs (e.g., $n = 5$), standard evaluation heuristics often conflict with strict mathematical reality. This document summarizes the statistical mechanics behind reporting error bars, calculating variance, and estimating confidence intervals for tiny sample sizes.

---

## 1. Calculating the Standard Error of the Mean (SEM)

When evaluating $n = 5$ distinct accuracy scores ($x_1, x_2, \dots, x_5$), calculating the standard error involves two distinct scaling factors in the denominator: $\sqrt{4}$ and $\sqrt{5}$.

### Step 1: Sample Standard Deviation ($s$)
To measure the natural variability between individual runs, we calculate the sample standard deviation. We use **Bessel's Correction** ($n-1 = 4$) in the denominator to ensure an unbiased estimator of the true population variance:
$$s = \sqrt{\frac{\sum_{i=1}^n (x_i - \bar{x})^2}{4}}$$

### Step 2: Scaling for the Mean (SEM)
The Standard Error of the Mean (SEM) measures the stability of the *average accuracy* rather than individual runs. By basic algebraic properties of variance, the variance of an average drops by a factor of $n$:
$$\text{Var}(\bar{x}) = \frac{s^2}{n} = \frac{s^2}{5}$$

Taking the square root yields the standard error formula:
$$\text{SEM} = \frac{s}{\sqrt{5}} = \frac{\sqrt{\frac{\sum (x_i - \bar{x})^2}{4}}}{\sqrt{5}} = \sqrt{\frac{\sum (x_i - \bar{x})^2}{20}}$$

---

## 2. Does the Central Limit Theorem (CLT) Apply at $n = 5$?

**No.** The Central Limit Theorem is an asymptotic theorem requiring a large sample size (typically $n \ge 30$) to guarantee that the distribution of the sample mean converges to a normal bell curve. At $n = 5$, the distribution of the sample mean remains highly sensitive to the underlying skewness of your accuracy scores.

### How to Resolve This:
* **The Formula is Valid:** You do not need the CLT to justify the $\sqrt{n}$ scaling in the SEM denominator. That is a pure algebraic property of independent variances, valid at any sample size.
* **The Multiplier Changes:** Because the CLT does not apply, you cannot use the standard normal $z$-score multiplier of $1.96$ for a 95% Confidence Interval. Instead, you must use the **Student's $t$-distribution** with $n-1 = 4$ degrees of freedom. This penalizes the small sample size by pushing the critical multiplier up to **$2.776$**.

---

## 3. The Algebraic Source of $\sqrt{n}$ vs. The Law of Large Numbers

* **Pure Algebra:** The $\sqrt{n}$ denominator is not a statistical law. It comes directly from the algebraic rule that scaling a random variable by $a$ alters its variance by $a^2$. When averaging, we scale the sum by $1/n$, meaning variance is scaled by $1/n^2$.
* **The Weak Law of Large Numbers (WLLN):** The WLLN is a *consequence* of this algebra. By placing $\sqrt{n}$ in the denominator of standard error, Chebyshev's Inequality proves that as $n \to \infty$, the probability of the sample mean drifting away from the true mean drops strictly to zero.

---

## 4. Practical Implementation (Python)

Below is the standard, rigorous way to extract reporting metrics from $5$ evaluation runs:

```python
import numpy as np
from scipy import stats

# 5 Example evaluation runs
accuracies = np.array([0.82, 0.85, 0.81, 0.84, 0.83])

mean = np.mean(accuracies)
sd = np.std(accuracies, ddof=1)  # Uses n-1 (4)
sem = stats.sem(accuracies)       # Uses s / sqrt(5)

# 95% Confidence Interval using Student's t-distribution (Critical t = 2.776)
ci_margin = stats.t.ppf(0.975, df=len(accuracies)-1) * sem

print(f"Mean ± SD:  {mean:.4f} ± {sd:.4f}")
print(f"Mean ± SEM: {mean:.4f} ± {sem:.4f}")
print(f"Mean ± CI:  {mean:.4f} ± {ci_margin:.4f}")
```

---

## 5. Summary Matrix for Error Bars

| Option | Formula | Best Used For... | What It Tells the Reader |
| :--- | :--- | :--- | :--- |
| **Standard Deviation (SD)** | $\text{Mean} \pm s$ | Showing raw experimental flux. | "If you run this model one more time, here is how much the score is likely to swing." |
| **Standard Error (SEM)** | $\text{Mean} \pm \frac{s}{\sqrt{n}}$ | Showing the baseline precision of your average. | "Here is the statistical stability of our recorded average accuracy." |
| **Confidence Interval (CI)** | $\text{Mean} \pm (t_{crit} \times \text{SEM})$ | Rigorous statistical benchmarking on small samples. | "We are 95% confident that the true, long-term average performance falls within this range." |
