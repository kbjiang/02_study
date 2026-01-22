# Why Bootstrapping Does Not (Directly) Follow the Central Limit Theorem

## 1. What the Central Limit Theorem (CLT) says

The CLT applies when:

- We have i.i.d. samples from a **fixed population distribution**
- We compute the **mean** (or sum)
- Sample size goes to infinity

Formally:

sqrt(n) (X̄ − μ) → N(0, σ²)

Key assumptions:
- Independent observations
- Drawn from the true population
- Large sample size

---

## 2. What bootstrapping actually does

In bootstrapping:

1. We observe one dataset: X₁, …, Xₙ
2. We treat the empirical distribution F̂ₙ as the population
3. We repeatedly resample *with replacement* from the same data
4. We recompute a statistic (mean, median, regression coefficient, etc.)

So bootstrap samples are:
- Not drawn from the true population
- Drawn from a discrete empirical distribution
- All dependent on the same original sample

---

## 3. Why CLT intuition breaks down

### (a) Not averaging independent population draws

Bootstrap samples are independent *conditional on the data*, but:
- They reuse the same observations
- Extreme values may appear many times
- The “population” has only n atoms

This violates the classical CLT setup.

---

### (b) Different probability questions

CLT asks:
"What is the distribution of a statistic across repeated real-world samples?"

Bootstrap asks:
"Given one observed sample, how variable could this statistic be?"

These live in **different probability spaces**.

---

### (c) Many bootstrap statistics are not means

CLT applies cleanly to:
- Sample means
- Sums

Bootstrap is often used for:
- Medians
- Quantiles
- Maxima
- Regression coefficients
- Other non-smooth statistics

For these:
- CLT may not apply
- Limiting distributions may be non-normal
- Bootstrap distributions may be skewed or irregular

---

## 4. When bootstrap *does* align with CLT

For smooth statistics (e.g., the mean):

- The sampling distribution is asymptotically normal (by CLT)
- The bootstrap distribution converges to the same normal distribution
- Bootstrap and CLT give similar standard errors

So bootstrap does **not contradict** the CLT.

---

## 5. When bootstrap does not look normal

Bootstrap distributions may be:
- Skewed
- Heavy-tailed
- Multi-modal

Common causes:
- Small sample size
- Skewed underlying data
- Nonlinear or non-smooth statistics
- Parameters on boundaries (e.g., variance ≥ 0)

CLT does not protect you in these cases.

---

## 6. Core intuition

CLT:
"If I could repeatedly sample from the real world..."

Bootstrap:
"Given what I saw once, how variable could this statistic be?"

CLT adds new randomness.  
Bootstrap reuses old randomness.

---

## 7. One-sentence takeaway

Bootstrapping does not follow the CLT because it resamples from the empirical distribution rather than the true population, so the CLT’s assumptions do not directly apply.

When the CLT applies, bootstrap agrees with it; when it doesn’t, bootstrap reveals the limitations.
