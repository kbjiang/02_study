# Why Vanilla Policy Gradient Has High Variance & A Full Numeric Example

This document contains:

1.  An explanation of **why** vanilla REINFORCE / policy gradient has
    high variance.\
2.  A complete **numeric example** showing how a baseline reduces
    variance.

------------------------------------------------------------------------

# Part 1 --- Why Does Vanilla Policy Gradient Have High Variance?

In vanilla (REINFORCE-style) policy gradient, we estimate:

\[

abla\_`\theta `{=tex}J = `\mathbb{E}`{=tex}`\left[ 
    \sum_t 
abla_\theta \log \pi_\theta(a_t|s_t) \, G_t
\right]`{=tex}. \]

This estimator has **high variance** due to four structural reasons:

------------------------------------------------------------------------

## **1. Multiplying many stochastic terms**

A single sample of the gradient involves:

-   stochastic action selection\
-   stochastic environment transitions\
-   stochastic rewards\
-   full return (G_t), which aggregates noise from *all* future steps

Thus:

\[

abla\_`\theta `{=tex}`\log `{=tex}`\pi`{=tex}\_`\theta`{=tex}(a_t\|s_t),
G_t \]

is a very noisy random variable.

------------------------------------------------------------------------

## **2. Long-term credit assignment amplifies noise**

Rewards arriving far in the future influence early actions:

\[ G_1 = r_1 + r_2 + `\dots `{=tex}+ r_T. \]

Small randomness late in the trajectory can cause large changes in
(G_1), creating high-variance gradient contributions for earlier steps.

------------------------------------------------------------------------

## **3. Rewards are unnormalized**

Returns (G_t) are:

-   large\
-   unbounded\
-   correlated\
-   often have long tails

Multiplying them by ( abla `\log `{=tex}`\pi`{=tex}) amplifies
variability.

------------------------------------------------------------------------

## **4. Using a single trajectory to estimate a huge expectation**

The true gradient averages over exponentially many trajectories.\
But REINFORCE uses only a small batch of samples → very noisy averages.

------------------------------------------------------------------------

## **Why baselines help**

Using a baseline:

\[

abla J = `\mathbb{E}`{=tex}`\left[
abla\log\pi(a_t|s_t)\,(G_t - b(s_t))ight]`{=tex}\]

reduces variance because:

-   subtracting something correlated with (G_t) reduces the random
    magnitude of (G_t - b)
-   the estimator remains unbiased because\
    ( `\mathbb{E}`{=tex}\[ abla `\log`{=tex}`\pi`{=tex}(a_t\|s_t)\] = 0
    )

Value-function baselines and advantages (e.g., GAE) provide *major*
variance reduction.

------------------------------------------------------------------------

# Part 2 --- Full Numeric Example Showing Variance Reduction

A small RL problem:

-   1-step episode\
-   Bernoulli policy with parameter (p = 0.6)\
-   Rewards:
    -   Action 1 → reward 1 with probability 0.8\
    -   Action 0 → reward 1 with probability 0.2

The score function terms:

\[

abla_p `\log`{=tex}`\pi`{=tex}(a{=}1)=1/p = 1.6666...,\

abla_p `\log`{=tex}`\pi`{=tex}(a{=}0)=-1/(1-p) = -2.5. \]

------------------------------------------------------------------------

# **1. Expected Gradient**

\[ `\mathbb{E}`{=tex}\[ abla`\log`{=tex}`\pi`{=tex}(a)G\] = 0.6
`\cdot 1.6666`{=tex} `\cdot 0.8`{=tex} + 0.4 `\cdot `{=tex}(-2.5)
`\cdot 0.2`{=tex} = 0.6. \]

------------------------------------------------------------------------

# **2. Variance Without a Baseline**

Compute:

\[ `\operatorname{Var}`{=tex} = `\mathbb{E}`{=tex}\[(
abla`\log`{=tex}`\pi`{=tex}`\cdot `{=tex}G)\^2\] - (0.6)\^2. \]

Final result:

\[ `\operatorname{Var}`{=tex}\_{ ext{no baseline}} pprox 1.4733. \]

------------------------------------------------------------------------

# **3. Optimal Constant Baseline**

Optimal:

\[ b\^\* = `\frac{ \mathbb{E}[
abla\log\pi(a)^2 G] }`{=tex} { `\mathbb{E}`{=tex}\[
abla`\log`{=tex}`\pi`{=tex}(a)\^2 \] }. \]

Numerically:

\[ b\^\* pprox 0.44. \]

------------------------------------------------------------------------

# **4. Variance With Optimal Baseline**

\[ `\operatorname{Var}`{=tex}\_{b\^\*} pprox 0.6667. \]

That's a **55% variance reduction**.

------------------------------------------------------------------------

# Summary Table

  Case                        Variance
  --------------------------- ----------
  No baseline                 \~1.4733
  Optimal constant baseline   \~0.6667

------------------------------------------------------------------------

# Key Takeaway

Baselines reduce variance because they remove components of the return
that are not predictive of the policy gradient direction---without
biasing the estimator.

This is the foundation for practical methods like actor--critic, GAE,
and advantage-based policy optimization.
