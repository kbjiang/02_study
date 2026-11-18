# Why Cross Entropy, Not MSE, for Comparing $p$ and $q$

MSE on probabilities (the Brier score) also measures distance and is also minimised at $q = p$. Cross entropy, $H(p,q) = -\sum_x p(x)\log q(x)$, is preferred for three reasons.

## 1. It punishes confident wrongness — and is ratio-sensitive

MSE saturates; CE does not (for the true class, $p = 1$):

| $q$ for true class | CE: $-\log q$ | MSE: $(1-q)^2$ |
|---|---|---|
| 0.9 | 0.11 | 0.01 |
| 0.1 | 2.30 | 0.81 |
| 0.01 | 4.61 | 0.98 |
| $\to 0$ | $\to \infty$ | $\to 1$ |

CE is linear in $\log q$, not in $q$ — so loss *changes* depend only on ratios:

$$L(q_1) - L(q_2) = \log\frac{q_2}{q_1}, \qquad dL = -\frac{dq}{q}$$

Numeric example — the same absolute step, $+0.01$, is worth completely different amounts:

- $q: 0.01 \to 0.02$ (ratio $2\times$): CE drops by $\log 2 \approx 0.69$
- $q: 0.51 \to 0.52$ (ratio $1.02\times$): CE drops by $\approx 0.02$

MSE responds to the absolute step and treats the two cases almost the same. Doubling the probability of the true class is worth $\log 2$ wherever it happens.

## 2. Its gradient does not vanish at saturation

With a softmax/sigmoid output:

- **CE:** $\partial L / \partial z_i = q_i - p_i$ — the log cancels the softmax derivative. Confident and wrong ($q \approx 0$, $p = 1$) gives the *largest* gradient, $\approx -1$.
- **MSE:** the gradient picks up an extra factor $q_i(1-q_i) \to 0$ as $q$ saturates. When the model is most confidently wrong, the gradient is near zero and learning stalls.

## 3. It is the natural geometry for probabilities

Probabilities of independent events multiply; logs turn products into sums. So CE is the negative log-likelihood — minimising CE = maximum likelihood — and

$$H(p, q) = H(p) + KL(p \,\|\, q), \qquad KL(p\|q) = \mathbb{E}_p\big[\log(p/q)\big]$$

an expected log-ratio. MSE treats the probability simplex as flat Euclidean space, where a 0.05 gap means the same near 0 as near 0.5; for distributions, it doesn't.

*Where MSE is still used: calibration evaluation (Brier score), soft/regression targets, some distillation. It is a valid distance — just a worse training loss for categorical probabilities, mainly because of Reason 2.*
