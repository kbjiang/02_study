## KL divergence (and entropy)

Three quantities, one information-theoretic picture (measured in *nats* with $\ln$, *bits* with $\log_2$):

- **Entropy** $H(p) = -\sum_i p_i \log p_i$ — the average surprise of samples from $p$; the irreducible cost to encode them. Uniform = high, peaked = low.

- **Cross-entropy** $H(p,q) = -\sum_i p_i \log q_i$ — the cost of encoding samples from $p$ using a code built for the *wrong* distribution $q$.

- **KL divergence** $D_{\mathrm{KL}}(p \parallel q) = \sum_i p_i \log \frac{p_i}{q_i}$ — precisely that *extra* cost of using $q$ instead of $p$.

They are one identity:

$$
H(p,q) = H(p) + D_{\mathrm{KL}}(p \parallel q)
$$

So minimizing cross-entropy is minimizing KL to the truth — $H(p)$ is fixed by the data. In LM training the label $p$ is one-hot, so $H(p) = 0$ and **cross-entropy = KL = NLL**. In RLHF the KL term is instead added *on purpose*, to keep the policy near its reference.
