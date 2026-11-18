## 1. The ordering intuition (the core idea)

AUC is not really measuring classification. It is measuring **ordering**:

> **AUC = the probability that a randomly chosen clicked job gets a higher score than a randomly chosen not-clicked job.**

Equivalently, in a finite dataset: form every clicked × not-clicked pair, and count the fraction of pairs the model orders correctly (a tied pair counts as half). That fraction is the empirical AUC (the Mann–Whitney form).

| AUC    | Meaning                                                                                                                                                                                                                      |
| ------ | ---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| 0.50   | No ordering information — random                                                                                                                                                                                             |
| 1.00   | Every clicked job scores above every not-clicked job                                                                                                                                                                         |
| < 0.50 | Systematically backwards: clicked jobs score *lower*. Flipping the score gives AUC $1 - \text{AUC}$, so 0.20 flipped is 0.80. Below 0.5 usually signals an inverted label / sign bug; a merely *weak* model hovers near 0.50 |
| 0.00   | Perfectly backwards                                                                                                                                                                                                          |
## 2. Why AUC is used for job ranking
1. **Ordering is the job.** A ranker never makes a binary call — it does not say "score > 0.5, therefore click." It sorts jobs by score and shows the top ones. A metric should therefore measure ordering, not classification at some operating point.
2. **Threshold-agnostic — which follows from (1).** Accuracy, F1, or precision would require first inventing a threshold to turn scores into click / no-click predictions, and the result would depend on that arbitrary choice — a threshold that exists nowhere in the real system. AUC is performance averaged over *every* possible threshold (the ROC curve is TPR vs FPR as the threshold is swept), so no threshold has to be invented. Note the direction of the logic: threshold-freedom is the *property*; measuring ordering is the *reason* AUC is chosen.
3. **Click labels are free and plentiful.** Human relevance judgments are expensive and small-scale; click logs give millions of real decisions to evaluate on held-out traffic.
4. **Accuracy would be useless.** Most shown jobs are not clicked, so "always predict no click" gets high accuracy and ranks nothing. AUC is not fooled by that class imbalance.

## 3. Proof: AUC $= P(A > B)$
### Proof 1
Let
- $A$ = score of a randomly chosen clicked job, with density $f_A$
- $B$ = score of a randomly chosen not-clicked job, with density $f_B$
- $A$ and $B$ are **independent draws** (see the note below)
**Step 1 — TPR and FPR at a threshold $t$.**
$$\text{TPR}(t) = P(A > t), \qquad \text{FPR}(t) = P(B > t)$$
**Step 2 — AUC as an integral over the threshold.**
AUC is the area under TPR plotted against FPR. Substitute $u = \text{FPR}(t)$:
$$\text{AUC} = \int_0^1 \text{TPR}\, du$$
Since $u = P(B>t)$, we have $du = -f_B(t)\,dt$. As $t$ goes from $+\infty$ to $-\infty$, $u$ goes from $0$ to $1$, and the minus sign flips the limits back:
$$\text{AUC} = \int_{-\infty}^{\infty} P(A > t)\, f_B(t)\, dt$$
**Step 3 — Read the integral as an expectation over $B$.**
In that integral, $t$ is not a random variable — it is a fixed value that $B$ might take, weighted by $f_B(t)$. Averaging $P(A>t)$ over all values of $B$ is conditioning on $B$ (law of total probability):
$$\text{AUC} = \mathbb{E}_B\big[\,P(A > B \mid B)\,\big] = P(A > B) \qquad \blacksquare$$
**Notes on the proof**
- **Independence assumption:** $A$ and $B$ are independent *by construction* — they are two separate draws the evaluator makes, one from the clicked-score distribution and one from the not-clicked-score distribution. This is what lets the joint density factorise, $f_{A,B}(a,b) = f_A(a)f_B(b)$. It does **not** claim that two jobs in the same real result list are independent (they share query, user, and context, and can be correlated); AUC is defined over the independent-draws pair.
- **Ties:** the proof above assumes continuous scores, so ties have probability zero. With ties, the exact statement is $\text{AUC} = P(A>B) + \tfrac{1}{2}P(A=B)$.
### Proof 2
Let $A$ = score of a randomly chosen clicked job and $B$ = score of a randomly chosen not-clicked job, drawn independently, with densities $f_A$ and $f_B$. Define $Z = A - B$. We calculate $P(Z > 0)$ and show it equals AUC.
**1. Density of $Z$ (convolution for a difference)**
$Z = z$ means $A = z + B$. Condition on $B = b$: then $A$ must equal $z + b$, with density $f_A(z + b)$, and $b$ itself has density $f_B(b)$. Averaging over all $b$ (this uses independence of $A$ and $B$, i.e. $f_{A,B}(a, b) = f_A(a) f_B(b)$):
$$f_Z(z) = \int_{-\infty}^{\infty} f_A(z + b) \, f_B(b) \, db$$
**2. Integrate over $z > 0$**
$$P(Z > 0) = \int_0^{\infty} f_Z(z) \, dz
= \int_0^{\infty} \int_{-\infty}^{\infty} f_A(z + b) \, f_B(b) \, db \, dz$$
Swap the order of integration (the integrand is $\ge 0$, so Fubini
allows it):
$$P(Z > 0) = \int_{-\infty}^{\infty} f_B(b)
\left[ \int_0^{\infty} f_A(z + b) \, dz \right] db$$
The inner integral: substitute $u = z + b$, so $z = 0 \to u = b$ and
$z = \infty \to u = \infty$:
$$\int_0^{\infty} f_A(z + b) \, dz = \int_b^{\infty} f_A(u) \, du = P(A > b)$$
Therefore:
$$P(Z > 0) = \int_{-\infty}^{\infty} f_B(b) \, P(A > b) \, db$$
**3. This is exactly AUC**
That integral is what the ROC calculation in Proof 1 reduced to, $\int P(A > t) f_B(t) \, dt$, with the dummy variable renamed $t \to b$. Hence:
$$\text{AUC} = P(Z > 0) = P(A - B > 0) = P(A > B) \qquad \blacksquare$$
*Note:* this assumes continuous scores, so ties have probability zero. With ties, $\text{AUC} = P(A > B) + \tfrac{1}{2} P(A = B)$.

## One-line version
Click AUC asks one question — *"when the model scores a clicked job and a not-clicked job, how often does it put them in the right order?"* — and 0.61 → 0.67 means: right order 61% of the time before, 67% after.
