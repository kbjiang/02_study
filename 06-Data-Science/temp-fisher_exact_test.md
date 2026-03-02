# Fisher’s Exact Test — Numeric Worked Example

This note shows **Fisher’s Exact Test with concrete numbers**, step by step, exactly how probabilities and the p-value are computed.

---

## 1. Observed A/B Test Result

Suppose we run a small A/B test:

|            | Success | Failure | Total |
|------------|---------|---------|-------|
| Treatment  | 8       | 2       | 10    |
| Control    | 4       | 6       | 10    |
| **Total**  | 12      | 8       | 20    |

Question:
> Are success rates different between Treatment and Control?

---

## 2. Null Hypothesis

Under the null hypothesis:

> Treatment assignment and success are independent.

Given this null:
- Total successes = 12
- Total failures = 8
- Treatment size = 10

Only **how many successes land in Treatment** is random.

---

## 3. Probability of the Observed Table

Let `a = number of Treatment successes`.

Observed value:
```
a = 8
```

Fisher’s Exact Test uses the **hypergeometric distribution**:

$$
P(a) = \frac{\binom{12}{a} \binom{8}{10-a}}{\binom{20}{10}}
$$

Plug in `a = 8`:

$$
P(8) = \frac{\binom{12}{8} \binom{8}{2}}{\binom{20}{10}}
     = \frac{495 \times 28}{184756}
     \approx 0.075
$$

---

## 4. All Possible Tables with Fixed Margins

Possible values of `a`:

$$
\max(0, 10-8)=2 \le a \le \min(10,12)=10
$$

So we enumerate `a = 2,3,4,...,10`.

| a (Treatment Successes) | Probability |
|-------------------------|-------------|
| 2 | 0.004 |
| 3 | 0.018 |
| 4 | 0.054 |
| 5 | 0.117 |
| 6 | 0.176 |
| 7 | 0.196 |
| **8 (observed)** | **0.075** |
| 9 | 0.020 |
| 10 | 0.002 |

(Values rounded for readability.)

---

## 5. Computing the Two-Sided p-value

For Fisher’s **two-sided** test:

> Sum probabilities of all tables **as unlikely or more unlikely than the observed one**.

Observed probability:
```
P_obs = 0.075
```

Include all tables with `P(a) ≤ 0.075`:

```
a = 2, 3, 4, 8, 9, 10
```

Sum:

$$
	\text{p-value} = 0.004 + 0.018 + 0.054 + 0.075 + 0.020 + 0.002
               \approx 0.173
$$

---

## 6. Interpretation

- p-value ≈ **0.17**
- Not statistically significant at 0.05

Even though Treatment looks much better (80% vs 40%),
**the sample is too small** to rule out chance.

---

## 7. Why This Matters in A/B Testing

- Exact (no normal approximation)
- Valid with very small samples
- Correct Type I error even when expected counts < 5

---

## 8. Interview One-Liner

> Fisher’s Exact Test fixes the row and column totals and computes the exact probability of every possible 2×2 table using the hypergeometric distribution, summing probabilities of tables at least as unlikely as the observed one.
