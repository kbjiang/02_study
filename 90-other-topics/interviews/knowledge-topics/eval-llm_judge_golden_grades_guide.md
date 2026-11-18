# Golden Dataset Calibration & Ground Truth Aggregation for LLM Judges

## Reference Statement

> **Defining product policy and golden product manager grades**
> A strong LLM judge begins with a clear product policy and high-quality product manager “golden” grades that demonstrate how that policy should be applied. Product managers act as a “Supreme Court,” regularly calibrating to resolve judgment differences and maintain a shared definition of what constitutes a good query–document match. These discussions refine the policy, making it clearer and less subjective. Once product managers reach high agreement (weighted Cohen’s Kappa ≥ 0.8), their labels are considered reliable ground truth.

---

## 1. What the Statement Means in Practice

In this workflow, product managers (PMs) grade query–document relevance on an ordered scale (such as 1–4 or 1–5) and use pairwise or averaged weighted Cohen's kappa to verify that their written rubric is clear and consistently applied before treating those labels as ground truth.

### Key Components of the Workflow
* **Ordered grading scale:** Because the process relies on **weighted** Cohen's kappa, the PMs assign ordinal relevance scores rather than binary yes/no labels (for example, *Perfect (4), Good (3), Fair (2), Bad (1)*). Weighting ensures that a minor disagreement (3 vs. 4) is penalized much less than a major miss (1 vs. 4).
* **Multi-rater calculation:** Standard weighted Cohen's kappa applies strictly to two raters at a time. When a panel of PMs ("Supreme Court") calibrates together, they typically compute this in one of two ways:
  * **Pairwise averaging (Light's weighted kappa):** Every PM grades the same calibration batch, and the team computes the arithmetic mean of weighted Cohen's kappa across all PM pairs.
  * **Paired calibration:** Two PMs independently grade each query–document pair in a calibration set, and weighted Cohen's kappa is computed across those paired judgments.
* **The ≥ 0.80 quality gate:** Reaching a weighted kappa of 0.80 or higher demonstrates that the product policy is unambiguous enough for different experts to interpret edge cases consistently.
* **Establishing ground truth:** Only after hitting the 0.80 threshold are the labels locked in as "golden" ground truth to evaluate or train the LLM judge. A score below 0.80 indicates that the rubric remains too subjective and requires further refinement.

---

## 2. Python Implementation (Three Raters, 1–4 Scale)

To compute Light's weighted kappa across three PM raters in Python, calculate the pairwise `cohen_kappa_score` from `scikit-learn` for every unique rater pair and take the average.

```python
from itertools import combinations
import numpy as np
from sklearn.metrics import cohen_kappa_score

# Rows = 8 query-document pairs
# Columns = 3 PM raters grading on a 1 (Bad) to 4 (Perfect) scale
ratings = np.array([
    [4, 4, 4],
    [3, 3, 4],
    [2, 2, 2],
    [1, 2, 1],
    [4, 3, 4],
    [1, 1, 1],
    [3, 4, 3],
    [2, 1, 2],
])

n_raters = ratings.shape[1]
pairwise_kappas = []

# Compute weighted Cohen's kappa for every pair of raters
for r1, r2 in combinations(range(n_raters), 2):
    score = cohen_kappa_score(
        ratings[:, r1],
        ratings[:, r2],
        weights="quadratic",  # Use 'linear' for equal-step penalties
    )
    pairwise_kappas.append((f"PM_{r1+1} vs PM_{r2+1}", round(score, 4)))

mean_kappa = np.mean([score for _, score in pairwise_kappas])

for pair, score in pairwise_kappas:
    print(f"{pair}: {score}")

print(f"\nAverage Weighted Kappa: {mean_kappa:.4f}")
```

### Example Output
```text
PM_1 vs PM_2: 0.8500
PM_1 vs PM_3: 0.9444
PM_2 vs PM_3: 0.7746

Average Weighted Kappa: 0.8564
```

### Implementation Notes
* **Pairwise combinations:** Using `itertools.combinations` across 3 raters generates the 3 unique pairs (`PM_1 vs PM_2`, `PM_1 vs PM_3`, and `PM_2 vs PM_3`).
* **Weighting scheme:** Setting `weights="quadratic"` penalizes large grade gaps (such as 1 vs. 4) heavily while treating adjacent differences (such as 3 vs. 4) gently. Switch to `weights="linear"` if each step on the scale carries equal penalty.
* **Pair-level diagnostics:** Inspecting individual pair scores alongside the overall average (`0.8564`) helps identify if a specific pair (`PM_2` and `PM_3` at `0.7746`) diverges on rubric interpretation.

---

## 3. Determining the Final Ground Truth Label

For a golden evaluation set, **adjudication (discussion and consensus)** should be used rather than blind majority vote, especially when raters disagree on ordered relevance scores.

### Adjudication / Consensus (Recommended for Golden Sets)
* **How it works:** Whenever raters disagree on an item (for example, grades of `[4, 3, 2]`), they review the item together, explain their reasoning, and agree on a single final score.
* **Why it works:** Disagreements among domain experts typically expose edge cases or ambiguities in the product policy. Resolving them produces both an accurate golden label and a clearer rubric.

### Automated Aggregation Rules (When Manual Review Is Not Feasible)

| Method | How It Works | Best Used When | Limitations |
| :--- | :--- | :--- | :--- |
| **Median** | Picks the middle score (e.g., `[4, 3, 1]` becomes `3`). | The scale is strictly ordinal (1–4) with an odd number of raters. | Masks edge-case disagreements that may require rubric updates. |
| **Mean (Rounded)** | Averages the grades (e.g., `[4, 4, 3]` becomes `3.67` or `4`). | The LLM judge outputs continuous scores or evaluation uses MSE/correlation. | Produces non-integer values (such as `3.33`) that do not map cleanly to discrete rubric definitions. |
| **Majority Vote** | Picks the most frequent grade (e.g., `[4, 4, 3]` becomes `4`). | Categories are nominal or baseline agreement is very high. | Fails on three-way ties (`[4, 3, 2]`) and ignores the distance between disagreeing scores. |

### Recommended Hybrid Workflow
1. **Auto-accept unanimous or adjacent-majority items:** If all three PMs agree (`[4, 4, 4]`), lock in `4`. If two agree and the third is one step away (`[4, 4, 3]`), auto-accept `4` (or flag for lightweight async review).
2. **Adjudicate wide splits and three-way ties:** Route any item with a 2+ point spread (`[4, 4, 2]`) or a three-way tie (`[4, 3, 2]`) to the PM calibration session to decide the final label and refine the product policy.
