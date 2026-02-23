# Multi‑Testing in Experiments: A Practical Decision Guide

> Goal: Choose the right statistical treatment depending on what kind of multi‑testing problem you actually have.

This note summarizes when and how to treat multiple tests **correctly**, with emphasis on **A/B testing, replication, and large‑scale experimentation**.

---

## 1. First: Identify what kind of multi‑testing problem you have

Everything hinges on this distinction.

### ❓ Key diagnostic question

> Are we testing multiple different effects, or the same effect multiple times?

|Scenario|What varies?|What stays the same?|
|---|---|---|
|Multiple effects|Hypotheses|Data|
|Same effect, repeated|Data|Hypothesis|

**Different problems → different statistical treatment.**

---

## 2. Case A — Multiple different effects (classic multiple testing)

### Examples

- 5 metrics for one experiment
- 10 features compared to baseline
- Multiple variants vs control

### Statistical risk

- **False positives accumulate**
- Probability of ≥1 false discovery increases with number of tests

### Correct framing

> “Which of these many effects are real?”

---

### A1. Small number of tests (≈ 2–10)

✅ **Control Family‑Wise Error Rate (FWER)**

**Use when:**

- Each false positive is costly
- Tests are confirmatory
- Stakeholders want strong guarantees

#### Recommended methods

- **Bonferroni**: α / m
- **Holm–Bonferroni** (preferred)

#### Interpretation

- If ≥1 rejection → reject *global null* (not all effects are zero)
- Only rejected tests are supported

> Mental model: "Zero tolerance for false positives"

---

### A2. Large number of tests (dozens → thousands)

✅ **Control False Discovery Rate (FDR)**

**Use when:**

- Exploratory / screening analysis
- Many parallel tests
- Some false positives are acceptable

#### Recommended method

- **Benjamini–Hochberg (BH)**

#### Interpretation

- Among all declared significant results, only a controlled fraction are expected to be false

> Mental model: "Manageable error rate among discoveries"

---

## 3. Case B — The same effect tested multiple times

This is **NOT** a multiple‑hypothesis problem.

### Examples

- Re‑running the same A/B test with new users
- Replication across time
- Follow‑up experiment with same metric

### What NOT to do ❌

- Do **not** apply Bonferroni / Holm / BH
- Do **not** count ≥1 rejection as “overall significant”

Why? Because there is **only one underlying hypothesis**.

---

## 4. Correct treatments for repeated testing of the same effect

### B1. Independent replications

✅ **Preferred: pool the data**

**Valid when:**

- Same estimand
- Independent samples
- Same data‑generating process

> Equivalent to running one larger experiment from the start.

Benefits:

- Higher power
- Smaller variance
- Single interpretable p‑value

---

### B2. Heterogeneous runs / contexts

✅ **Meta‑analysis logic**

- Combine effect sizes (weighted)
- Or combine p‑values (e.g. Fisher)

Used when:

- Time / segment effects differ
- Design or environment changes

---

### B3. Sequential peeking / optional stopping

✅ **Sequential testing methods**

- Alpha‑spending
- Group sequential designs
- Bayesian stopping rules

Key idea:

> Allocate error budget over time, not across hypotheses.

---

## 5. Pooling data: when it is correct (and when it is not)

### ✅ Pool data if

- Same hypothesis
- Independent samples
- Comparable populations

### ❌ Do NOT pool if

- Re‑use of users
- Design changed mid‑stream
- Strong heterogeneity ignored

Fallbacks:

- Meta‑analysis
- Regression with run/time fixed effects

---

## 6. Quick decision table

|Situation|Correct approach|
|---|---|
|Few different effects|Holm–Bonferroni|
|Many different effects|Benjamini–Hochberg (FDR)|
|Same effect, more data|Pool data|
|Same effect, varied context|Meta‑analysis|
|Same effect, peeking|Sequential testing|

---

## 7. Core principle to remember

> Multiple‑testing corrections are for many effects.
> Replication and pooling are for one effect tested many times.

Misclassifying the problem leads to:

- Over‑penalization (loss of power), or
- Inflated false positives

---

## 8. Stakeholder‑safe summary

> “Before adjusting for multiple testing, we first decide whether we’re testing many different ideas or validating the same idea repeatedly. These require different statistical treatments. Using the wrong correction answers the wrong question.”

---

## 9. One‑line interview answer

> I first classify whether I’m testing many effects or repeating one effect; only the former needs multiple‑testing correction, while the latter calls for pooling, meta‑analysis, or sequential inference.
