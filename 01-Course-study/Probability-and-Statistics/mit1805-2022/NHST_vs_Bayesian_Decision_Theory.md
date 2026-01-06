# NHST vs Bayesian Decision Theory: Side-by-Side Examples

This document contrasts **Null Hypothesis Significance Testing (NHST)**
with **Bayesian decision theory** using the same real-world decision
problems.\
The key point: **both frameworks can justify the same actions, but for
different reasons.**

------------------------------------------------------------------------

## Example 1: Drug Approval

### NHST framing

-   **H₀**: Drug is no better than placebo\
-   **H₁**: Drug improves outcomes\
-   **α = 0.01**
-   **Decision rule**: Approve only if H₀ is rejected

**Guarantee**: \[ P(`\text{approve ineffective drug}`{=tex}
`\mid `{=tex}H_0) `\le 1`{=tex}% \]

This controls regulatory risk and protects patients.

------------------------------------------------------------------------

### Bayesian framing

-   Prior belief about drug efficacy (from preclinical studies)
-   Likelihood from trial data
-   Posterior: \[ P(`\text{drug effective}`{=tex}
    `\mid `{=tex}`\text{data}`{=tex}) \]

**Decision rule**: Approve if: \[ E\[`\text{utility}`{=tex}
`\mid `{=tex}`\text{approve}`{=tex}\] \> E\[`\text{utility}`{=tex}
`\mid `{=tex}`\text{reject}`{=tex}\] \]

Where utility accounts for: - Patient benefit - Harm from side effects -
Cost of delaying approval

**Key difference**: Bayesian decision-making depends explicitly on
**priors and utilities**.

------------------------------------------------------------------------

## Example 2: A/B Testing (Product Rollout)

### NHST framing

-   **H₀**: New design does not improve conversion
-   **α = 0.05**
-   Roll out only if H₀ is rejected

**Guarantee**: - At most 5% of rollouts are false positives in the long
run

This allows predictable business risk.

------------------------------------------------------------------------

### Bayesian framing

-   Prior on lift (often centered near zero)
-   Posterior distribution over conversion lift

**Decision rule**: Roll out if: \[ E\[`\text{profit gain}`{=tex}
`\mid `{=tex}`\text{data}`{=tex}\] \> 0 \]

This can: - Roll out earlier with strong priors - Delay rollout if
uncertainty remains costly

**Tradeoff**: More flexible, but sensitive to prior assumptions.

------------------------------------------------------------------------

## Example 3: Manufacturing Quality Control

### NHST framing

-   **H₀**: Mean strength ≥ safety threshold
-   **α = 0.001**
-   Shut down production if H₀ is rejected

**Guarantee**: \[ P(`\text{unnecessary shutdown}`{=tex})
`\le 0.1`{=tex}% \]

This is legally and operationally defensible.

------------------------------------------------------------------------

### Bayesian framing

-   Prior on defect rate
-   Posterior probability strength \< threshold

**Decision rule**: Shut down if: \[ P(`\text{unsafe}`{=tex}
`\mid `{=tex}`\text{data}`{=tex}) \> c \] where (c) reflects safety
tolerance.

**Advantage**: Allows explicit tradeoff between safety risk and shutdown
cost.

------------------------------------------------------------------------

## Example 4: Scientific Screening

### NHST framing

-   Thousands of hypotheses
-   Very small α (e.g., 0.001)
-   Only significant effects get follow-up

**Purpose**: Control false discoveries and conserve resources.

------------------------------------------------------------------------

### Bayesian framing

-   Hierarchical priors over effect sizes
-   Posterior probabilities of nonzero effects

**Decision rule**: Follow up if: \[
P(`\text{effect is meaningful}`{=tex} `\mid `{=tex}`\text{data}`{=tex})
\> c \]

More efficient, but computationally and conceptually heavier.

------------------------------------------------------------------------

## Key Conceptual Difference

  Aspect             NHST                  Bayesian Decision Theory
  ------------------ --------------------- --------------------------
  Output             Reject / Not reject   Posterior + decision
  Error control      Long-run guarantees   Depends on priors
  Subjectivity       Minimal               Explicit
  Interpretability   Procedural            Belief-based
  Auditability       High                  Medium
  Flexibility        Low                   High

------------------------------------------------------------------------

## Why NHST Still Matters

NHST is preferred when: - Decisions must be **defensible** - Procedures
must be **standardized** - Priors are disputed or political - Legal or
regulatory guarantees are required

Bayesian methods shine when: - Priors are reliable - Utilities are
well-defined - Adaptive decisions matter

------------------------------------------------------------------------

## Final Takeaway

> **NHST controls how often you make mistakes.\
> Bayesian decision theory tells you what to do given what you
> believe.**

They answer different questions --- and are best seen as complementary,
not competing.
