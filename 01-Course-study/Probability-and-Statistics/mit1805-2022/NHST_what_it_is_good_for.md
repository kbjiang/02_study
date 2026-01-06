# What NHST Is Good For (and What It Is Not)

## Short answer

**NHST is useful because it gives a guarantee about long-run error rates
of a decision procedure --- not because it tells you how likely a
hypothesis is true.**

It is a **tool for controlling mistakes**, not for quantifying belief.

------------------------------------------------------------------------

## What NHST actually gives you (and why that's valuable)

NHST answers this question:

> *If I follow this decision rule repeatedly, how often will I make a
> wrong rejection?*

Formally:

$P(\text{reject } H_0 \mid H_0 \text{ is true}) \le \alpha$

This is extremely useful when: - Decisions are **irreversible or
costly** - You need **procedural guarantees** - You care about **false
alarms**

Examples: - Approving a drug - Shipping a model change - Declaring a
scientific effect - Triggering an expensive follow-up experiment

In all these cases, you may not care *how true* the null is --- you care
about **not falsely claiming an effect too often**.

------------------------------------------------------------------------

## What NHST deliberately refuses to do

NHST does **not** attempt to answer: - "How likely is the hypothesis
true?" - "What should I believe?"

Because answering those requires: - Priors - Loss functions - Subjective
inputs

NHST says:

> "I won't tell you what to believe --- I'll tell you how to behave."

------------------------------------------------------------------------

## Why this is not as weak as it sounds

Think of NHST like **quality control**.

A factory might say:

> "Our process produces defective items less than 1% of the time."

That does *not* tell you: - Whether today's item is defective - The
probability *this item* is defective

But it **guarantees the system is reliable over time**.

NHST plays the same role in scientific decision-making.

------------------------------------------------------------------------

## Why p-values feel unsatisfying (and you're right to feel that)

Your discomfort is justified because: - People **interpret p-values as
belief** (which they are not) - NHST is often used **alone**, without
effect sizes - It encourages **binary thinking**

That's why modern best practice is: - NHST **plus** - Effect sizes -
Confidence intervals - Domain reasoning

------------------------------------------------------------------------

## What NHST is good at (summary)

NHST is good when you want to: - Control **false discoveries** - Compare
**procedures** - Make **rules that scale** - Ensure **replicability
discipline**

It is *not* good for: - Belief updating - Measuring evidence strength
alone - Decision-making without context

------------------------------------------------------------------------

## If you want "how likely is it true", use something else

If your real question is:

> "How confident should I be that there is a real effect?"

Then better tools are: - Bayesian posterior probabilities - Bayes
factors - Likelihood ratios - Predictive checks

These answer **different questions**.

------------------------------------------------------------------------

## One-sentence takeaway

> **NHST is a rule for controlling how often you fool yourself --- not a
> calculator for how true something is.**
