# Variance Reduction Example in REINFORCE

This document walks through a complete numeric example showing why the
vanilla policy gradient estimator has high variance and how a simple
baseline reduces that variance.

## Problem Setup

-   One-step episodic RL problem.
-   Bernoulli policy with probability ( p = 0.6 ) of taking action 1.
-   Rewards:
    -   If action 1: ( G = 1 ) with probability 0.8, else 0.
    -   If action 0: ( G = 1 ) with probability 0.2, else 0.

For this parameterization:

\[

abla_p `\log `{=tex}`\pi`{=tex}(a{=}1) = rac{1}{p}, `\quad `{=tex}

abla_p `\log `{=tex}`\pi`{=tex}(a{=}0) = -rac{1}{1-p}. \]

With ( p = 0.6 ):

\[ g_1 = 1.6666`\ldots`{=tex}, `\quad `{=tex}g_0 = -2.5. \]

The gradient estimator sample is:

\[

abla_p `\log `{=tex}`\pi`{=tex}(a), G. \]

## 1. True Expected Gradient

\[ `\mathbb{E}`{=tex}\[ abla `\log`{=tex}`\pi`{=tex}(a),G\] = p g_1
`\mathbb{E}`{=tex}\[G\|a=1\] + (1-p) g_0 `\mathbb{E}`{=tex}\[G\|a=0\].
\]

With: - ( `\mathbb{E}`{=tex}\[G\|a=1\] = 0.8 ) - (
`\mathbb{E}`{=tex}\[G\|a=0\] = 0.2 )

We get:

\[ 0.8 + (-0.2) = 0.6. \]

------------------------------------------------------------------------

## 2. Variance Without a Baseline

We compute:

\[ `\operatorname{Var}`{=tex} = `\mathbb{E}`{=tex}\[(
abla`\log`{=tex}`\pi`{=tex}`\cdot `{=tex}G)\^2\] - (0.6)\^2. \]

Since (G `\in `{=tex}{0,1}), (G\^2 = G).

Final result:

\[ `\operatorname{Var}`{=tex}\_{ ext{no baseline}} pprox 1.4733. \]

------------------------------------------------------------------------

## 3. Optimal Constant Baseline

The best constant baseline is:

\[ b\^`\star `{=tex}= rac{`\mathbb{E}`{=tex}\[
abla`\log`{=tex}`\pi`{=tex}(a)\^2 G\]} {`\mathbb{E}`{=tex}\[
abla`\log`{=tex}`\pi`{=tex}(a)\^2\]}. \]

Numerically:

\[ b\^`\star `{=tex}pprox 0.44. \]

------------------------------------------------------------------------

## 4. Variance With the Optimal Baseline

We compute:

\[ `\operatorname{Var}`{=tex}\_{b\^`\star`{=tex}} pprox 0.6667. \]

This is a **55% reduction** in variance.

------------------------------------------------------------------------

## Summary

  Case                        Variance
  --------------------------- ----------
  No baseline                 \~1.4733
  Optimal constant baseline   \~0.6667

This example illustrates the core idea behind variance reduction in
policy gradient methods such as REINFORCE: subtracting a baseline lowers
variance without changing the expected gradient.
