# Curriculum Addendum: Bayesian Perspective (End of Semester 1)

## Purpose

Add a short Bayesian module after Semester 1 to strengthen probabilistic
intuition before moving into deep learning and post-training.

## Learning Objectives

- Explain priors, likelihoods, posteriors, and evidence.
- Distinguish MLE from MAP.
- Understand Bayesian updating.
- Explain Bayesian Linear Regression.
- Understand Gaussian Processes at a conceptual level.
- Understand Variational Inference and ELBO.
- Recognize Bayesian ideas in VAEs, RLHF, and LLM research.

## Part 1 --- Bayesian Thinking (Essential)

Topics - Prior - Likelihood - Posterior - Evidence - Bayesian Updating -
MAP vs. MLE - Conjugate Priors (intuition only)

Recommended Material 1. Think Bayes (Allen Downey) 2. Probabilistic
Machine Learning: An Introduction (Kevin Murphy) --- Bayesian inference,
MAP, Bayesian linear regression 3. StatQuest videos on Bayesian
Statistics, Bayesian Inference, MAP vs MLE, Bayesian Linear Regression

Homework - Write a short note comparing MLE and MAP using linear
regression.

## Part 2 --- Bayesian Machine Learning (Recommended)

Topics - Bayesian Linear Regression - Gaussian Processes (conceptual) -
Variational Inference - ELBO

Recommended Material - Murphy (selected chapters) - Stanford CS228
(Variational Inference) - MacKay: Information Theory, Inference, and
Learning Algorithms (selected chapters)

Homework - Explain Bayesian Linear Regression. - Explain why ELBO
contains a KL divergence. - Summarize advantages of Bayesian methods
over point estimates.

## Part 3 --- Modern Bayesian Deep Learning (Optional)

Topics - Bayesian Neural Networks - Monte Carlo Dropout - Laplace
Approximation - Deep Ensembles - SWAG - Uncertainty Estimation

## Connection to LLMs

Bayesian Inference → Posterior → KL Divergence → Variational Inference →
Trust Regions → PPO → DPO → RLHF

Many modern objectives can be viewed as balancing fit to new data with
staying close to prior beliefs or previous policies.

## Reflection

Write a short essay: 'How has Bayesian thinking changed the way I think
about machine learning?'
