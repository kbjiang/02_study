## Introduction
1. RL is much more focused on goal-directed learning from interaction than are other approaches to machine learning.
2. RL: *trial-and-error* search and *delayed reward*
3. How is RL different?
	1. vs supervised ML: no labels available in uncharted territory. "..the essential character of trial-and-error learning as selecting actions on the basis of evaluative feedback that does not rely on knowledge of what the correct action should be."
	2. vs unsupervised ML: aims to maximize reward signal instead of learning hidden structure.
	3. RL uses training info that *evaluates* actions taken rather than *instructs* by giving correct actions. In ML, instructive feedback is *independent* of action taken.
 4. Rewards vs values
	 1. rewards: immediate, easy to get; values: long-term, hard to eval
	 2. "To make a human analogy, rewards are somewhat like pleasure (if high) and pain (if low), whereas values correspond to a more refined and farsighted judgment of how pleased or displeased we are that our environment is in a particular state."--sec 1.3
	 3. *"In fact, the most important component of almost all reinforcement learning algorithms we consider is a method for eciently estimating values"*--sec 1.3
	 4.  "...Methods for solving reinforcement learning problems that use models and planning are called model-based methods, as opposed to simpler model-free methods that are explicitly trial-and-error learners—viewed as almost the opposite of planning...Modern reinforcement learning spans the spectrum from low-level, trial-and-error learning to high-level, deliberative planning."--sec 1.3
5. Fig 1.1. Move leads to $e$ is an exploratory move, which does not result in any learning (therefore no red arrow), since $e$ and its value is not in the value table yet.

## Part 1
> 1. Deal with *finite* states and actions where value functions can be represented as a table
> 2. Three methods and their combinations: dynamic programming, Monte Carlo and temporal-difference learning

### Chapter 2
1. *Evaluative vs instructive* feedbacks
2.  Sample definition/estimation of action values
$$
\begin{align*}
	q_*(a) &\doteq \mathbb{E}(R_t | A_t = a) \qquad &&\text{action value; independent of t} \\
	Q_t(a) &= \frac{\sum^{t-1}_{i=1}R_i \cdot \mathbb{1}_{A_i=a}}{\sum^{t-1}_{i=1}\mathbb{1}_{A_i=a}} && \text{estimated; approaches true value when t is infinity}
\end{align*}
$$
3. stationary with $1/n$ step size $\rightarrow$ non-stationary with constant step size.

1. 