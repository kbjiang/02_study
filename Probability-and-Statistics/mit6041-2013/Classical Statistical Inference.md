Date: 2024-07-17
Course:
Chapter: 
Lecture: 
Topic: #Probability #Inference 

## Lectures
### Lecture 23-25: Bayesian Statistical Inference
1. $p(X;\theta)$ to emphasize that this is not a conditional distribution, since $\theta$ is not an R.V.
	1. Imagine a collection of  probabilistic models, each with a different $\theta$, and we are trying to guess which one is the true one.
	2. $\mathbf{E}_{\theta}[\hat{\theta}]=\int f_{\hat{\theta}}(\hat{\theta}; \theta) \hat{\theta} d\hat{\theta}$. Here $\hat{\theta}$ is an R.V. and depends on $\theta$ via observations $X$; $f_{\hat{\theta}}$ apparent depends on value of $\theta$.
2. Maximum likelihood Estimation
	1. *Equivalent to MAP* with uniform flat prior dist.
3. Confidence interval
	1. Interpretation is subtle. The intervals are the R.V.s, not $\theta$. 
	2. Instead of "the true parameter lies in the confidence interval with probability 0.95.", it should be "For a concrete interpretation, suppose that $\theta$ is fixed. *We construct a confidence interval many times*, using the same statistical procedure, i.e., each time, we obtain an independent collection of $n$ observations and construct the corresponding $95\%$ confidence interval. We then expect that about $95\%$ of these confidence intervals will include $n$. This should be true regardless of what the value of $\theta$ is." (bottom of p469)
	3. Two approximations involved
		1. sample mean has a normal dist. (CLT)
		2. approximated variance
## My comments
## Interesting/Challenging problems and tricks
