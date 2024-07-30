Date: 2024-07-17
Course:
Chapter: 
Lecture: 
Topic: #Probability #Inference 

## Lectures
### Lecture 23: Classical Statistical Inference
1. Unknow parameters $\theta$ as constants. 
	1. Observation $X$ is random and its distribution $p_X(x;\theta)$ depends on the value of $\theta$. ";" is to distinguish from conditional distribution, since $\theta$ is *not* an R.V.
	2. A separate probabilistic model for each possible value of $\theta$; a "good" hypothesis testing or estimation procedure will be one possesses certain desirable properties *under every candidate model*.
	3. $\mathbf{E}_{\theta}[\hat{\theta}]=\int f_{\hat{\theta}}(\hat{\theta}; \theta) \hat{\theta} d\hat{\theta}$. Here $\hat{\theta}$ is an R.V. and depends on $\theta$ via observations $X$; $f_{\hat{\theta}}$ depends on value of $\theta$. See example on notation [[#^fd8ec2|here]].
	4. Useful terminology.
		1. ![[Pasted image 20240724071345.png|800]]
		2. Highlighted: $\mathbf{E}_{\theta}[\hat{\theta}_n]$ is population mean. Think of many $n$ sample experiments therefore independent of $X$. See [[#^ffe4fa|here]] for an example. 
2. Maximum likelihood Estimation
	1. *Equivalent to MAP* with uniform flat prior dist.
3. Confidence interval
	1. Interpretation is subtle. The intervals, $\hat{\Theta}_n \pm \delta$, are the R.V.s, not $\theta$. The intervals are supposed to contain $\theta$ with a certain high probability, for every possible value if it.
	2. Instead of "the true parameter lies in the confidence interval with probability 0.95.", it should be "For a concrete interpretation, suppose that $\theta$ is fixed. *We construct a confidence interval many times*, using the same statistical procedure, i.e., each time, we obtain an independent collection of $n$ observations and construct the corresponding $95\%$ confidence interval. We then expect that about $95\%$ of these confidence intervals will include $n$. This should be true regardless of what the value of $\theta$ is." (bottom of p469)
	3. Two possible approximations
		1. sample mean has a normal dist. (CLT)
		2. variance approximated (is true variance is not known)
	4. Student's distribution ($t$-distribution) for small $n$ ($n\le 50$) 
### Lecture 24: Classical Statistical Inference
## My comments
1. Do NOT confuse *variance of an estimator* with *an estimator of variance*.
	1. p466. $M_n=\frac{X_1 + ... + X_n}{n}$ is the estimator of true/population mean, which is also an R.V., which has its own variance.
	2. p467. $\bar{S}_n^2 = \frac{1}{n}\sum_{i=1}{n}(X_i - M_n)^2$ is the estimator of true/population variance.
## Interesting/Challenging problems and tricks
1. Tutorial 11, problem 2, (b) & (c)
	1. *Notation*: $\hat{p}_n = \frac{1}{3n}\sum_{i=1}^{n}k_i$ vs $\hat{P}_n = \frac{1}{3n}\sum_{i=1}^{n}X_i$. ^fd8ec2
	2. how to show unbiased. $$
		\begin{align}
		\mathbf{E}_p[\hat{P}_n] = &\ \frac{1}{3n}\sum_{i=1}^n\mathbf{E}_p[X_i] \\
		=&\ \frac{1}{3n}\sum_{i=1}^n 3p \\
		=&\ p
		\end{align}
		$$ ^ffe4fa