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
### Lecture 24: Classical Statistical Inference II
1. Linear Regression
	1. It is NOT probabilistic, but can have probabilistic explanations. See *Justification of the Least Squares Formulation* on p478 and *Bayesian Linear Regression* on p480.
	2. *Practical Considerations* p484
		1. *Heteroskedasticity*. Noise $W_i$ is dependent on value of $x_i$.
		2. *Multicollinearity*. Say $y=\theta_0 + \theta_1 x_1 + \theta_2 x_2$ and $x_1, x_2$ are highly correlated, then $\theta_{1,2}$ will become very sensitive.
2. Binary Hypothesis Testing
	1. The goal is to find a rejection region $R$ to minimize the overall probability of error (Type I & Type II)
		1. False rejection probability (Type I): $P(L(X)>\xi; H_0)$; False acceptance probability (Type II): $P(L(X)\le\xi; H_1)$.
		2. Given a target false rejection prob $\alpha$ and some test/*statistic* $h(x)$, one infer the threshold $\xi$ by solving $\mathbf{P}(h(X)>\xi; H_0) = \alpha$, which usually involves going from $P(h(X))$ to $P(X)$, which leads to rejection region $R=\{x|h(x)>\xi\}$. 
		3. To find $R$, one may use different $h(X)$; See E.g. 9.13 p493.
		4. ![[Pasted image 20240809074815.png|600]]
	2. Likelihood Ration Test (LRT)
		1. Likelihood Ratio $L(X)=\frac{p_X(x; H_1)}{p_X(x; H_0)}$.
		2.  Basically, greater $\xi$ favors $H_0$, i.e., $H_1$ needs to be a lot more likely to be considered.
	3. About $\xi$
		1. when $\xi=1$ we treat two hypothesis equally; basically doing Maximum Likelihood at every $x$.
		2. when $\xi$ takes extreme values, prob. of error can be 0 or 1. Think ROC.
			1. ![[Pasted image 20240809075850.png|300]]
3. Significance Testing
	1. Subtlety of *$H_0$ is rejected*...
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
2. E.g. 9.10, p487