Date: 2024-05-08
Course:
Chapter: 
Lecture: 
Topic: #Probability #Bayesian #Inference 

## Lectures
### Lecture 21: Bayesian Statistical Inference - I
1. Bayesian treats the unknown as a R.V. It does *not* mean nature is uncertain, but our *prior* belief of the distribution of the unknown.
	1. ![[Pasted image 20240529073352.png|500]]
	2. Note in both cases, output $\hat{\Theta}$ is a R.V. 
2. Maximum a Posteriori Probability (MAP) estimator
	1. Defined as the R.V. $\hat{\Theta}=g(X)$. In natural language, some decision rule. 
	2. It is *optimal* for *any* given $x$ in terms of "making a correct decision" (p420 footnote), which means $\hat{\theta}=\theta_{\text{True}}$. By definition, MAP selects the $\theta$ with most probability of being $\theta_{\text{True}}$ according to posteriori dist. 
	3. This is the "greedy" one.
3. Conditional Expectation estimator
	1. it is unbiased
	2. It is *optimal* for *any* given $x$ in *terms* of minimizing mean square error.
	3. I.e., $\hat{\theta} = \mathbf{E}[\theta|X=x]$ minimizes $\mathbf{E}[(\Theta - \hat{\theta})^2|X=x]$ and similarly, $\hat{\Theta} = \mathbf{E}[\Theta|X]$ and $\mathbf{E}[(\theta - \hat{\theta})^2]$.
		1. Former is *number*, while latter is *R.V.*.
		3. In following screenshot. First point is average over $f_{\tilde{\Theta}}$, second is over $f_{\tilde{\Theta}|X}$, third is over $f_{\tilde{\Theta}}$ (where $\tilde{\Theta}\coloneqq \Theta - \hat{\Theta}$) and was obtained by using Law of Iterated Expectation.
			1. ![[Pasted image 20240617084538.png|800]]
			2. it's usually tedious to calculate $f_{\tilde{\Theta}}$, luckily we only need the expectation.
### Lecture 20: Central Limit Theorem
## My comments
1. hypothesis testing aims, few possible values for unknown, aims at small *probability* of incorrect decision; think MAP
2. Estimation: aim at a small *estimation error*; think Conditional Expectation Estimator
3. $\tilde{\Theta}, \hat{\Theta}$ are both R.V.s with their own prob. dist. Think of $Z = X + Y$.
4. posterior => how to report it? => estimator
5. A lot of averages, need to be careful about 
	1. over *which variable* it is averaged.
		1. Total variance: $\text{var}(\Theta) = \text{var}(\mathbf{E}[\Theta|X]) + \mathbf{E}[\text{var}(\Theta|X)]$, the LHS is *unconditional* and a number, RHS arguments are both functions of $X$ basing on $f_{\theta|X}$, then $f_X$.
	2. if it's a calculation of a number (conditional) or a R.V. (unconditional)
6. Why observation $X$ needs to be a R.V. as well?
	1. This is demanded by Bayesian inference. I.e., the joint distribution of $\Theta, X$. In theory, $X$ being deterministic could be a special case.
	2. Shows how $X$ is affected by $\Theta$.
## Interesting/Challenging problems and tricks
1. Example 8.3 and 8.4, P415 & 417. 
	1. The same family of posterior and prior distributions enables *recursive inference*
	2. Beta functions