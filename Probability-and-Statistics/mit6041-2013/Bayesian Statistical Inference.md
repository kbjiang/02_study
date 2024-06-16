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
2. Estimator
	1. Defined as the R.V. $\hat{\Theta}=g(X)$. In natural language, some decision rule. 
	2. Maximum a Posteriori Probability (MAP) estimator
		1. It is *optimal* for *any* given $x$ in terms of "making a correct decision" (p420 footnote), which means $\hat{\theta}=\theta_{\text{True}}$. By definition, MAP selects the $\theta$ with most probability of being $\theta_{\text{True}}$ according to posteriori dist.
		2. This is the "greedy" one.
	3. Conditional Expectation estimator
		1. It is *optimal* for *any* given $x$ in *terms* of minimizing mean square error.
		2. I.e., $\hat{\theta} = \mathbf{E}[\theta|X=x]$ minimizes $\mathbf{E}[(\Theta - \hat{\theta})^2|X=x]$ and similarly, $\hat{\Theta} = \mathbf{E}[\Theta|X]$ and $\mathbf{E}[(\theta - \hat{\theta})^2|X]$.
### Lecture 20: Central Limit Theorem
## My comments
## Interesting/Challenging problems and tricks
1. Example 8.3 and 8.4, P415 & 417. 
	1. The same family of posterior and prior distributions enables *recursive inference*
	2. Beta functions