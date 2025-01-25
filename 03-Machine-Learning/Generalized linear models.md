## Three components of a GLM
1. Systematic component
	1. Basically the linear part, i.e., the $L$ in GLM. 
		1. E.g., $z = b_0 + b_1 x$. Here $x$ is the independent variable and $z$ is the indicator(?)
2. Link function
	1. It projects $z$ to $y$. This is decided by the trend of the data, could be *independent of distribution of residual*.
		1. E.g., For ordinary LM,  the link function is the identity function 
		2. E.g., in Poisson Regression, the default link function is $y={e^z}$, we use exponential function to project $z$ $(-\infty, \infty)$ to $(0, \infty)$.
			1. We usually use Poisson Regression to model counts, which is usually exponential in nature
3. Distribution of the residual
	1. This is where the probability models come in to play.
		1. E.g., for logistic regression, the residual is either $0$ or $1$, then we use a binomial distribution.
		2. E.g., for residuals that have comparable mean and variance, we use Poisson distribution ![[Pasted image 20250120180718.png|600]]
	2. The estimator function $\hat{y}(x)$ models the expectation, the probability models tracks the residual $e = y - \hat{y}$.
		1. for every point $x$, we have a distribution; could be i.i.d (normal) or not (Poisson).
4. MISC
	1. Summary table ![[Pasted image 20250120163435.png|600]]
	2. Ref: https://youtu.be/Obpz_Uvo2rQ?list=PLLTSM0eKjC2cYVUoex9WZpTEYyvw5buRc


## Logistic regression
1. [StatQuest: Logistic Regression](https://www.youtube.com/@statquest)

## Reference
1. Very good lectures on statistics: https://www.tilestats.com/
2. Python Bayesian model-building interface https://bambinos.github.io/bambi/