1. ML from a statistics perspective
	1. https://ocw.mit.edu/courses/18-657-mathematics-of-machine-learning-fall-2015/
2. Statistics for applications
	1. https://ocw.mit.edu/courses/18-650-statistics-for-applications-fall-2016/
		1. https://github.com/hoangnguyen7699/StatisticsForApplication_solution
	2. https://ocw.mit.edu/courses/18-443-statistics-for-applications-spring-2015/


## Statistics for applications Fall 2016
### Lecture 1
1. Probability vs Statistics, both stem from *randomness*.
	1. Probability: 
		1. simple random process where the distribution of r.v. (random variable) is clear, e.g., rolling dice
	2. Statistics: 
		1. more complicated process; need to estimate parameters from data
		2. statistical modeling = simple process (what we know) + random noise (lack of info, randomness etc.)
	3. They both study randomness, but under different conditions.
2. *Laws of large numbers* and *Central limit theorem*.
	1. LLN guarantees *sample average is a good estimator of expectation*. Probability and almost surely. (a.s.)
	2. CLT quantifies how good this estimate is.
3. Different types of convergence for a sequence of r.v. ($T_n$)
	1. different r.vs. can have same distribution. E.g., $X \sim \text{Ber(0.5)}$ and $Y=1-X$. Both $X$ and $Y$ have same distribution but they are never equal. [Ref](https://www.quora.com/Is-it-true-that-two-random-variables-are-the-same-if-and-only-if-they-have-the-same-probability-distribution)
	2. Same example shows that convergence in distribution does not lead to convergence in probability. [Ref](https://math.stackexchange.com/questions/1348177/example-of-convergence-in-distribution-but-not-in-probability)

### Lecture 2
1. The rational behind statistical modeling. 
	1. The goal is to learn distribution of $X$
	2. Purpose of modeling is to restrict the space of possible distributions s.t. it's plausible and easier to estimate.
		1. e.g., the number of sibling problem. Poisson (1 parameter) is better than tracking all seven individually.
2. the formalization of statistical modeling
	1. $X = \mathbb{1}(Y>0)$ and $Y \~ \mathbb{N}(\mu, \sigma)$ so ${{0, 1}, {\mu, \sigma}}$.

### Lecture 3
### Lecture 4
1. In the definition of Confidence interval, keep in mind that C.I. is also a r.v., otherwise the probability greater than $1-\alpha$ won't make sense.
