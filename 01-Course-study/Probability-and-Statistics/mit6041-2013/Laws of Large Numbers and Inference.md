Date: 2024-05-08
Course:
Chapter: 
Lecture: 
Topic: #Probability #LawsOfLargeNumber

## Lectures
### Lecture 19: Weak Law of Large Numbers
1. It considers *long sequence* of identical independent R.V.s. 
	1. Sample mean $M_n$ is average of $n$ samples from one sampling process (one experiment)
		1. sample size $n$ is what matters. See 3B1B part of [[#My comments]]
	2. Expectation $\mathbf{E}[M_n]$ is average of infinite $M_n$
	3. But we usually concerns with $\mathbf{P}(M_n)$ where $M_n$ is a single experiment and ask the likelihood of it being in some interval?
2. Markov/Chebyshev Inequality
	1. links expectation/expectation and variance with probability
		1. Useful when mean and variance are easy to compute but dist. is hard to get
	2. valid for *ANY* R.V.s.
3. The Weak Law of Large Numbers
	1. Is an application of Chebyshev. *Only* deals with sample mean R.V.
4. Convergence in Probability however can be about any R.V.
	1. $Y_n$ converges in probability $\nRightarrow$ $\mathbf{E}[Y_n]$ converges too. See Example 5.8, page 272.
5. The example of Y_n = {0, n} at lec. vid. 23:00
6. intuition leads to CLT around 40:30

### Lecture 20: Central Limit Theorem #CLT
1. First read [[Practical statistics for data scientists#Sampling Distribution of a Statistic CLT]] 
	1. to grasp the concept of *sample statistic* and *sampling distribution*.
	2. to build the intuition of CLT
2. *Only* requires mean and variance!
3. For $X$ that are *discrete distribution*, the CDF always converges to normal, but PMF may differ between different $X$ dist.
	1. For Binomial, PMF can be approximated with "half" approximation
## My comments
1. Chebyshev is for any R.V., while Weak LLN is specific for sample mean R.V.
2. Other intuitions in 3B1B videos
	1. This [time stamp](https://youtu.be/zeJD6dqJ5lo?t=1373) is an lopsided example where sample size is too small to apply CLT. 
	2. This [time stamp](https://youtu.be/IaSGqQa5O-M?t=658) is more on convolution and shows the sum of R.V.s are like moving average which gets more smooth with more R.V.s.
## Interesting/Challenging problems and tricks
1. 