# Practical Statistics for Data Scientists
## Chapter 1. Exploratory Data Analysis
### Estimates of location
**Definition**: To get a "typical value": an estimate of where most of the data is located. Things like *mean*, *quantile* and *outliers*
1. P3. *Key Terms for Data Types*. Taxonomy of data types.
2. P9. *Metrics and Estimates*. Nice distinction between statistician and ds
### Estimates of variability
**Definition**: measures whether the data values are tightly clustered or spread out. Things like *deviation*
1. *deviation* is $x_i - \bar{x}$,  and *standard deviation* is related to $(x_i - \bar{x})^2$.
2. *standard deviation* is much easier to interpret than *variance* since it's on same scale as the original data
3. P15 box. Why $n-1$ for variance? Because one constraint.
	1. For rigorous proof, see [[Bertsekas-Introduction-to-probability-2nd.pdf]] P467-P468

## Chapter 2. Data and Sampling Distributions
### Selection Bias
1. *vast search effect*. Abusing the data will lead to unreal patterns
	1. P54. "The difference between a phenomenon that you verify when you test a hypothesis using an experiment and a phenomenon that you discover by perusing available data..." and the example followed
	2. "Specifying a hypothesis and then collecting data following randomization and random sampling principles ensures against bias."
2. *regression to the Mean*

### Sampling Distribution of a Statistic
1. Definitions in box on P57
	1. *sample statistic*: A metric, also a random variable, calculated for a sample of data drawn from a larger population.
		1. E.g., mean of five values (individual data is *value*)
		2. "1000 means of five values", *sample size* is five; 1000 is *number of samples/experiments*
	2. *standard Error*
		1.  *standard error* measures the variability of a sample metric, *standard deviation* measures the variability of individual data points
		2. For population distribution, the error of mean is always zero; the *standard deviation* is fixed and is for infinite data points
		3. ==For *sampling distribution*, the error of mean is the *standard error* which depends on *sample size n*; one sample is as good as infinite samples in terms of estimation. Multiple samples are just for visualization.==
2. Central Limit Theorem
	1. Illustrating examples
		1. Galton Board: each collision with a peg is a Bernoulli, more levels of pegs, i.e., larger sample size, leads to Gaussian distribution of the final displacement.
		2. Sum of dice: each dice is uniform, but the distribution (*sampling distribution*) of sum of large number of dice (*sample statistic*) follows Gaussian
3. 
