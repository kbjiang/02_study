1. still not comfortable with Poisson process and Markov chain
2. covariance/correlation
	1. Regression to mean: best understood to think of $X$ and $Y$ as first and second measurement--unless perfect correlation, $\mathbb{E}[Y|X]$ tends to be less extreme than $X$.
	2. see [[temp-regression-to-mean]]
3. QQ plot?
	1. [Quantile-Quantile Plots (QQ plots), Clearly Explained!!!](https://www.youtube.com/@statquest)
4. Permutation test as a modern alternative to $t$-test and $f$-test? [3. Statistical Experiments and Significance Testing | Practical Statistics for Data Scientists, 2nd Edition](https://learning.oreilly.com/library/view/practical-statistics-for/9781492072935/ch03.html#Permutation)
	1. it does not assume data is normally distributed
	2. it permutes data many times and look at the distribution of resulting statistic distribution to figure out $p$-value.
		1. Conceptual: ==Chi-sq distribution and F-distribution are theoretical approximation of $\chi^2$/f statistics!== For e.g., for permutation test, you still calculate $\chi^2$ statistic and see its distribution via resampling. But no Chi-sq test/distribution is involved!
5. ANOVA vs Chi-sq
	1. both can deal with multiple groups?
	2. ANOVA is on value
	3. Chi-sq is on counts/distributions? Class 19, Section 4.5, 4.6
		1. Goodness of fit has one row of data, because the underlying dist. is given
		2. Test of homogeneity has more rows because it's between multiple sets of data
		3. Either way, the degree of freedom is calculated on ==number of observed data points==!