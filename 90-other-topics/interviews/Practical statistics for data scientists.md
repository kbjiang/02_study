## Chapter 1. Exploratory Data Analysis
> both 'estimates of location/variability' use a single number to describe the aspects of the data.
### Estimates of location
1. **Definition**: To get a "typical value": an estimate of where most of the data is located. Things like *mean*, *quantile* and *outliers*
	1. P3. *Key Terms for Data Types*. Taxonomy of data types.
	2. P9. *Metrics and Estimates*. Nice distinction between statistician and ds
### Estimates of variability
1. **Definition**: measures whether the data values are tightly clustered or spread out. 
2. *deviation* is $x_i - \bar{x}$,  and *standard deviation* is related to $(x_i - \bar{x})^2$.
3. *standard deviation* is much easier to interpret than *variance* since it's on same scale as the original data
4. ==Sensitivity of outliers from high to low==
	1. var/std > mean absolute deviation >  median absolute deviation from the median (MAD)
	2. var/std > percentile
5. P15 box. $n-1$ degree of freedom because the sample mean is fixed.
	1. For proof, see [[Bertsekas-Introduction-to-probability-2nd.pdf]] P467-P468
### Exploring the data distribution
1. Visualization functions in `Pandas`
	1. Boxplot (1D): `whiskers` is 1.5 times the IQR (interquantile range, i.e., 0.25 ~ 0.75)
	2. Histogram (2D): `bin` vs `frequency`

### Exploring Two or More Variables
1. *Contingency table*: this relates to $\chi^2$ test

### Code
1. use `df.plot.*` for regular plots
2. Good use of `groupby`: 
	```python
	# `observed = False` to keep rows with zero `binnedPopulation`
	for group, subset in df.groupby(by="binnedPopulation", observed=False)
	```
3. `bins` in histogram
	```python
	# `bins` can be list of end points
	np.histogram(state['Murder.Rate'], bins=range(1, 12))
	# equivalently, use `pd.cut`
	# however, use `right=False` to make left edge closed (as in `numpy`)
	pd.cut(state['Murder.Rate'], bins=range(1, 12), right=False).value_counts().sort_index()
	# Confusingly, `plt.hist()` use `np` convention and is recommended for plotting histogram.
	state['Murder.Rate'].plot.hist(density=True, xlim=[0, 12], bins=range(1, 12))
	```
4. use `pd.loc`
```python
# `loc[row condition, col condition]`
telecom = sp500_px.loc[sp500_px.index >= '2012-07-01', telecomSymbols]
```
5. by category
```python
# use `ax`
# use `by` to plot by category
ax = airline_stats.boxplot(by='airline', column='pct_carrier_delay', figsize=(5, 5))
```

## Chapter 2. Data and Sampling Distributions
### Random sampling
1. Even though there are many methodologies (e.g., stratified sampling) trying to make a sample representative, they *only work properly if the data ultimately come from random sampling*. 
	1. Namely, within each group, the data you observed are like the data you didn’t observe.
2. Stratified sampling
	1. A simple random sample taken from the population would yield too few blacks and Hispanics, so those strata could be *overweighted* in stratified sampling to yield equivalent sample sizes.
	2. the overweighting makes sure the estimate in under-represented groups is accurate, but also affects overall average cross group and needs to be fixed afterwards
### Selection Bias
1. Definition: bias resulting from the way in which observations are selected.
2. Vast search effect
	1. repeated use of data which lead to something interesting *by chance*.  ==This is why we need holdout set==, e.g. train/test split.
	2. "...difference between a phenomenon that you verify when you test a hypothesis using an experiment and a phenomenon that you discover by perusing available data..." and the coin flip example followed
3. Regression to mean
	1. selecting bias toward extremely high performers
### The Bootstrap (vs CLT) #Bootstrap
1. In data science, we use bootstrap to estimate sampling distribution of a statistic, instead of CLT.
	1. Instead of calculating $\sigma/\sqrt{n}$, we resample and get the distribution/deviation of the statistic
	2. For statistics like median, the CLT for its variance is less well-defined (comparing to sample mean), while bootstrapping is straightforward
2. The random subset of training data in Random Forest is also an example of bootstrapping, i.e., bootstrap aggregating, or *bagging*.
### QQ plot
1. Given sample data, create quantiles
2. Given quantile, plot $z$-score of sample against $z$-score of standard normal distribution
	1. E.g., if heavy tail, then $z$-score of sample corresponds to quantile 0.1 will be *much more negative* than that of standard normal dist.
### The Bootstrap
1. To mimic multiple samples. See  [[#^whybootstrap]].
	1. P63, python snippet. `sample = resample(loans_income)` is same size as `loans_income` but could have duplicates, i.e., some row in `loans_income` was drawn multiple times
2. *Bootstrap* vs *Permutation*. See [[#^resampling]]
	1. see...
### Student's t-Distribution
1. See [[Bertsekas-Introduction-to-probability-2nd.pdf]] P471

## Chapter 3. Statistical Experiments and Significance Testing
### Hypothesis Tests
1. P93 box. Nice examples on *misinterpreting randomness*.
	1. "we do see the real-world equivalent of six Hs in a row... we are inclined to attribute it to something real, not just to chance."
2. Usually used to assess the output from A/B test
3. *Null hypothesis*
	1. The difference could be by change. Our hope is to prove this is not true.
### *Resampling*^resampling
1. Both *bootstrap* and *permutation*


https://youtu.be/JQc3yx0-Q9E


### Sample error and Confidence interval
1. see this [example](02_study/90-other-topics/interviews/confidence_intervals_summary.md)

### To do
1. Central Limit Theorem #CLT 
	1. Illustrating examples from 3B1B [video](https://youtu.be/zeJD6dqJ5lo)
		1. Galton Board: each collision with a peg is a Bernoulli, more levels of pegs, i.e., *larger sample size*, leads to Gaussian distribution of the final displacement.
		2. Sum of dice: each dice is uniform, but the distribution (*sampling distribution*) of sum of large number of dice (*sample statistic*) follows Gaussian