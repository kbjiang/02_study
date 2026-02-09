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
	3. similar to $p$-hacking?
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
	1. E.g., if heavy tail, then $z$-score of sample corresponds to quantile 0.1 will be *much more negative* than that of standard normal dist. I.e., extreme values are more likely.
### Student's t-Distribution
1. See [[Bertsekas-Introduction-to-probability-2nd.pdf]] P471
2. $t$-test vs $z$-test
	1. both require the test statistic, e.g. sample mean, to be normally distributed. This means either initial population is normal, or sample size is large ($n \ge 30$) so that CLT can be applied. 
	2. Use $t$-test instead of $z$-test when (a) number of data is small and (b) unknown population variance.
### ==Chi-Squared distribution vs F-distribution==
1. Chi-Squared is typically concerned with *counts* of items falling into category
	1. Probability is just for calculating expected counts, never directly in $\chi^2$ statistic.
2. F is similar except that it deals with *measured continuous* values rather than counts
	1. *Continuous* does NOT mean float, can be integers like level of pain.

## Chapter 3. Statistical Experiments and Significance Testing
> All these tests, $z$, $t$, ANOVA and $\chi^2$ are unified under NHST, in the sense all of their distributions, normal, $t$, $F$ and $\chi^2$, are derived under null hypothesis of ineffectiveness of the treatment, i.e., pure chance.
### Hypothesis Tests
1. P93 box. Nice examples on *misinterpreting randomness*.
	1. "we do see the real-world equivalent of six Hs in a row... we are inclined to attribute it to something real, not just to chance."
2. Usually used to assess the output from A/B test
### ==Resampling==
1. Goal: *to assess random variability in a statistic*
	1. Two main types: *bootstrap* and *permutation*
2. Bootstrap
	1. Mostly for assessing the reliability (e.g. confidence interval) of an estimate, i.e., empirical distribution instead of relying on CLT
	2. Always resamples with replacement
3. Permutation
	1. Mostly for testing hypotheses, typically involves ==two or more groups==
	2. Usually resamples without replacement
	3. ==It can be the empirical alternative to many tests, such as $t$, $\chi^2$ and ANOVA.==
		1. See both examples in this section, one continuous (web stickiness), the other counts (ecommerce conversion)
4. ==When sample size is low, statistical theory approximation becomes rough, the resampling is preferred.==
### $t$-tests
1. ==Conceptual==: I now prefer "==$t$-distribution is a reference distribution to which the observed $t$-statistic can be compared==" to "the $t$-statistic follows a $t$-distribution", because it's closer to the procedure
	1. "Studentize" data, i.e., test statistic => compare against *predefined* $t$-distribution
2. ==Many variants==
	1. One-sample $t$-test for the mean: measures the variability of sample mean
	2. two-sample $t$-test for comparing means: 
		1. $H_0$ is $\bar{x}-\bar{y}=\Delta\mu$
		2. DOF is $n+m-2$ coz both sample means are used for calculating sample variances
	3. Paired two-sample $t$-test.
### Multiple testing
1. Multiplicity (multiple comparisons, many variables etc) increases the risk of concluding that something is significant by chance
2. the $p$-value needs to be adjusted when multiple comparisons
3. use holdout samples (train/test split)
### Good examples
1. "web stickiness" examples
	1. Has code for permutation
	2. The A/B version can be tested by either *two-sample* $t$-test of unequal variances or ANOVA
	3. The A/B/C/D version can only be tested by ANOVA because it has more than two groups
### ANOVA
1. ANOVA (Analysis on variances)
	1. More than two groups, alternative to $t$-test.
2. Big idea
	1. A *single overall* test (Omnibus test) that addresses "could all groups have the same value for the statistic of interest? could the ==difference among groups== due to chance?"  
		1. In $t$-test, we quantify the difference by $\bar{x}-\bar{y}$ since only two groups
		2. In ANOVA, we quantify the difference among groups with the variance of the means
			1. This is also the quantity simulated in permutation test
	2. Avoids the multiple pair-wise comparison, which has the risk of p-hacking
3. $\chi^2$ is similar in logic
4. "You can see that ANOVA and then two-way ANOVA are the first steps on the road toward a full statistical model, such as regression and logistic regression, in which multiple factors and their effects can be modeled"
5. Q: is ANOVA symmetric in $n$ and $m$, i.e., does it care how to group the data? How about $\chi^2$? See 18.05 Class 19 4.4
### Chi-square test
1. Always calculate expected counts, do NOT use probability!
2. `stats.chi2_contigency` for independence/homogeneity test, i.e., multiple rows of data
3. `stats.chisquare` for "goodness-of-fit", i.e., one row of data.
### Multi-arm Bandit algorithm
1. NOT about test significance, but to reach conclusions faster than traditional statistical designs
	1. Notice when 3+ treatments present, neither ANOVA nor $\chi^2$ reveals which treatment is the best, rather just if they have the same mean.
2. "imagine a slot machine with more than one arm, each arm paying out at a different rate, you would have a multi-armed bandit, which is the full name for this algorithm."
### Code
1. "web stickiness" example. Mimic distribution of "Mean diff"
	```python
	# `equal_var` can be either True or False; DOF is `n+m-2`.
	# `res.pvalue` counts for both sides, therefore `res.pvalue/2` to get single sided
	res = stats.ttest_ind(
	    session_times[session_times.Page == 'Page A'].Time,
	    session_times[session_times.Page == 'Page B'].Time,
	    equal_var=False)
	print(f'p-value for single sided test: {res.pvalue / 2:.4f}')
	```
2. "ecommerce conversion" example. The out-of-box $p$-value was divided by two to make it one-sided, so that it can be compared with permutation calculation `np.mean([diff > obs_pct_diff for diff in perm_diffs])`.
	1. The out-of-box $p$-value is the prob. of two groups being different, in either direction.
3. `df.pivot`. 
	```python
	# example from section "Chi-Square Test: A Resampling Approach"
	click_rate.pivot_table(index="Click", columns="Headline", values="Rate")
	# Headline  Headline A  Headline B  Headline C
	# Click                                       
	# Click             14           8          12
	# No-click         986         992         988
	```
4. code up example from section "Chi-Square Test: A Resampling Approach"

## 4. Regression and prediction
### Simple linear regression
1. least square is sensitive to outliers
2. ==regression equation does not prove the direction of causation==
3. metrics
	1. `mean_squared_error`, `r2_score` (RSS)
### Simple linear regression (Rice 14.2)
1. $y_i = \beta_0 + \beta_1 x_i + e_i$, here $x_i$ are numbers, estimator of $\beta$'s, i.e., $\hat{\beta}$'s are R.V.s. Therefore, they will have variances and confidence intervals
	1. Null hypothesis: $\beta$ is zero.
	2. $\sigma$ is the ==variance of $e_i$==
2. estimates and their variance
	1. When $\sigma$ is known $$
		\begin{aligned}
		\hat{\beta}_1 &=\frac{\sum_{i=1}^n (x_i - \bar{x})y_i}{(x_i - \bar{x})^2} \\
		\text{Var}(\hat{\beta}_1) &=\frac{\sigma^2}{\sum_{i=1}^n(x_i - \bar{x})^2} \text{, and compares against normal distribution}\\
		\end{aligned}$$
	2. When $\sigma$ is unknown, variance of $e_i$ needs to be estimated via RSS. $$
		\begin{aligned}
		\text{RSS} &= \sum_{i=1}^n e_i^2 = \sum_{i=1}^n (y_i - \hat{\beta}_0 - \hat{\beta}_1 x_i)^2 \\
		s^2 &= \frac{\text{RSS}}{n-2}, \text{ estimated variance, $n-2$ DOF because two parameters estimated} \\
		s_{\hat{\beta_1}}^2 &=\frac{s^2}{\sum_{i=1}^n(x_i - \bar{x})^2}, \text{ replace variance with sample error} \\
		\frac{\hat{\beta_i}-\beta_i}{s_{\hat{\beta_1}}}	& \sim t_{n-2}, \text{ $\beta_i=0$ when null hypothesis means no linear relation between $x_i$'s and $y$.}  
		\end{aligned}$$
	3. $t$-statistic is asking "is this ONE predictor useful in predicting the target", while $F$-statistics asks "is there ANY predictor useful in predicting the target". 
		1. Check out [[temp-ols-dof]] for more. 
		2. The duality of viewing data
			1. Row Space (Each observation is a point); for clustering, PCA, visualization etc
			2. Column Space (one vector in **n-dimensional observation space**); for solving OLS, projection, understanding DOF etc.
				1. ==Duality between data and feature==: E.g., the interception has a measurement of vector **1**
				2. ??? (still missing a mental picture on beta) ==Relation to underlying distribution==: E.g., the interception has a delta distribution and always observes the same 1; similarly, each true $\beta_i$ dictates the observation $y_j$ given $x_{ij}$, where $\hat{\beta}_i$ will be inferred!
			3. see [[temp-ols-dof#^geometric-view]] 
### Confidence and Prediction intervals
1. Confidence intervals:
	1. Variability about coefficients; calculated from all data points
2. Prediction:
	1. about single prediction; greater than CI
	2. With infinite data and a correctly specified model, confidence intervals shrink to zero because parameter uncertainty vanishes. Prediction intervals remain wide because they include irreducible noise from individual outcomes ($e_i$).
### Factor variables in Regression
1. For linear regression, convert to dummy variables
	1.  `drop_first` to keep `k-1` types to avoid multicollinearity
		1. when all are zero, indicates kth type
	2. This also uses `multiplex` as reference
		1. assuming coefficient for `townhouse` is `-50000`, this means that, while everything else the same, a `townhouse` is $50$k cheaper than a `mltuplex`. 
	3. ==`get_dummies` only converts dtypes `bool`, `object` and `category`. Convert dtypes when necessary.==
		```python
		house["PropertyType"].value_counts()
		# PropertyType
		# Single Family    20720
		# Townhouse         1710
		# Multiplex          257
		# Name: count, dtype: int64
		
		# Encoded to multiple dummies
		# `drop_first`: keep `k-1` types, when all are zero, indicates kth type
		X = pd.get_dummies(house["PropertyType"], drop_first=True, dtype=int)
		# PropertyType_Single Family  PropertyType_Townhouse
		# 1                           0                       0  # this is Multiplex
		# 2                           1                       0
		# 3                           1                       0
		```
	4. `dtype`:  before `get_dummies`, `PropertyType` is `object`, i.e. string; after `PropertyType_Townhouse` is `bool`
### Factor Variables with Many Levels
1. `dtype`: before `astype('category')`, `ZipGroup` is `int`; after it becomes `category`
	1. Signals "this is a categorical variable, not a number"
	2. `category` features will need to be `get_dummies` when fitting model; `int` feature won't work with `get_dummies`
2. ==Nice coding example==
	```python
	# skipped code on grouping by `zipcode` and get counts by `zipcode`
	# `groupby` then count makes sure same `ZipCode` falls under same `ZipGroup`
	# needs `cum_count` for quantiling; think of CDF
	zip_groups['cum_count'] = np.cumsum(zip_groups['count'])
	# `labels=False` to return integer labels
	# `retbins=False` to not return bin edges
	zip_groups['ZipGroup'] = pd.qcut(zip_groups['cum_count'], q=5, labels=False,
	                                 retbins=False)
	
	# skipped...
	house['ZipGroup'] = house['ZipGroup'].astype('category')
	```
### Ordered Factor Variables
1. Where values reflect different levels
2. Can be converted to numerical values and used as is
### Interpreting the regression equation
1. Correlated predictors
	1. Coefficient for "Bedrooms" is positive when fitted alone, negative when fitted with other predictors. This is because the predictor variables are correlated: between two homes of the exact same size, the one with more but smaller bedrooms would be considered less desirable.
2. Multicollinearity
	1. Multicollinearity in regression must be addressed—variables should be removed until the multicollinearity is gone. See [[temp-multicollinearity_cheat_sheet#^9c9059]]
3. Confounding variables
	1. An important predictor missing from model and can lead to spurious relationships
		1. E.g., has variables "being fit" and "has a gym membership", but missing "goes to gym daily"
4. Interaction term between two variables
	1. Need to include this interaction if interdependent with response
### Regression Diagnostics
1. Heteroskedasticity
	1. Prediction errors differ for different ranges of the predicted value, and may suggest an incomplete model, i.e., more variables required
2. ==Partial residual==
	1. Isolate the relationship between predictor $X_i$ and the target, *taking into account all of the other predictors*
		1. $\text{Partial residual = Residual } + \hat{b}_iX_i$, where Residual is $y - \sum \hat{b}_jX_j$.
		2. Apparently this is better than just plot $X_i$ against $y$, where none of the other predictors are considered.
### Code
1. Linear regression with statistical info
```python
import statsmodels.api as sm

X_with_const = sm.add_constant(X)  # by default, no interception
model = sm.OLS(y, X_with_const).fit()

print(model.pvalues)      # p-values
print(model.bse)          # standard errors
print(model.summary())    # full summary table
```
2. `pd.cut` vs `pd.qcut`
	1. `cut` bins data to equal width intervals, therefore bins may have unequal counts
	2. `qcut` bins data to equal count intervals, therefore bins have roughly equal counts
		```python
		# basic example
		data = [1, 2, 3, 4, 5, 100]
		pd.cut(data, 3)   # Bins: (1, 34], (34, 67], (67, 100] — unequal counts
		pd.qcut(data, 3)  # Bins: (0.9, 2.67], (2.67, 4.33], (4.33, 100] — ~2 items each
		```
3. Weighted regression
	1. inverse-variance: variables with higher variances should have lower weights
	2. by importance: e.g., recent events carry more weights
	```python
	house['Weight'] = house.Year - 2005
	house_wt.fit(house[predictors], house[outcome], sample_weight=house.Weight)
	```


### To do
1. Central Limit Theorem #CLT 
	1. Illustrating examples from 3B1B [video](https://youtu.be/zeJD6dqJ5lo)
		1. Galton Board: each collision with a peg is a Bernoulli, more levels of pegs, i.e., *larger sample size*, leads to Gaussian distribution of the final displacement.
		2. Sum of dice: each dice is uniform, but the distribution (*sampling distribution*) of sum of large number of dice (*sample statistic*) follows Gaussian