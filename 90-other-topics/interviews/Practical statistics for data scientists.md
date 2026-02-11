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
			1. Row Space (Feature being axes); for clustering, PCA,  visualization etc
				1. basically the redundancy of the data provides info about feature
			2. Column Space (Data/realizations being axes); for solving OLS, projection, understanding DOF etc.
			3. ==Duality between data and feature==: data is the realizations of the feature. Therefore, data reflects the statistical nature/model of the feature. E.g., heights and weights of a group of men is correlated, and their covariance (data) can be leveraged. 
			4. see [[temp-ols-dof#^geometric-view]] 
### Confidence and Prediction intervals
1. Confidence intervals:
	1. Variability about coefficients; calculated from all data points
2. Prediction:
	1. about single prediction; greater than CI
	2. With infinite data and a correctly specified model, confidence intervals shrink to zero because parameter uncertainty vanishes. Prediction intervals remain wide because they include irreducible noise from individual outcomes ($e_i$).
### ==Factor variables in Regression==
1. For linear/logistic regression, convert to dummy variables
	1.  `drop_first` to keep `k-1` types to avoid multicollinearity
		1. when all are zero, indicates kth type
		2. The dummies without dropping is multicollinear with intercept: intercept column ($x_0$) is $(1, 1, ..., 1)$, so is the sum of all dummies. 
		3. let's say a factor variable has two values `v1` and `v2`. By dropping `v1` we make it the baseline. Let's say we fit and get $\beta_2=1.3$, this means $v_2=1$ contribute 1.3 more to $\hat{y}$ than when $v_{i\ne 1}=0$.
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
	5. `dtype`:  before `get_dummies`, `PropertyType` is `object`, i.e. string; after `PropertyType_Townhouse` is `bool`
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
### TBD
1. $R^2$ will always increase or stay the same as more variables are added, which is why **Adjusted R squared**is often used for multiple regression to account for model complexity.
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

## 5. Classification
### Naive Bayesian
1. Calculates posterior probabilities, no coefficients needed
2. *works only with categorical predictors;* needs to bin numeric ones into levels
3. `sklearn`: smoothing parameter $\alpha$ to deal with absent predictor in some classes, i.e., zero likelihood

| Generative (Naive Bayes, LDA)             | Discriminative (Logistic Regression, SVM) |
| ----------------------------------------- | ----------------------------------------- |
| Models P(X \| Y) and P(Y)                 | Models P(Y \| X) directly                 |
| Can generate new samples                  | Only classifies                           |
| Makes assumptions about data distribution | Makes fewer assumptions                   |
### Logistic regression
1. Model *log-odds/logits* with linear regression
	1. then map logits to probability, via logistic function, to make decision
2. Interpreting
	1. Use odds ratio, or, $\text{Odds}(Y=1|X=x_1)/\text{Odds}(Y=1|X=x_0)$. 
		1. E.g., factor variable. "regression coefficient for `purpose_small_business` is 1.21526. This means that a loan to a small business compared to a loan to pay off credit card debt reduces the odds of defaulting versus being paid off by $exp(1.21526)\approx 3.4$".
		2. E.g., numeric variable. "The variable `borrower_score` is a score on the borrowers’ creditworthiness and ranges from 0 (low) to 1 (high). The odds of the best borrowers relative to the worst borrowers defaulting on their loans is smaller by a factor of $exp(-4.61264) \approx 0.01$." A hundred times less likely to default!
3. VS linear regression
	1. Do not mix `log-odds` with target; the targets are always binary, not continuous as `log-odds`.

| Aspect                   | Linear Regression               | Logistic Regression                 |
| ------------------------ | ------------------------------- | ----------------------------------- |
| Target variable          | Continuous                      | Binary                              |
| Assumed distribution     | Gaussian (Normal)               | Bernoulli                           |
| Link function            | Identity                        | Logit (sigmoid)                     |
| Objective                | Minimize sum of squared errors  | Maximize log-likelihood             |
| Loss function            | Mean Squared Error (MSE)        | Log loss / Cross-entropy            |
| Estimation method        | Least Squares (OLS)             | Maximum Likelihood Estimation (MLE) |
| Closed-form solution     | Yes                             | No                                  |
| Optimization in practice | Closed-form or Gradient Descent | Gradient Descent / Newton / LBFGS   |
| Output range             | (-∞, ∞)                         | [0, 1]                              |
| Typical use case         | Regression                      | Classification                      |
### Metrics
1. Confusion table is the key
	1. Precision/Recall (i.e., TPR): one is column-wise, the other is row-wise
	2. True Positive Rate/True Negative Rate: both row-wise, i.e., true value based
		![[Pasted image 20260209140910.png|600]]
### Imbalanced Data
1. prioritize up/down weighting over up/down sampling?

### Code
1. Logistic regression
	```python
	# `penalty` is on by default
	# `C` is the inverse of regularization strength; large value negates penality
	logit_reg = LogisticRegression(penalty='l2', C=1e42, solver='liblinear')
	
	# prediction
	pred_log_prob = pd.DataFrame(logit_reg.predict_log_proba(X), columns=logit_reg.classes_)
	pred_prob = pd.DataFrame(logit_reg.predict_proba(X), columns=logit_reg.classes_)
	```


## 6. Statistical ML
1. "...are distinguished from classical statistical methods in that they are data-driven and do not seek to impose linear or other overall structure on the data". In other words, less statistical theories.
### KNN
1. for imbalanced data, set the cutoff at the probability of the rare event.
2. important parameter is number of neighbor to consider $k$.
3. Since it calculates the distance between data points, ==feature standardization is important!==
4. *Curse of dimensionality:* Distance metrics become less meaningful in high-d
```python
from sklearn import preprocessing
scaler = preprocessing.StandardScaler()
scaler.fit(X)
X_std = scaler.transform(X)
newloan_std = scaler.transform(newloan)
```
### Tree models
1. Compared to linear models, able to learn complex pattern; compared to KNN or naive Bayesian, very easy to interpret!
2. Understand best by plotting
	1. The rule at the top of each box actually applies to the children nodes; the `class=` is for current box
	2. Conventionally, left `True`
	3. ![[Pasted image 20260209221240.png|600]]
3. The *loss* on a given pair of feature and split, $k$ and $t_k$, is recursively minimized during training.
	$$J(k, t_k) = \frac{m_\text{left}}{m}\text{G}_{\text{left impurity}} + \frac{m_\text{left}}{m} \text{G}_{\text{right impurity}}$$
	1. Where impurity is measured by:
		- **Gini**: $p_k(1 - p_k)$
		- **Entropy**: $-p_k \log_2(p_k)-(1-p_k) \log_2(1-p_k)$
	2. where $m = m_{\text{left}} + m_{\text{right}}$, i.e., total counts is the sum of both children's counts
	3. The recursive partitioning algorithm is *not optimal*, because it only consider immediate children's impurities, i.e., *greedy*; exhaustive search is NP-hard.
4. Avoid overfitting
	1. `min_samples_split` (default `2`) and `min_sample_leaf` (default `1`)
	2. ==`min_impurity_decrease`== to enforce certain purity gain for each split
	3. TODO: Minimal Cost-Complexity *Pruning*. 
		1. Point is to prune nodes with gain smaller than threshold `ccp_alpha`. Greater threshold leads to harsher pruning, and looking for sweet spot where train/test accuracy both high and close
			1. ![[Pasted image 20260210064648.png|400]]
		2. [1.10. Decision Trees — scikit-learn 1.8.0 documentation](https://scikit-learn.org/stable/modules/tree.html#minimal-cost-complexity-pruning) and [Post pruning decision trees with cost complexity pruning — scikit-learn 1.8.0 documentation](https://scikit-learn.org/stable/auto_examples/tree/plot_cost_complexity_pruning.html)

### Random Forest
> `ensemble` -> bagging -> random forest
> Bagging: Bootstrapping aggregating
1. Randomness
	1. Bootstrapped $n$, $n < N$ (total num of records), records for each tree
		1. This resample is also called the 'bag', as in 'out-of-bag' (OOB).
		2. It gains in accuracy but lost interpretability of individual trees
	2. Random select $p$, $p < P$ (total num of variables) and usually $\sqrt{P}$, variables at *each split*
	3. Why helpful
		1. Reduce overfitting and correlation between DTs.
		2. Enables OOB score evaluation; this can be understood as *validation loss*
2. Additional eval `oob_score_`. 
	1. The misclassification rate on trees out-of-bag, i.e., not participated in training this tree, i.e., *validation loss*. 
		```python
		rf = RandomForestClassifier(n_estimators=500, random_state=1, oob_score=True)
		print(rf.oob_score_)
		```
3. Feature importance
	1. By accuracy decrease 
		1. Randomly permuting the values has the effect of removing all predictive power for that variable. The accuracy can be computed from the out-of-bag data (so this measure is effectively a cross-validated estimate).
	2. By Gini impurity decrease
		1. Can be accessed in `rf.feature_importances_`. This measure is based on the training set and is therefore less reliable than a measure calculated on out-of-bag data.
	3. See code!
4. important hyperparameters
	1. `min_samples_leaf` and `max_leaf_nodes`. The latter controls maximum number of nodes as well, given it's a complete binary tree.
### Code
1. Make categorical variables ordinal
```python
# `ordered=True` encodes `paid off` as 0 and `default` as 1
loan_data['outcome'] = pd.Categorical(loan_data.outcome, categories=['paid off', 'default'], ordered=True)
```
2. Feature importance from RF
	1. This is an good example, with `cross validation` as well!
```python
from sklearn.model_selection import train_test_split
```
### Boosting
> `ensemble` -> boosting -> Adaboost, gradient boosting...
1. *How is boosting different from bagging*
	1. weaker learners: usually one or two layers (trees become *stumps*)
	2. different trees, different weights in final prediction
	3. trees built sequentially and dependently (on the previous tree's mistakes)
2. How AdaBoost is Trained
	1. **Stump weight in final prediction**: The $j$th stump's weight (amount of say) is:
   $$\alpha_j = \frac{1}{2} \ln\left(\frac{1 - \epsilon_j}{\epsilon_j}\right)$$
		1. Lower error rate → higher $\alpha$ → more influence
		2. Higher error rate → lower $\alpha$ → less influence
		3. Error = 0.5 (random guess) → $\alpha = 0$ (ignored)
			1. it's possible, but not in practice, for $\alpha < 0$ and the stump's prediction gets flipped in the final vote.
	2. **Sample weight update**: After each iteration:
		1.  Misclassified samples → weight increases (focus more on hard examples)
		2. Correctly classified samples → weight decreases
3. How Gradient Boosting works
	1. it fit the new stump to the *residual errors* made by previous one
	2. Nice manually worked out example in [Hands-On ML]([6. Ensemble Learning and Random Forests | Hands-On Machine Learning with Scikit-Learn and PyTorch](https://learning.oreilly.com/library/view/hands-on-machine-learning/9798341607972/ch06.html#id112))
		1. also good discussion on `n_estimators_` and `n_iter_no_change` for early stopping
4. Detailed explanation on many boosting algos from StatQuest
	1. E.g., [AdaBoost, Clearly Explained](https://youtu.be/LsK-xG1cLYA) 
5. Random Sampling/Features in Boosting
	1. **AdaBoost**: Uses *all* data, but with *weights* that change each iteration. No randomness in sampling.
	 2.  **Gradient Boosting**: Fits residuals sequentially on *all* data. Pure gradient descent, no randomness.
	3. **Stochastic Gradient Boosting** (XGBoost, etc.): Adds optional randomization for regularization, not for creating diversity like in RF.
6. #code Smart way to return error `abs(true_prob - pred_prob) > 0.5`
7. XGBoost hyperparameters at the end of the section
### Boosting vs Bagging: Overfitting Tendency

| Method                | Random Samples             | Random Features          |
| --------------------- | -------------------------- | ------------------------ |
| **AdaBoost**          | No (uses weighted samples) | No                       |
| **Gradient Boosting** | No (by default)            | No (by default)          |
| **XGBoost/LightGBM**  | Optional (`subsample`)     | Optional (`colsample_*`) |
| **Random Forest**     | Yes (bootstrap)            | Yes (at each split)      |
|                       |                            |                          |

| Aspect          | Bagging (Random Forest)        | Boosting (XGBoost, AdaBoost)                |
| --------------- | ------------------------------ | ------------------------------------------- |
| Training        | Parallel, independent trees    | Sequential, each tree fixes previous errors |
| Focus           | Random subsets, diverse trees  | Hard examples get more weight               |
| Noise handling  | Averaging reduces noise impact | Can chase noise (outliers get high weight)  |
| More iterations | Usually helps or plateaus      | Can overfit if too many                     |
1. Why Boosting is More Prone to Overfitting
	1. **Bagging**: Trees are independent → errors are random → averaging cancels them out. Hard to overfit by adding more trees.
	2. **Boosting**: Each tree deliberately corrects mistakes → if those "mistakes" are noise/outliers, the model learns noise. Sequential dependency amplifies errors.
2. Regularization in Modern Boosting (XGBoost, LightGBM)
	1. `max_depth`, `min_child_weight` — control tree complexity
	2. `reg_lambda`, `reg_alpha` — L2/L1 penalties
	3. `subsample`, `colsample_*` — stochastic boosting
	4. Early stopping — stop before overfitting
3. With proper tuning, boosting can generalize as well or better than bagging—but it requires more care.
## Unsupervised ML
1. Use cases
	1. create predictive rules without labels, e.g. `k-means clustering`
	2. dimension reduction, e.g. `PCA`
	3. extended data exploration
### Principal components analysis
1. The idea is to decompose the Sample Covariance Matrix ($S$) into rank 1 components, each ranked by its contribution to the total covariance.
	1. **Matrix orientation**:  If $X$ is centered data (n×p, samples as rows), then $S = \frac{1}{n-1} X^T X$ . The SVD of $X = U\Sigma V^T$ gives you the principal components directly as columns of $V$.
	2. **Connection to SVD**: The right singular vectors $V$ are eigenvectors of $X^T X$, and eigenvalues of $S$ relate to singular values by $\lambda_i = \frac{\sigma_i^2}{n-1}$.
	3. **Why SVD is cheaper**: Computing $X^T X$ explicitly squares the condition number and loses numerical precision. SVD works directly on $X$, avoiding this.
2. The geometry behind
	1. The sum of squared distances from the data points to $\mathbf{v}_1$ line is a minimum. In other words, $\mathbf{v}_1$ lies in the direction of most variance.
	2. Here distance is *perpendicular* to singular vectors, unlike least square, where distance is vertical. More detail: Gilbert Strang, *Linear algebra and learning from data*, 77.
		1. When project data $a_j$ onto singular vectors, the 2nd sum on the RHS is minimized, i.e., the residual variance. $$\sum_1^n \left\lVert a_j \right\rVert^2=\sum_1^n \left\lVert a_j^T u_1 \right\rVert^2 + \sum_1^n \left\lVert a_j^T u_2 \right\rVert^2$$
3. *Centering is essential* since PCA captures variance around the mean.
## Topics
### Things to do when new data
```python
df.head()  # feel the data
df.dtypes  # check column types, numeric vs factor
df.describe()  # check numeric variables
df.select_dtypes(include=["object", "category", "bool"]).nunique()  # number of possible values for non-numeric variables
df.object1 = df.object1.astype("category")  # change object col to 'category' dtype
```
### Multicollinearity
1. Collinearity
	1. Don't need to worry about it:
		1. Tree-based models (Random Forest, XGBoost, Decision Trees) - they handle collinearity fine
		2. Naive Bayes, k-NN - no coefficient estimation
	2. Should worry about it:
		1. Logistic regression (without regularization) - same multicollinearity issue as linear regression
		2. Regularized logistic regression (L1/L2) - can handle it, but dropping doesn't hurt
		3. do `drop_first=True`!
### Standardization
- Required/Strongly Recommended:
	- KNN, SVM, Neural Networks - distance/gradient-based, sensitive to scale
	- PCA - maximizes variance, needs equal scaling
	- Ridge/Lasso - penalty applies uniformly only when scaled
	- K-Means, hierarchical clustering
- Not Required:
	- Tree-based models (Decision Trees, Random Forest, XGBoost) - split on thresholds, scale-invariant
	- Naive Bayes - probability-based
	- Plain linear/logistic regression - coefficients adjust, though centering helps interpretability
- One-hot variables (possibly from `get_dummies`) should be standardized as well
- Practical rule
	- If the algorithm uses distances, gradients, or regularization penalties, scale. If it uses splits or conditional probabilities, skip it.
## To do
1. ==Cheat sheet==
	1. algorithms: popularity, required data processing, missing data, standardize data, how does it work, when to use, important parameters and how to adjust them, model specific metrics, model or data centric, search of hyperparameter (`GridSearchCV`?), regularization
2. Central Limit Theorem #CLT 
	1. Illustrating examples from 3B1B [video](https://youtu.be/zeJD6dqJ5lo)
		1. Galton Board: each collision with a peg is a Bernoulli, more levels of pegs, i.e., *larger sample size*, leads to Gaussian distribution of the final displacement.
		2. Sum of dice: each dice is uniform, but the distribution (*sampling distribution*) of sum of large number of dice (*sample statistic*) follows Gaussian