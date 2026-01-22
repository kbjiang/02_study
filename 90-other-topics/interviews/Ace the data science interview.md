## Probability
1. 5.5. Do remember to take overcounting into account (by including the denominator!)
	1. $$\frac{(n+m)!}{n!m!}$$.
2. 5.11. Markov chains model
	1. Why is it a Markov chain? Because given different current states, e.g. $H$ or $T$, the transition probabilities are different, e.g. if $T$ then never Win. ![[Pasted image 20260112104425.png]]
		1. State space: *S={S,H,T,Win,Lose}*
	2. In this case, the Markov chain has **absorbing states** (Win and Lose), so it does **not** have a traditional stationary distribution over all states because the chain is not irreducible—it eventually gets absorbed and stops.
	3. Similar *problem 5.23*.
3. 5.16. The recursive relation
	1. Solve for $P_{\text{A wins game}}$. Here $0$ and $1$ are the certainties. $$P_{\text{A wins game}} = P_{\text{A wins 1st 2 points}} \times 1 + P_{\text{A wins 1 of 1st two points}} \times P_{\text{A wins game}} + P_{\text{A loses 1st two points}} \times 0$$
	2. Similar *problem 5.23*.
4. 5.18. Good trick, but not generic?
5. 5.22. Use multivariant PDF. Things to pay attention to
	1. Assuming $x_1 > x_2 > x_3$, the limits of each variable can be tricky; i.e., $x_1$ should be in $[1/3,\ 1/2]$ and $x_2$ in $[1/2(1-x_1), x_1]$.
	2. Need to consider permutation of $x_i$ to avoid over-estimation.
	3. The normalization $z$ is not 1!.

## Statistics
1. 6.1., 6.2., say it out loud
2. 6.4. Covariance and correlation
	1. both means *linear* relationship between two R.V.
	2. former ranges from $(-\infty, \infty)$ and has unit, latter is former normalized by variances, and ranges from $(-1, 1)$ and is unit-less. 
3. 6.9. $t$-test and $z$-test
	1. both require the test statistic, e.g. sample mean, to be normally distributed. This means either initial population is normal, or sample size is large ($n \ge 30$).
	2. Use $t$-test instead of $z$-test when (a) number of data is small and (b) unknown population variance.
4. 6.13, 6.14. Sum of expectations
	1. Also geometric distribution
5. 6.18. The CDF trick.
	1. let $Z=\text{min}(X, Y)$ then calculate CDF $P(Z \le z)$. 
6. 6.20. The use of indicator random variable.
	1. similar: 6.26
7. 6.21. Answer has minor issue. No recursion required.
8. 6.28. Unbiased vs consistent estimator
	1. unbiased: expected value of the estimator equals true value. E.g., sample mean
9. Function of R.V.
	1. 6.32, 6.35
### Major tricks
1. recursive relation
2. use CDF and differentiation to find PDF
3. set the right R.V.; e.g., indicator, last occurrence etc.

## Machine Learning
1. Linear regression
	1. RSS, ESS
	2. Assumptions
		1. Linearity, homoscedasticity...
2. Classification
	1. ROC and AUC
		1. Mental picture: all data points lined up by its predicted likelihood of being positive. If perfect model, than all positive samples will be on the right of all negative samples; in other words, any overlapping indicates that the model is not able to perfectly separate the data. See illustrations from [Classification: ROC and AUC  |  Machine Learning  |  Google for Developers](https://developers.google.com/machine-learning/crash-course/classification/roc-and-auc)
		2. AUC "represents the probability that the model, if given a randomly chosen positive and negative example, will rank the positive higher than the negative." 
			1. To understand in terms of TPR/FPR, needs mathematical proof. But intuitively, positive samples in overlapping area would have lower "weights", because there are more "likely" negative samples; and the weighted sum is the AUC. 
			2. Why $1/2$ baseline? All data points are completely randomly ranked, the overlap is uniform.
		3. AUC is independent of threshold; the thresholds are just to trace out the area
	2. Decision Tree and its ensembles
		1. Entropy and Gini index
		2. Random Forest
			1. Bagging: reduce variance between individual DT
				1. Each tree receive different random sample from training set, i.e., bootstrapping.
			2. Random subset of features: reduce overfitting and correlation between DTs.
		3. Boosting
	3. SVM, DT-based and NN can handle *data not linearly separable*.
3. End-to-end ML
	1. Problem frame and *constraints*
		1. dependent variable, shareholder, baseline, impact of error...
		2. latency, throughput, cloud or edge
	2. Define metrics
	3. Data
		1. Data source
		2. Data exploration and cleaning
		3. Feature engineering
	4. Model
		1. model selection
		2. model training and evaluation
		3. model deployment
	5. Iterate
### Interview Questions
1. 7.5. Multicollinearity
	1. PS: thought it made coefficients unstable and interpretation difficult, the prediction accuracy usually stays
2. 7.6. Random forest over Decision Tree
3. 7.7. Missing data
4. 7.11. Enough data to create an accurate model?
	1. what is "accurate"
	2. metrics and baseline
	3. performances at different data sizes
	4. if not good enough, more data? feature engineering?
5. 7.15. Cross validation
	1. It is for ==evaluating== a model and uses training internally "Cross validation is a technique used to assess the performance of an algorithm in several resamples/ subsamples of training data."
	2. It can be used for tuning hyperparameters. E.g., degree of polynomial in SVM
6. 7.17. Collaborative Filtering
	1. *Embeddings* for both users and items, then *minimize* the difference between observed rating and inner products. See [[temp-Collaborative-filtering]] for more.
7. 7.19. Information gain in decision tree
	1. Information gain is calculated using entropy, as opposed to Gini impurity, but the equation is the same
		1. $\text{Gain} = I(\text{parent})-\sum_i w_i I(\text{child}_i)$, where $I$ is either entropy or Gini.
	2. The entropy is calculated on target classes, NOT splitting feature!
8. 7.20. $L_1$ vs $L_2$ penalties 
	1. Loss contours from Chapter 4 of [Hands on ML](https://learning.oreilly.com/library/view/hands-on-machine-learning/9798341607972/ch04.html) ![[Pasted image 20260118163129.png]]
	2. During gradient descent
		1. $L_1$ likely to "zero out" particular weights, in this case $\theta_2$, while $L_2$ does not.
		2. when approaching minimal, $L_1$ bounces around because its gradient is always $\pm 1$, while gradients of $L_2$ shrinks .
9. 7.26. $k$-means gradient descent
10. 7.27. Kernel trick in SVM 
	1. TODO: 
		1. go over appendix of [Hands-On Machine Learning with Scikit-Learn and PyTorch](https://ageron.github.io/homlp/HOMLP_Appendix_C.pdf)
			1. *Kernel* is a function computes the dot product $\phi(\mathbf{a})^T\ \phi(\mathbf{b})$ based only on the original vectors $\mathbf{a}$ and $\mathbf{b}$, without having to know the transformation $\phi$.
				1. So they cannot be arbitrary, but follow certain conditions, i.e. *Mercer's conditions*
		2. go over [Support Vector Machines Part 1 (of 3): Main Ideas!!!](https://www.youtube.com/@statquest)
			1. Hard margin classifier => Support vector classifier (soft margin, allows misclassification, bias variance trade-off) => Support vector machine (project to higher dimensional space to accommodate nonlinearity)
		3. go over [16. Learning: Support Vector Machines](https://www.youtube.com/@mitocw)
			1. the math and the dual problem
		4. also good https://youtu.be/iEQ0e-WLgkQ
11. 7.30. The steps of building a churn prediction model
	1. understand business usecase
	2. model consideration: interpretability?
	3. data and feature
	4. deploy and refine
12. 7.32. PCA
	1. to frame it as a Lagrangian by maximizing variance of data in direction $w$ while constraining $\lVert w \rVert^2 = 1$
	2. see [[temp-PCA]] for full detail
13. 7.3.5. variance of least square estimates
	1. see answer
		1. matrix form of the solution and $y=X \beta + \epsilon$.
		2. for last equation, $((X^T\ X)^{-1}X\epsilon)^2 = (X^T\ X)^{-1}X\ \Sigma_{\epsilon\epsilon} ((X^T\ X)^{-1}X)^T$, where $\mathbb{E}[\Sigma_{\epsilon\epsilon}]=\mathbb{I}*\sigma^2$ is the covariance matrix of noise, because it's NOT a dot product!
	2. see 14.4.2 of Rice.

## Product sense
### Always 
1. Ask clarifying question about the product. Do NOT assume anything.
	1. E.g., "what is the goal for Youtube Premium pricing?" Sometimes current pricing might be for long term gain.
2. Explain the product's business goal before providing solution
3. Think about cost and align with stakeholders from different departments

### Questions
1. 10.10. Four steps for diagnosing metric change
	1. scope of the metric change
	2. hypothesize contributing factors
	3. validate each factor
	4. ?
2. 10.12. Stickiness
	1. Daily active user (DAU), WAU, MAU
	2. the ratios such as DAU/WAU measures stickiness; lower meaning many people only use it weekly
3. 10.16. Pricing
	1. three dimensions: cost-based, value-based, competitor-based
