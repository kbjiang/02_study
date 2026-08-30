https://www.cs.cornell.edu/courses/cs4780/2018fa/
# Lecture 1 "Supervised Learning Setup" -Cornell CS4780 Machine Learning for Decision Making
## Notes
1. ML vs Traditional CS
	1. Traditional CS: input + program -> output
		1. this is also what happens at ML inference time
	2. ML: input + output -> program
2. ML vs AI
	1. AI: top down, try to mimic human, focus on *Logic*
	2. ML: bottom up, smaller goals, statistic + optimization
3. *Inductive* [[#^xiguashu]]
	1. Inductive is the process of moving from specific facts or observations to a broader, general conclusion, i.e., *generalization*. Deductive is the opposite direction.
	2. No free lunch (NFL) theorem: no inductive bias is universally good. 
## Reference
1. 西瓜书 Section 1.3 ^xiguashu
	1. Inductive, NFL and proof 
	
---
# Lecture 4 "KNN and Curse of Dimensionality"
## Notes
### KNN
1. Assumption
	1.  Distances reliably reflect a semantically meaningful notion of the dissimilarity.
2. why KNN still works (despite of Curse of Dimensionality)
	1. real data intrinsic low dim: subspace/manifold
		1. e.g., an image of a face is 10k pixel, but a face may only need ~20 features (nose, eyes...) to describe
	2. manifold def: locally Euclidean distance is valid
	3. this is also why PCA and dimension reduction is important
		1. KNN is very slow in high dimensional $O(n \times d)$
### Curse of Dimensionality
1. As $d \gg 0$, points drawn from a *probability distribution* stop being similar to each other, and the $k$NN assumption breaks down.
2. Pairwise distances grow with dimensionality 
	1. pairwise $L^2$ distance when each dimension follows $U[0, 1]$ ![[Pasted image 20260729210325.png]]
## Reference
1. [Mathematics of Data Science](https://arxiv.org/abs/2607.11938) Chapter 2, *Curses, Blessings, and Surprises in High Dimensions*
	1. Strange geometry in high dimension -- curse
	2. Concentration of measure (large sample size $\leftrightarrow$ high dimension) -- blessing
---
# Lecture 5 "Perceptron"
## Notes
1. Adding a dimension for bias, i.e., $[\mathbf{x}, 1] \text{ and } [\mathbf{w}, b]$, maintains the linear separability of the data.
	1. (Left:) The original data is 1-dimensional (top row) or 2-dimensional (bottom row). (Right:) After a constant dimension was added to all data points such a hyperplane exists.![[Pasted image 20260805212158.png|600]]
2. Why Perceptron works:
	1. In high-dimensional space, data points are almost always far away from each other therefore separable by a hyperplane.
3. Geometric Intuition of Perceptron algorithm
	1. Update rule
		1. for each wrongly classified $(x_i, y_i)$, do $\mathbf{w}_{t+1}=\mathbf{w}_t+(y_i)\mathbf{x}_i$ 
		2. if $y_i=-1$, misclassify means $\text{sign}(\mathbf{w}_t \cdot \mathbf{x})>0$, $\mathbf{w}_t$ needs to move away from $\mathbf{x}_i$, and vice versa.
	2. Illustration of a Perceptron update. (Left:) The hyperplane defined by $\mathbf{w}_t$ misclassifies one red (-1) and one blue (+1) point. (Middle:) The red point $\mathbf{x}$ is chosen and used for an update. Because its label is -1 we need to subtract $\mathbf{x}$ from $\mathbf{w}_t$. (Right:) The updated hyperplane $\mathbf{w}_{t+1}=\mathbf{w}_t+(−1)\mathbf{x}$ separates the two classes and the Perceptron algorithm has converged.![[Pasted image 20260805213344.png]]
4. Perceptron convergence
	1. If a dataset is linearly separable, perceptron algo is guaranteed to find a separating hyperplane in a finite number of updates.
	2. the number of steps relates to margin $\gamma$.

---
# Lecture 7-10 "Estimating Probabilities " and "Naive Bayes"

## Notes
### Student's $t$-distribution
1. It is used when *both*
	1. population std $\sigma$ is unknow--otherwise just use Gaussian to calculate exact sample std
	2. small sample size--when DOF $\approx 30$ it becomes Gaussian.
2. The fatter tail makes it harder to reject the null hypothesis-- it is good to be cautious when sample size is small. 
	1. Equivalently, this means Student's $t$-dist is more robust to outliers, as shown below. ![[Pasted image 20260812065630.png]]
### Estimating Probabilities from data
1. Assume data $D={(x_1, y_1),...,(x_n, y_n)}$ is drawn from $P(X, Y)$, i.e., $P(D)=\prod_{\alpha=1}^{n}P(x_\alpha, y_\alpha)$. *This is the core of probabilistic ML approches.*
2. If we have enough data, then we can estimate the true distribution by *counting*:
	$$
	 \begin{aligned}
	 \hat{P}(x, y)&=\frac{\sum{I}(x=x_i\wedge y=y_i)}{n} \text{, with $I$ being the indicator function;}\\
	\hat{P}(y|x)&=\frac{\sum{I}(x=x_i\wedge y=y_i)}{\sum\mathbb{I}(x=x_i)}.
	\end{aligned}
	$$
	However, it's rarely the case, especially when $x$ is high dimensional. 
3. *Therefore we need tricks*
	1. modeling: assume certain probabilistic models, e.g., Gaussian, Binomial...; and estimate their parameters, the number of which is usually small comparing to sample size, with algorithms such as MLE/MAP.
	2. additional assumptions: Naive Bayes for example
### MLE and MAP (explained as coin toss)
1. From MLE to MAP intuition
	1. With pure MLE, one may ran into 0 heads in first $n$ toss, but we know a coin *should not* have 0 chance for heads
	2. Then we assume imaginary toss--prior belief (MAP) to avoid 0 probability. For e.g.: if we believe it's a fair coin, assume 5 head and 5 tails before tossing
2. Beta distribution as the prior distribution intuition
	1. it is of the same distributional family as the binomial distribution (**conjugate prior**) $\rightarrow$ the math will turn out nicely
	2. $P(\theta)=\frac{\theta^{\alpha-1} (1-\theta)^{\beta-1}}{B(\alpha, \beta)}$, means we start with $\alpha-1$ imaginary heads and $\beta-1$ imaginary tails.
3. "True" Bayesian approach
	1. $P(Y|D) = \int_{\theta} P(Y, \theta | D) d\theta =  \int_{\theta} P(Y| \theta, D) P(\theta | D) d\theta$. The 2nd equal sign is because $P(Y, \theta)=P(Y, \theta)P(Y|\theta)$.  
	2. $\theta$ is integrated out - our prediction takes all possible models into account.
4. ==TODO==
	1. As always the differences are subtle. In MLE we maximize log[P(D;θ)] in MAP we maximize log[P(D|θ)]+log[P(θ)]. So essentially in MAP we only add the term log[P(θ)] to our optimization. This term is independent of the data and penalizes if the parameters, θ deviate too much from what we believe is reasonable. We will later revisit this as a form of [regularization](https://www.cs.cornell.edu/courses/cs4780/2018fa/lectures/lecturenote10.html), where log[P(θ)] will be interpreted as a measure of classifier complexity.
### Naive Bayes
1. Instead of estimating $P(y|x)$ directly, estimate $P(y)$ (easy) and $P(x|y)$ (hard, therefore simplify with assumptions) instead.
2. Num. of dimension = num. of features.  ![[Pasted image 20260817214448.png|1000]]
	1. For example, spam filter. Each word in the vocab $\leftrightarrow$ each dimension; counts is the value in that dimension (bag of words)
	2. $P(x_\alpha|y=c)$ is the class $c$ (e.g., spam) conditional distribution of feature $x_\alpha$ (e.g., # of word 'the')
3. When is NB valid? When $y$ is the confounder of the features. Around Lec 8 39:50; relates to confounder in causal analysis

## References
1. MLAPP
	1. The other interpretation is called the Bayesian interpretation of probability. In this view, probability is used to quantify our uncertainty about something; hence it is fundamentally related to information rather than repeated trials (Jaynes 2003). In the Bayesian view, the above statement means we believe the coin is equally likely to land heads or tails on the next toss. One big advantage of the Bayesian interpretation is that it can be used to model our uncer tainty about events that do not have long term frequencies. For example, we might want to compute the probability that the polar ice cap will melt by 2020 CE. This event will happen zero or one times, but cannot happen repeatedly. P27
2. MLAPP visualization #MLAPP #vis #stats
	1. [pyprobml/notebooks at master · probml/pyprobml · GitHub](https://github.com/probml/pyprobml/tree/master/notebooks)
	2. 2012 version, matlab https://github.com/probml/pmtk3/tree/master/demos
3. [Joint_MLE_MAP.pdf](https://www.cs.cmu.edu/~tom/mlbook/Joint_MLE_MAP.pdf) by Tom Mitchell
---
# Lecture 11-13 Logistic/linear regression
## Notes
### Logistic Regression
1. Parametric form $P(Y=y|X=x)=\frac{1}{1+\exp (-(w^T x + b) y)}$, where $y=\pm 1$. 
### Logistic Regression vs Naive Bayes
1. MLE vs MAP
	1. discrete input and smoothing
	2. with $\ell_1$ (Laplace dist prior) and $\ell_2$ (Gaussian)
2. Generative vs discriminative
	1. GNB and Logistic Regression
3. both are ways to learn $P(Y|X)$
	1. Gaussian NB implies LR form. Intro of Chap 3 in [[#^tom]]
4. fit distribution vs directly learn from data 
	1. generative vs discriminative
5. different assumptions?
6. different learning algorithm
7. "Logistic Regression is the discriminative counterpart to Naive Bayes. In Naive Bayes, we first model P(x|y) for each label y, and then obtain the decision boundary that best discriminates between these two distributions. In Logistic Regression we do not attempt to model the data distribution P(x|y), instead, we model P(y|x) directly. We assume the same probabilistic form P(y|xi)=11+e−y(wTxi+b) , but we do not restrict ourselves in any way by making assumptions about P(x|y) (in fact it can be any member of the [Exponential Family](https://en.wikipedia.org/wiki/Exponential_family)). This allows logistic regression to be more flexible, but such flexibility also requires more data to avoid overfitting. Typically, in scenarios with little data and if the modeling assumption is appropriate, Naive Bayes tends to outperform Logistic Regression. However, as data sets become large logistic regression often outperforms Naive Bayes, which suffers from the fact that the assumptions made on P(x|y) are probably not exactly correct. If the assumptions hold exactly, i.e. the data is truly drawn from the distribution that we assumed in Naive Bayes, then Logistic Regression and Naive Bayes converge to the exact same result in the limit (but NB will be faster)."
### Logistic Regression vs Linear Regression
1. LogReg cares if the data point is on the right side of the hyper-plane
2. LinReg cares about the distance from the hyper-plane, doesn't matter which side.

### Gradient Descent
1.  the challenge is to find the right step size
	1. all the methods with momentum for example
2. Ways to optimize: Newton's method, line search, MLAPP Fig 8.2
	1. Newton.  "if the function is flat or almost flat with respect to some dimension. In that case the second derivatives are close to zero, and their inverse becomes very large - resulting in gigantic steps."
### MISC
1. OLS and its geometry. Ambient space and its axes
## References
1. [NBayesLogReg.pdf](https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf) by Tom Mitchell ^tom
2. d