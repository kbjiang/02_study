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
		1. "the fact that certain random fluctuations can be well controlled in high dimensions, while it would be very complicated to make such predictive statements in moderate dimensions."
	3. The corners of hypercubes in high dimension 'sticks out', which leads to feature sparsity in L1 regularizations. ^l1
		1. For 2D, the distance from the center to a vertex (radius of the circumscribed sphere) is $\sqrt{(1/2)^2+(1/2)^2}=\sqrt{2}/2$ and the apothem (radius of the inscribed sphere) is $1/2$; for 4D, the former is $\sqrt{(1/2)^2+(1/2)^2+(1/2)^2+(1/2)^2}=1$ and the latter stays $1/2$--the 2D projection no longer appear convex. ![[Pasted image 20261004175343.png|200]]![[Pasted image 20261004175621.png|200]]![[Pasted image 20261004175657.png|200]]
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
	==However, it's rarely the case, especially when $x$ is high dimensional.== 
3. *Therefore we need tricks*
	1. modeling: assume certain probabilistic models, e.g., Gaussian, Binomial...; and estimate their parameters, the number of which is usually small comparing to sample size, with algorithms such as MLE/MAP.
	2. additional assumptions: Naive Bayes for example
### MLE and MAP (explained as coin toss)
1. From MLE to MAP intuition
	1. With pure MLE, one may ran into 0 heads in first $n$ toss, but we know a coin *should not* have 0 chance for heads
	2. Then we assume imaginary toss--prior belief (MAP) to avoid 0 probability. For e.g.: if we believe a fair coin, then assume 10 head and 10 tails before tossing
2. Beta distribution as the prior distribution intuition
	1. it is of the same distributional family as the binomial distribution (**conjugate prior**) $\rightarrow$ the math will turn out nicely
	2. $P(\theta)=\frac{\theta^{\alpha-1} (1-\theta)^{\beta-1}}{B(\alpha, \beta)}$, means we start with $\alpha-1$ imaginary heads and $\beta-1$ imaginary tails.
		1. Greater values of $\alpha$ and $\beta$ means stronger belief. E.g., if we are quite sure $\theta=0.5$ then we can have 100 imaginary heads/tails--the prior Beta distribution is narrow around 0.5.
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
2. From MLAPP, $P(y|x,w)=\text{Ber}(\text{sigm}(w^Tx))$. The mental picture is that for each $x$, the $y$ follows a Bernoulli distribution whose $p$ is the $\text{sigm}(w^Tx)$. The error bar for Bernoulli is not well defined, it's just for pedogogical purpose.![[Pasted image 20260915115346.png]]
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
### Linear regression in Stats vs ML
1. Why zero-centered Gaussian prior leads to L2 regularization?
	1. Comparing to uniform prior, this prior has higher probability for smaller values of parameters.
### Linear regression in Stats vs ML
1. Angles
	1. In stats Linear Regression (LR) is approached as a parameter estimation, where uncertainty matters quite a bit (confidence interval for e.g.); in ML it's a way to introduce neural nets and gradient descent, people settle with parameters perform best on test set.
2. Methodology: MLE vs Optimization
	1. In stats, we use models for $P()$ with well-known prob. dist.
3. Generative vs Discriminative
4. ![[Pasted image 20260910121805.png]]
### Gradient Descent
1.  the challenge is to find the right step size
	1. all the methods with momentum for example
2. Ways to optimize: Newton's method, line search, MLAPP Fig 8.2
	1. Newton.  "if the function is flat or almost flat with respect to some dimension. In that case the second derivatives are close to zero, and their inverse becomes very large - resulting in gigantic steps."
### MISC
1. OLS and its geometry. Ambient space and its axes
## References
1. [NBayesLogReg.pdf](https://www.cs.cmu.edu/~tom/mlbook/NBayesLogReg.pdf) by Tom Mitchell ^tom
2. [Linear Regression: When it feels complicated, you’re probably reading a stats book](https://medium.com/towards-artificial-intelligence/linear-regression-statistical-vs-machine-learning-view-b157a1efeb3e)
# Lecture 14-15 SVM 
1. dd
## References
1. MIT 6.034 [Lecture 16](https://youtu.be/_PwhiWxHK8o)
	1. For positive ($y_+=1$) and negative ($y_-=-1$) supporting samples, we have  $y_+(w^T x_+ +b)$ = $y_-(w^Tx_- +b)=1$.  Therefore, the distance between these two samples in $w$ direction, i.e., the width of the gutter, is just $w^T(x_+ - x_-)/||w||_2 = ((1-b)-(-1-b))/||w||_2=\frac{2}{||w||_2}$. Namely, *smaller weight, greater separation*. 
# Lecture 16 Generic ML framework
## ML framework
1. Take SVM loss as e.g., $\min_{w, b} C\sum_i \max[1-y_i(w^Tx_i+b), 0] +||w||_2^2$. 
	1. The former is the loss, or the mistakes made, and involves the data
	2. The later is the regularizer, which does NOT involve on the data
	3. The trade-off: get every training point correct, vs, a simpler solution that generalizes better (Occam's razor principle). Or, *simplest model that explains the data well enough.*
## Losses
1. Exponential loss freaks out about most misclassified points, while hinge cares about points all the same. See the plot below.
# Lecture Bias-Variance tradeoff
## Bias-variance tradeoff
1. same test (x, y), different training D leading to different $h_D(x)$, this is variance.
	1. $h_D$ is  also a R.V.
2. The mathematical definition.![[Pasted image 20261021170818.png|800]]
3. High variance: given a test data $x$, each dot corresponds to the test error associated with a model trained with some training set $D$ . When you take the average over all $D$s, the test error might be small (low bias)
	1. ![[Pasted image 20261021170713.png|400]]
4. How to map this graph to the math
	1. As # training instances go up, test error drops and training error increases.
		1. test loss is lower bounded by train loss
	2. Regimen #1: small train set, low train loss, high variance, overfitting. 
		1. Test loss can be misleading: with a new training set, the test error might be quite low--on average, the test error would be low. This is because *when the model is powerful enough to learn noise from every small train set, overall the noises will cancel out*. In other words, $h_D(x)$ varies a lot, but $\bar h(x)$ can be close to $\bar y(x)$, i.e. low bias.
		2. This graph cannot tell if the trained model is good, even if both losses are low, due to small size of train set. Instead of loss curves, report confidence interval. See how [[#^CI]]
	3. Regimen #2: large train set, high test and train loss, high bias, underfitting. I.e., $\bar h(x)$ far from $\bar y(x)$.
	4. Note that adding training data to move from region #1 to #2, we are trading variance for bias. That's the *tradeoff*.
	5. ![[Pasted image 20261021171520.png|600]]
5. How to get CI of the error. ^CI
	1. Analytical: normal approximation of the binomial proportion
		1. $\text{CI} = \text{error} \pm 1.96 \times \sqrt{\frac{\text{error} \times (1-\text{error})}{n}}$ 
	2. Bootstrapping
6. Let's say I did evaluation five times, each with accuracy $x_i$, how do I report the overal accuracy with error bar?
	1. $s = \sqrt{\frac{\sum(x_i-\bar x)^2}{n-1}}$
## How to debug?
1. Identify which regime we are in.
2. Regime #1, high bias
	1. Kernel (linear classifiers)
3. Regime #2

# Kernel 
1. Motivation:
	1. Need non-linear features to deal with problems non-linear separable in original space
2. bad news: can easily have too many features $2^d$, where $d$ is number of original features
3. KNN does not require kernel because it's already nonlinear and not in high-bias regime.
4. Gram matrix vs Kernel function
	1. $K_{ij}=k(\mathbf{x}_i, \mathbf{x}_j)$
### The Logic for the Kernel Trick
1. *Dual* Optimization Framing
	1. Frame the loss function purely in terms of the pairwise dot products of the data vectors ($\mathbf{x}_j^T \mathbf{x}_i$). 
	2. As a result, the parameters to update during training (learning) are the *dual* coefficients ($\alpha$).  
	3. Inference is also performed entirely using these coefficients and pairwise data interactions.
 2. Kernel Definition & Mapping Properties
	 1. Define a kernel function $K(\mathbf{x},\mathbf{z}) = \phi(\mathbf{x})^T \phi(\mathbf{z})$ to compute similarities directly in the *original feature space*.
	 2. **Space Dimensionality:** Different kernels map data to different feature spaces. 
		 1. The **RBF kernel** maps data to an infinite-dimensional space.
		  2. The **Polynomial kernel** $K(\mathbf{x},\mathbf{z}) = (\mathbf{x}^T \mathbf{z} + c)^p$ maps data into a finite higher-dimensional space of exactly $\binom{d + p}{p}$ dimensions (where $d$ is the original feature dimension).
	3. **Hyperparameter Leverage:** Choose kernel functions with meaningful levers in the input space:
		1. The **Polynomial kernel** controls complexity via its degree $p$ and the trading constant $c$.
		2. The **RBF kernel** controls complexity via its radius of influence $\gamma$. 
3. How Input Space Kernels Affect Higher-Dimensional Training (RBF Example)
	1. Extreme Case 1: Large $\gamma$ (Very Small Radius of Influence)
		1. **In the Input Space:** The kernel acts like a sharp spike. $K(\mathbf{x},\mathbf{z})$ drops to $0$ rapidly if $\mathbf{x}$ and $\mathbf{z}$ are even slightly separated.
		2. **In the Infinite-Dimensional Space:** Because $K(\mathbf{x},\mathbf{z}) = \phi(\mathbf{x})^T \phi(\mathbf{z}) \to 0$, every single training point becomes **perfectly orthogonal** to every other training point ($\phi(\mathbf{x}_i) \perp \phi(\mathbf{x}_j)$).
		3. **The Training Impact:** If every data point sits on its own completely mutually perpendicular, independent dimension, the model can lineally separate *any* arbitrary assignment of labels perfectly. This grants the model infinite capacity to memorize training data, leading to a jagged decision boundary and **severe overfitting (High Variance)**.
	 2. Extreme Case 2: Small $\gamma$ (Very Large Radius of Influence)
		1. **In the Input Space:** The kernel is broad and flat. Even for points far apart, $K(\mathbf{x},\mathbf{z}) \approx 1$.
		2. **In the Infinite-Dimensional Space:** Because the vectors are normalized to length 1, an inner product close to 1 means $\cos(\theta) \approx 1$. Thus, the mapped vectors all point in **virtually the exact same direction** (highly collinear).
		3. **The Training Impact:** Because the vectors are squeezed together into a tight, almost one-dimensional cluster on the hypersphere, the model's effective capacity drops. It becomes impossible to find a separating hyperplane, forcing a highly smoothed, flat decision boundary which leads to **underfitting (High Bias)**.
### Notes
1. We then run gradient descent on $\mathbf{\alpha}$ in dual space like we did with $\mathbf{w}$ in primal space. See the other md file #TODO
2. Prediction
	1. $h(\mathbf{z})=\mathbf{z}^T\mathbf{w}=\mathbf{z}^T\mathbf{X}\mathbf{\alpha}=\mathbf{k_*}\mathbf{\alpha}$, meaning the prediction is a linear combination of data points 
3. Beautiful [discussion](https://youtu.be/7LlpTABc27s?list=PLl8OlHZGYOQ7bkVbuRthEsaLr7bONzbXS&t=459) about the essence of SVM.
	1. Linear SVM resembles KNN in high dimensional space.
4. The regularization parameter $C$ in dual representation. [西瓜书 6.4]

### References
1. A good [video](https://youtu.be/OKFMZQyDROI) for mapping intuition.

# Lecture 15 Gaussian Process #TODO 
# Lecture 16 KD Trees #TODO 

# Lecture 17 Decision Trees #TODO 
1. The CART cost function for classification
	$$G(S)=\frac{|S_L|}{|S|}G(S_L)+\frac{|S_R|}{|S|}G(S_R),$$where $G$ is the Gini Impurity, and $\frac{|S_{L/R}|}{|S|}$ is the fraction of inputs in the left/right sub-tree. 
	1. Not $S_{R/L}$ may contain all $C$ classes.
	2. It's a  greedy algorithm--it looks for optimal split at *current* level only, without checking whether this leads to overall optimal result.
	3. It does so by looking at every possible split at the level... 
2. Parametric or not? $\star$
	1. Primal algo is parametric, coz number of $w_i$ is fixed regardless of number of data points; dual algo is non-parametric, coz number of $\alpha_i$ increases as training data increases.

# Lecture 18 Bagging ==> Reduce variance
1. Trees have high variance problem--bagging is the fix
