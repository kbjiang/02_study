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
# Lecture 7 "MLE"
## Notes
1. $P(D;\theta)$ vs $P(D|\theta)$ 
2. parameter $\theta$ vs data $D$
3. Hallucinated toss for coins--prior belief to avoid 0 probability; mental picture: Assuming already 10 head and 10 tails, start tossing; related to Beta distribution initial condition
4. Naive Bayes
	1. the monkey + two typewriters (spam/no-spam) metaphor