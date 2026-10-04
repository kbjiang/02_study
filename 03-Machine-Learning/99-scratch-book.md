1. Curse of dimensionality
	1. Interesting visualization https://youtu.be/1enQMVh1_Gw
	2. another lecture with intuition https://youtu.be/dZrGXYty3qc?t=527


## [Overview of Statistical Learning Theory Part 1](https://youtu.be/BxQxsuRjoR8)
## [Optimization's Hidden Gift to Learning: Implicit Regularization](https://youtu.be/gh9vrvLx7Mo #DL-generalization 
1. Nathan Srebro

https://developers.google.com/machine-learning

### L1 vs L2 regression
#Lasso #DIMAP
1. Definition
	1. L1 and L2 are both ways to mitigate overfitting by controlling the value of the feature weights. 
2. Intuition
	1. L1 aggressively drives some features to zero while L2 spreads the penality uniformly among features. Geometrically, the corners of L1 unit 'ball' are most likely to intersect with objective function, while for L2, the intersection can happen at any point.
	2. This is especially true in high dimension, where corners of L1 'sticks out'. See [[02_study/03-Machine-Learning/Cornell-CS4780-F18/Notes#^l1]]
3. Mechanics & Math
	1. L1  $\operatorname{argmin}_w \text{NLL}(w) + \lambda ||w_i||_1$, while L2 $\operatorname{argmin}_w \text{NLL}(w) + \lambda ||w_i||_2$.
	2. ==Bayesian view==: L1 assumes prior $p(w_j) \sim \exp(-\lambda |w_j|)$, while L2 assumes normal prior. The former has a spike around 0, which promotes sparsity.
4. Assumptions/Limitations
	1. L1 is great for feature selection, which might be detrimantal when multicollinearity, e.g. `age` and `year of experience` presents--it could arbitrarily drop one of them while both are important.
5. Practical Application
	1. "In practice, I’d use L1 if I’m working with a high-dimensional dataset, such as genetics or text, where I know most features are noise and I want a simpler model. I’d default to L2 for standard predictive tasks to prevent my neural network or regression model from memorising the training data." [source](https://medium.com/data-science-collective/how-i-prepared-for-ml-theory-interviews-at-top-tech-companies-336414e87cae)