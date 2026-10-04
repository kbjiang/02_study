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
	2. This is especially true in high dimension, where corners of L1 'sticks out'. 
3. Mechanics & Math
	1. L1  $\operatorname{argmin}_w \text{NLL}(w) + \lambda ||w_i||_1$, while L2 $\operatorname{argmin}_w \text{NLL}(w) + \lambda ||w_i||_2$.
	2. ==Bayesian view==: L1 assumes $p(w_j) \sim \exp(-\lambda |w_j|)$, while L2 assumes normal prior. The former has a spike around 0.  