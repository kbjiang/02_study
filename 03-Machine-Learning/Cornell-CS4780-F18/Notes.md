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
3. *Inductive* [[#^inductive]]
	1. Inductive is the process of moving from specific facts or observations to a broader, general conclusion, i.e., *generalization*. Deductive is the opposite direction.
	2. No free lunch (NFL) theorem: no inductive bias is universally good. 
## Reference
1. 西瓜书 Section 1.3 ^inductive

# Lecture 4 "Curse of Dimensionality"
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
3. Distance to hyperplane stay small? (WIP, Lect #2 Syllabus)
	1. distance between a point and a $d-1$ hyperplane does not change with $d$. This is why #SVM / #perceptron works.
### No Free Lunch (NFL) Theorem 
# Lecture 5 "Perceptron"
### Perceptron
1. In high-dimensional space, data points are almost always far away from each other therefore separable by a hyperplane.
2. the geometry of adding an additional dimension for bias...
3. overfitting has not become a thing yet
