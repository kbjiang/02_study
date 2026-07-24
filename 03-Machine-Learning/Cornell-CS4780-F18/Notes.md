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
## My takeaway
1. What was the main idea?  

  

2. What surprised me?  

  

3. What does this connect to?

# Lecture 4 "Curse of Dimensionality / Perceptron"
## Notes
### Curse of Dimensionality
1. random points live close to edges; in other words, no points are close to each other, closeness does not apply
2. why KNN still works
	1. real data intrinsic low dim: subspace/manifold
		1. e.g., an image of a face is 10k pixel, but a face may only need ~20 features to describe
	2. manifold def: locally Euclidean distance is valid
	3. this is also why PCA and dimension reduction is important
		1. KNN is very slow in high dimensional $O(n \times d)$
3. Question
	1. in his Curse of dim demo, why the peak in the middle?

### Perceptron
1. In high-dimensional space, data points are almost always far away from each other therefore separable by a hyperplane.