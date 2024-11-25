## V. inequalities
1. The model of data
	1. $m$ experiments/feature/random variables at once. $n$ is the number of samples. Data $A$ is $n\times m$.
2. Covariance matrices
	1. Think of flipping $m=2$ coins, independently or correlated (glued together). Another e.g., height and weight.
	2. It's always pair-wise between features
		1. $\sigma_{xy}= \underset{i, j}{\sum}p_{ij}(x_i - \mu_x)(y_j - \mu_y)$, $x$ and $y$ are the features whose covariance we are after, while $x_i$ and $y_j$ are the possible values of those features.
		2. Joint probability can be a tensor, i.e., when $m > 2$; covariance matrix is always 2D.
	3. Eqn (3) and (10) on P296-297.
		1. *Sample* (no $p_{ij}$) vs distribution
		2. new data point $\Leftrightarrow$ rank-one component of covariance matrix

## notes to me
1. Go to PCA p76-77 when covariance matrices.