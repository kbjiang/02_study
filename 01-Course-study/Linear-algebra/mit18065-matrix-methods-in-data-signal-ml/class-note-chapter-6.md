## VI.1 Optimization
1. Convexity
	1. Set $K$
	2. Function $F(x)$ and $x\in K$.
2. It's about iteratively approaching the optimal solution $x^*,\ (x_0, ..., x_k, x_{k+1}...)$.
3. Newton's method vs Gradient descent
	1. Newton error decreases quadratically $(x^*-x)^2$.

## VI.4 Gradient Descent
1. The geometry
	1. $f(x, y)=ax + by$ is a plane consists of all parallel $ax+by=c$ lines.
	2. The gradient $[a, b]$ is the steepest direction, perpendicular to $c$ lines.
		1. The gradient is NOT along the $f$ surface, but in $x\textendash y$ plane!
		2. think of movement in the  $x\textendash y$ plane!
	3. Fig VI.9 on P349. See how we have $90\degree$ zig-zag for exact line search.
2. Momentum
	1. Add momentum to smooth the zig-zag. For quadratic model, the descent is much faster. Eqn (17) P353.

## notes to me
1. Go to PCA p76-77 when covariance matrices.