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
		1. Exact line search: $x_{k+1}$ exactly minimizes $f(x)$ along the direction of $x_k - s \nabla f(x)$.
2. Momentum
	1. Add momentum to achieve faster descent (much less zig-zag). 
	2. Analytical results for quadratic model, Eqn (17) P353.

## VI.5 Stochastic Gradient Descent
1. Two challenges with Full GD; both addressed by SGD.
	1. Too slow
	2. $\nabla_x L=0$ leads to a lot of saddle points. This is replaced by SGD and early stopping
2. SGD
		1. early steps of SGD converge fast and in the right direction
			1. [Intuition](https://youtu.be/k3AiUhwHQ28?list=PLUl4u3cNGP63oMNUHXqIUcrkS2PivhN3k&t=1569) and P362
			2. Too large of a mini batch size leads to a small "region of confusion" and overfitting
		2. we do NOT want to get to the real minimal, which leads to overfitting, therefore early stopping at vicinity.
3. In ML, optimization is the sum of error at each data point, i.e., $f=\text{min}_x\frac{1}{n}\underset{i}{\sum} f_i(x)$.
	1. This is different from classical optimization where we have analytical $f$; here $f$ is dictated by data
	2. ML loss function is not even convex.
	1. Even more so with full GD.
4. Random Kaczmarz
	1. It's the SGD for solving normal equation, 
		1. i.e., solve random components $a_i^T x_k=b_i$ at each step
		2. step size is the projection. See Eqn(3) on P363 and Eqn(4) on P364. Latter is a projection on $x_k - x^*$.
5. References
	1. [[October#^inductive-bias]] argued that SGD converges to solutions with minimum norm, which leads to good generalization.
	2. [[November#UNDERSTANDING DEEP LEARNING REQUIRES RETHINKING GENERALIZATION]] 