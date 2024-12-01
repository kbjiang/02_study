## II. 2 Least Squares: Four ways
1. Least square
	1. Find $\underset{x}{\text{min}}\|Ax-b\|^2$ which minimizes the error.
	2. Note that $x$ is not necessarily in $C(A^T)$: it could have components in $\text{null}(A)$ which does not increase the error, only it's norm.
2. Pseudoinverse $A^{+}$: even when $A$ is not invertible!
	1. $A^+ = A^{-1}$ if $A$ is invertible
		1. $A$ has to be *square* to have $AA^{-1}=A^{-1}A$
		2. any $A$ can have $A^+$ with accommodating shape
	2. $A^+$ share the same nullspace as $A^T$
	3. $A^+$ solves LS in one step
	4. Special case: when $A$ have independent columns, i.e., $r=n$.
			1. $A^+ A=1$. No guarantee $AA^+=1$
			2. pseudoinverse and normal equation are equivalent.

## II.4 Randomized Linear Algebra
1. Approximation of $AB$ can be modeled as choosing rank-1 component of $a_i b^{T}_i$ with probability $\{p_{i}\}$.
	1. The lecture is kind of tedious. Just understand Eqn (4) on P149.
		1. Eqn (4) is so to have the correct *expected value*. $s$ is sample size.
		2. Sampling matrix $S$ is not necessary
	2. The point is to find the right sampling $\{p_i\}$ so that the sampled matrix has minimal variance*
		1. Uniform won't work--best to choose large columns/rows more often. That's why norm-squared sampling.
		2. use Lagrangian multiplier to do so

## References
1. https://math.stackexchange.com/questions/2253443/difference-between-least-squares-and-minimum-norm-solution
	1. The confusingly named "solution of minimum norm" is just the particular solution.
2. https://see.stanford.edu/materials/lsoeldsee263/08-min-norm.pdf
## Notes to myself
skipped most of *II.1 Numeric Linear Algebra*
