## III. 1 Changes in $A^{-1}$ from changes in $A$.
1. Perturb $A$, how does $A^{-1}$ change?
	1. Three examples: $(I-uv^T)^{-1}$, $(I-UV^T)^{-1}$ and $(A-UV^T)^{-1}$
2. Two points
	1. rank-$k$ change in $A$ leads to same rank change in $A^{-1}$
	2. Instead of $n\times n$ inverse, just need to do $k \times k$ inverse, where $k$ is the rank of $U$ and $V$. 
3. Least square, adding new data
	1. Solve normal equation leveraging $A$, instead of the perturbed version.

## II.4 Randomized Linear Algebra
1. Approximation of $AB$ can be modeled as choosing rank-1 component of $a_i b^{T}_i$ with probability $\{p_{i}\}$.
	1. The lecture is kind of tedious. Just understand Eqn (4) on P149.
		1. Eqn (4) is so to have the correct *expected value*. $s$ is sample size.
		2. Sampling matrix $S$ is not necessary
	2. The point is to find the right sampling $\{p_i\}$ so that the sampled matrix has minimal variance*
		1. Uniform won't work--best to choose large columns/rows more often. That's why norm-squared sampling.
		2. use Lagrangian multiplier to do so

## Notes to myself
skipped most of *II.1 Numeric Linear Algebra*