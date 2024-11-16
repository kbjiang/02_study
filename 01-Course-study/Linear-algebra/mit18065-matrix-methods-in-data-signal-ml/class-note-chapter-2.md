## II. 2 Least Squares: Four ways
1. Least square
	1. Find $\underset{x}{\text{min}}\|Ax-b\|^2$ , note $x$ is in $C(A^T)$.
2. Pseudoinverse of $A$: $A^{+}$
	1. $A^+ = A^{-1}$ if $A$ is invertible
		1. $A$ has to be *square* to have $AA^{-1}=A^{-1}A$
		2. any $A$ can have $A^+$ with accommodating shape
	2. $A^+A=I$ in $C(A)$ and $AA^+=I$ in $C(A^T)$
		1. $A^+A = \left[\begin{array}{cc}  I & 0 \\ 0 & 0  \\ \end{array}\right] \begin{array}{c} \text{row space} \\	\text{nullspace} \\ \end{array}$
	3. $A^+$ share the same nullspace as $A^T$
	4. $A^+$ solves LS in one step

## Notes to myself
skipped most of *II.1 Numeric Linear Algebra*