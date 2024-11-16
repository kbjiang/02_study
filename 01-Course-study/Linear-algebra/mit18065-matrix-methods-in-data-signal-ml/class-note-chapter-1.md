## I.1 Multiplication with columns of $A$
1. Five basic problems on page 1 
2. $A = C R$ where  $C$ are columns in $A$ and $R$ is the companion matrix. 
	1. Turns out rows in $R$ are the *basis* of $A$'s *row* space.
	2. Running example	
	$$\begin{align*}
	A &= \left[\begin{array}{ccc} 2 & 1 & 3\\ 3 & 1 & 4\\ 5 & 7 & 12 \end{array}\right] \\
	&= \left[\begin{array}{cc} 2 & 1 \\ 3 & 1\\ 5 & 7 \end{array}\right]
	 \left[\begin{array}[ccc] 1 & 0 & 1\\ 0 & 1 & 1 \end{array}\right]	= CR \\
	\end{align*}
	$$
## I.2 Matrix-matrix multiplication
1. Factoring a matrix $A$
	1. Why: reveals the most important pieces (rank-1 components) of it
	2. Five types
	3. Motivated by understanding matrix multiplication as the sum of a series of rank-1 outer products. (video ~22:00)
3. Fundamental picture
	1. dimension, rank  and geometrical relation of each
	2. how to go from one space to another (Figure 1.6 p31)
		1. ![[Pasted image 20241101063341.png|600]]
		2. the nullspace of $A^{T}$ is less useful imo.

## I.5 Orthogonal Matrices and Subspaces
1. $Q$ with orthonormal columns: $Q^{T} Q = I$ 
	1. only when $Q$ is square do we also have $Q Q^{T} = I$ 


## I.6 Eigenvalues and Eigenvectors
1. Eigenvalues and eigenvectors
	1. $\sum \lambda_i == \text{trace}(A)$ and $\prod \lambda_{i} == \text{det}(A)$.
	2. $A$ and $A^{-1}$ have the same eigenvectors and reciprocal eigenvalues. Think of undoing previous stretching.
	3. $A$ and $A^{T}$ have the same eigenvalues. 
	4. One eigenvalue may have *multiple* eigenvectors. Just think in terms of nullspace of $A-\lambda I$.
		1. E.g., identity matrix has $\lambda = 1$ and $Ax=x,\ \forall x$.  
		2. See *Quick proofs* on P44.
2. similar matrices ($B = MAM^{-1}$) have same eigenvalues
	1. $A = X \Lambda X^{-1}$ where $X$ is the matrix of eigenvectors and $\Lambda$ has eigenvalues on its diagonal.
	2. if $A$ is $S$ (symmetric), $S = Q \Lambda Q^{T}$.
3. P40. shortage of eigenvectors & rank of $A-\lambda I$ & zero eigenvalue
	1. Eigenvector with zero eigenvalue lives in nullspace of $A$
	2. P42, ex 15. $A = [1\  1; 3\  3]$  has rank 1 of $A-\lambda I$. But it has multiple eigenvalues, so no shortage of eigenvectors, but zero eigenvalue.
4. Nice visualization/application [[02_study/01-Course-study/Linear-algebra/online-resources#^eigens-visual]]

## I.7 Symmetric Positive Definite Matrices
1. Why 2nd derivatives being positive definite means minimum? What does eigenvalue mean in this case? P49 bottom.
2. Symmetric matrix $S$ is a special case of $A$
	1. real eigenvalues
	2. *can chose* orthogonal eigenvectors, therefore a basis, therefore decomposition
3. SPDM is a special case of $S$

## I.8
1. Singular values $\Leftrightarrow$ eigenvalues
2. $A^T A v = \sigma^2 v,\ A A^T u = \sigma^2 u$.
3. The geometry of SVD
	1. When see *orthogonal* matrix, think of *rotation*
4. Other references
	1. [Steve Brunton](https://youtu.be/nbBvuuNVfco)

## I.9
1. PCA
	1. is *linear*
	2. Data matrix $A$ (centered at zero mean) $\Rightarrow$ Covariance matrix $S=AA^T/(n-1) \Rightarrow$  P76, 77
		1. It's NOT $A^T A$ because each row is a data point and we should have $S$ of shape $m\times m$.
		2. It represents the *joint distribution* of the columns, e.g., one col for height the other for weight. 
		3. Think of vector $(h_i, w_i)$ and the diagonal/off-diagonal entries of $S$ relates to variances/covariances respectively. 
			1. E.g., if weight and height are highly correlated, you'd have large off-diagonal entries
2. PCA vs Least Square
	1. PCA always centered (zero mean) and look for error along the direction of singular vectors. Eqn 20 on P77.
	2. ![[PCA-vs-LS.excalidraw|800]]


## I. 11
1. Norm is a *function* that calculates the "size" of a nonzero vector $v$
	1. *Norm does NOT affect the vector nor their operation.* It's just a useful function one can define, like a metric.
	2. Must have $\|v\|\ge 0$, $\| c v\|= |c| \|v\|$ and  $\| v + w\| \le \|v\| +\|w\|$.
		1. Counterexample is $p=1/2$, see P88b.
		2. Norm defines a space, e.g., $\forall v$ s.t. the $\mathscr{l}^p$ properties are satisfied, we say $v$ is in $\mathscr{l}^p$ space.
	3. Only $\mathscr{l}^2$ makes sense in ordinary geometry, via inner product and angles between vectors. P90t.
		1. As a result, its unit length is isotropic as we experience in daily life.
2. Norm for matrices
	1. *Norm does NOT affect the vector nor their operation.* It's just a useful function one can define, like a metric.
	2. Must have $\|v\|\ge 0$, $\| c v\|= |c| \|v\|$ and  $\| v + w\| \le \|v\| +\|w\|$.
		1. $\|A\|=\underset{v\neq 0}{\text{max}} \frac{\|Av\|}{\|v\|}$ leads to $|AB| \le \|A\|\|B\|$.
	3. Frobenius is $\mathscr{l}^2$ of vectors, but not matrices?
3. Norms matter in optimization problem
	1. Different norms lead to different solutions
	2. E.g., $\mathscr{l}^1$ is good cause it is sparse. Bottom of P89.