## I.1 Multiplication with columns of $A$
1. Five basic problems on page 1 
2. $A = C R$ where  $C$ are columns in $A$ and $R$ is the companion matrix. 
	1. Turns out rows in $R$ are the *basis* of $A$'s *row* space.
	2. Running example	
	$$\begin{align*}
	A &= \begin{pmatrix} 2 & 1 & 3\\ 3 & 1 & 4\\ 5 & 7 & 12 \end{pmatrix} \\
	&= \begin{pmatrix} 2 & 1 \\ 3 & 1\\ 5 & 7 \end{pmatrix}
	 \begin{pmatrix} 1 & 0 & 1\\ 0 & 1 & 1 \end{pmatrix}	= CR \\
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
2. similar matrices ($B = MAM^{-1}$) have same eigenvalues
	1. $A = X \Lambda X^{-1}$ where $X$ is the matrix of eigenvectors and $\Lambda$ has eigenvalues on its diagonal.
	2. if $A$ is $S$ (symmetric), $S = Q \Lambda Q^{T}$.