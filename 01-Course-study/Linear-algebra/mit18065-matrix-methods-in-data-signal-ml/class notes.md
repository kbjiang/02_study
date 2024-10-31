## Lecture 1: The Column Space of A Contains All Vectors Ax
1. all $AX$ is a column space $C(A)$ 
2. decomposition


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
	1. dimension and rank of each
	2. geometry of each pair
