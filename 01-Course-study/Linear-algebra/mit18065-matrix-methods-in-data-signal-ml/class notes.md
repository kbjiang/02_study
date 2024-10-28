## Lecture 1: The Column Space of A Contains All Vectors Ax
1. all $AX$ is a column space $C(A)$ 
2. decomposition
	0. Running example	
	$$\begin{align*}
	A &= \begin{pmatrix} 2 & 1 & 3\\ 3 & 1 & 4\\ 5 & 7 & 12 \end{pmatrix} \\
	&= \begin{pmatrix} 2 & 1 \\ 3 & 1\\ 5 & 7 \end{pmatrix}
	 \begin{pmatrix} 1 & 0 & 1\\ 0 & 1 & 1 \end{pmatrix}	= CR \\
	&= \begin{pmatrix} 2 & 1 \\ 3 & 1\\ 5 & 7 \end{pmatrix}
	 \begin{pmatrix} 1 & 0 & 1\\ 0 & 1 & 1 \end{pmatrix}	= CUR 
	\end{align*}
	$$
	1. $A = C R$ where  $C$ are columns in $A$ and $R$ is the companion matrix. 
		1. Turns out rows in $R$ are the *basis* of $A$'s *row* space.
	2. $A = C U \tilde{R}$ where  $C$ are columns in $A$ and $\tilde{R}$ are rows in $A$.

## Lecture 2: Multiplying and Factoring Matrices
1. $A=LU$ can be understood as the sum of a series of rank-1 matrices multiplication, one layer at a time. (video ~22:00)