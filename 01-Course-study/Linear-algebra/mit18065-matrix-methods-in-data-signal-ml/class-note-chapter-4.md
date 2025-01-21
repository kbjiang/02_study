## IV. 2. Shift matrices and circulant matrices
1. Eigenvectors of projection matrix $P$ is the Fourier matrix $F$. Eqn. 12 on P216
2. For circulant matrix $C$
	1. It is fully defined by any of its row; in particular, the top row $c$. 
	2. Same eigenvectors as $P$ since $C$ is linear combination of $P$.
	3. Eigenvalues are components of $Fc$. Eq 14 on P217
3. The convolution rule.  P218.
	1. $F(c\circledast d)=Fc.\ast Fd$. 
	2. Proof; using two circulant matrices $C$ and $D$, with top rows $c$ and $d$ respectively.
		1. top row of $CD$ equals to $c\circledast d$. Eq 16 P218.
		2. Eigenvalues of $CD$ is $F(c\circledast d)$. Eq 14 P217.
		3. Eigenvalues of $CD$ is also $Fc.\ast Fd$. Eq 17 P218.
4. Kronecker for 2D convolution
## Notes to myself
1. The point is to do separate Fourier Transform followed by Hadamard product instead of FT of convolution
	1. Former is much faster. $O(N)$ vs $O(N^2)$.
2. Cyclic convolution matrix $\Leftrightarrow$ circulant matrix
3. Convolution in general
	1. It's always on *vectors/functions*! So does Fourier transform.
		1. Think of an image as a $\mathbb{R}^{n^2}$ vector with its $n^2$ components in a square.
	2. polynomials
	3. functions