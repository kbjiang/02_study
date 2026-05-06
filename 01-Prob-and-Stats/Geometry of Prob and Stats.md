## Linear Regression Done Right (again, geometrically)
Date of publish: April 2025
Authors: Math with Ming
Tags: #statistics #linear_regression  

Video link: https://youtu.be/R-8B2S4KxjI?si=d98pLlS9mRam3MSQ
Paper link: 
Related: 
### Main results
1. visualization of three data points $(x_i, y_i)$ and noise $\varepsilon_i$. ![[Pasted image 20260504063918.png]]
	1. $\pmb{\mu} = \beta_0\mathbf{1} + \beta_1\mathbf{x}$ is the modeled part of $\mathbf{y}$, which is on the plane spanned $\mathcal{M}$ by $\mathbf{1}$ and $\mathbf{x}$. 
	2. Noise $\pmb{\varepsilon}$ follows $\mathcal{N}(0, 1)$ therefore forms a unit sphere around $\pmb{\mu}$.
		1. when doing regression, we get $\hat\beta_i$ by projecting $\mathbf{Y}$ onto $\mathcal{M}$, which will be different from true values $\beta_i$. ![[Pasted image 20260506073721.png]]
2. Connection with statistics: hypotheses testing
	1. Is there a linear relationship between $x$ and $y$? I.e. $H_0: \pmb{\mu} \in \mathbb{1}$ vs $H_1: \pmb{\mu} \in \mathcal{M}$.
### Misc
1. [Degrees of Freedom, Actually Explained - The Geometry of Statistics](https://youtu.be/VDlnuO96p58?list=TLPQMDMwMjIwMjbJ-Nrl2sTgmg) Similar


## References
1. https://www.youtube.com/@MathwithMing Very good channel on Math, with a lot of intuition
2. https://youtube.com/playlist?list=PLmtbsGjqSdcbF261LACoVqibOFCGQT3GI&si=c9dNljqEkkCtQJMz A nice playlist on DOF
