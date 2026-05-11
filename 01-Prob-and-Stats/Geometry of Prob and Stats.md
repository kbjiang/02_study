## Linear Regression Done Right (again, geometrically)
Date of publish: April 2025
Authors: Math with Ming
Tags: #statistics #linear_regression  

Video link: https://youtu.be/R-8B2S4KxjI?si=d98pLlS9mRam3MSQ
Paper link: 
Related: 
### Main results
1. visualization of three data points $(x_i, y_i)$ and noise $\varepsilon_i$. ![[Pasted image 20260504063918.png|600]]
	1. $\pmb{\mu} = \beta_0\mathbf{1} + \beta_1\mathbf{x}$ is the modeled part of $\mathbf{y}$, which is on the plane spanned $\mathcal{M}$ by $\mathbf{1}$ and $\mathbf{x}$. 
	2. Noise $\pmb{\varepsilon}$ follows $\mathcal{N}(0, 1)$ therefore forms a unit sphere around $\pmb{\mu}$.
		1. when doing regression, we get $\hat\beta_i$ by projecting $\mathbf{Y}$ onto $\mathcal{M}$, which will be different from true values $\beta_i$. ![[Pasted image 20260506073721.png]]
2. Connection with statistics: hypotheses testing
	1. Is there a linear relationship between $x$ and $y$? I.e. $H_0: \pmb{\mu} \in \mathbb{1}$ vs $H_1: \pmb{\mu} \in \mathcal{M}$.
		1. In words, rejecting $H_0$ means there is a nontrivial linear relationship between $x$ and $y$.
	2. Identify $Y$'s projections onto $\mathcal{M}$ and $\mathbf{1}$,  as well as their interpretations as residuals i.r.t what model. ![[Pasted image 20260511075654.png|600]]
		1. $\overrightarrow{AC}$ is the residual when trend is allowed ($P_{\mathcal{M}}^{\perp} \varepsilon$), while $\overrightarrow{AD}$ is the residual when trend is not allowed ($P_{\mathbb{1}}^{\perp} \varepsilon$).
			
			| Goodness of Fit | Sum of Squares                                      | DoF of residual | Distribution            |
			| --------------- | --------------------------------------------------- | --------------- | ----------------------- |
			| Reduced model   | $\|P_1^\perp Y\|^2$                                 | 2               | $\sigma^2 \chi^2_{(2)}$ |
			| Full model      | $\|P_{\mathcal{M}}^\perp Y\|^2$                     | 1               | $\sigma^2 \chi^2_{(1)}$ |
			| Difference      | $\|P_1^\perp Y\|^2 - \|P_{\mathcal{M}}^\perp Y\|^2$ | 1               | $\sigma^2 \chi^2_{(1)}$ |
	3.  **Decision rule using F statistics.**  
		1. Under $H_0 : \mu \in \mathbf{1}$, $$ F := \frac{\|P_1^\perp Y\|^2 - \|P_{\mathcal{M}}^\perp Y\|^2}{\|P_{\mathcal{M}}^\perp Y\|^2} = \frac{DC^2}{AC^2} \overset{d}{=} \frac{U^2}{V^2}, \qquad U, V \overset{\mathrm{iid}}{\sim} N(0,1). $$ Moreover $$\frac{U}{V} \sim \mathrm{Cauchy}.$$The distribution of a squared standard Cauchy RV is $F_{1,1}$. To find its CDF, let $X \sim \mathrm{Cauchy}$ and define $Y = X^2$. Then for $-\infty < x < \infty$, $$ P(X \le x) = \frac{1}{\pi}\arctan x + \frac{1}{2}. $$
		2. With significance level $\alpha \in (0,1)$, reject $H_0$ if $$ F > F_{1,1;\alpha} = \tan^2\!\left(\frac{\pi(1-\alpha)}{2}\right) \overset{\text{$\alpha$=0.05}}{\approx} 161 $$ I.e., reject $H_0$ only when $\triangle{ACD}$ is extremely flat. 
	4. Starting with $\pmb{\varepsilon}=\overrightarrow{BA}=\overrightarrow{BD}+\overrightarrow{DC} + \overrightarrow{CA} \sim \mathcal{N(0, \sigma^2\mathbf{I}_3)}$, and once proved mutually perpendicular, we have $\overrightarrow{BD},\overrightarrow{DC},\overrightarrow{CA} \sim \mathcal{N(0, \sigma^2})$. (WHY?) 
### Misc
1. [Degrees of Freedom, Actually Explained - The Geometry of Statistics](https://youtu.be/VDlnuO96p58?list=TLPQMDMwMjIwMjbJ-Nrl2sTgmg) Similar


## References
1. https://www.youtube.com/@MathwithMing Very good channel on Math, with a lot of intuition
2. https://youtube.com/playlist?list=PLmtbsGjqSdcbF261LACoVqibOFCGQT3GI&si=c9dNljqEkkCtQJMz A nice playlist on DOF
3. Rice, chapter 14.
