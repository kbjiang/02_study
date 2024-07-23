Date: 2024-05-08
Course:
Chapter: 
Lecture: 
Topic: #Probability #Bayesian #Inference 

## Lectures
### Lecture 21-22: Bayesian Statistical Inference
1. The goal is to find estimators (*functions of the observations*) that have some desirable properties, such as small estimation error, for the unknown $\Theta$ (Bayesian) or $\theta$ (Classical).
2. Bayesian treats the unknown as a R.V. It does *not* mean nature is uncertain, but our *prior* belief of the distribution of the unknown.
	1. ![[Pasted image 20240529073352.png|500]]
	2. Note in both cases, estimator $\hat{\Theta}$ is a R.V. 
	3. Estimators are *functions of observations $X$*
3. Maximum a Posteriori Probability (MAP) estimator
	1. Defined as the R.V. $\hat{\Theta}=g(X)$. In natural language, some decision rule. 
	2. It is *optimal* for *any* given $x$ in terms of "making a correct decision" (p420 footnote), which means $\hat{\theta}=\theta_{\text{True}}$. By definition, MAP selects the $\theta$ with most probability of being $\theta_{\text{True}}$ according to posteriori dist. 
	3. This is the "greedy" one.
4. Conditional Expectation estimator or Least Mean Square (LMS) estimator
	1. it is unbiased, i.e. $E[\Theta - E[\Theta|X]]=0$
		1. This is over PDF $f_{\Theta, X}$
		2. Does NOT mean every experiment has zero bias; only on AVERAGE over many experiments ($X$)
	2. It is *optimal* for *any* given $x$ in *terms* of minimizing mean square error.
		1. see bottom plot in Figures 8.10, 8.11 and 8.12, the LMS minimize MSE at every $x$. ![[Pasted image 20240711070113.png|600]]
	3. I.e., $\hat{\theta} = \mathbf{E}[\theta|X=x]$ minimizes $\mathbf{E}[(\Theta - \hat{\theta})^2|X=x]$ and similarly, $\hat{\Theta} = \mathbf{E}[\Theta|X]$ and $\mathbf{E}[(\theta - \hat{\theta})^2]$.
		1. Former is *number*, while latter is *R.V.*.
		2. In following screenshot. First point is average over $f_{\tilde{\Theta}}$, second is over $f_{\tilde{\Theta}|X}$, third is over $f_{\tilde{\Theta}}$ (where $\tilde{\Theta}\coloneqq \Theta - \hat{\Theta}$) or equivalently $f_{\Theta, X}$, and it was obtained by using Law of Iterated Expectation. 
			1. ![[ Pasted image 20240617084538.png|600]]
5. Linear Least Mean Square (LMS) estimator
	1. Nice intuition at the beginning of P439
	2. When distributions are normal, Linear LMS and LMS coincide. "Thus, there are two alternative perspectives on linear LMS estimation: either as a computational shortcut (avoid the evaluation of a possibly complicated formula for $\mathbf{E}[\Theta|X]$, or as a model simplification (replace less tractable distributions by normal ones) ."
	3. Transformation of observation $Y=h(X)$ does not affect LMS but Linear LMS, since the latter assume that linearity is a good proximation. Therefore right transformation, e.g. $X$ or $X^2$, is important to Linear LMS.
## My comments
1. hypothesis testing: few possible values for unknown, aims at small *probability* of incorrect decision; think MAP
2. Estimation: aim at a small *estimation error*; think Conditional Expectation Estimator
3. $\tilde{\Theta}, \hat{\Theta}$ are both R.V.s with their own prob. dist. Think of $Z = X + Y$.
	1. Similarly, $\hat{\Theta} - \Theta = 0$ is an equality between R.V.s; the RHS is a trivial R.V.. 
4. posterior => how to report it? => estimator
5. A lot of averages, need to be careful about 
	1. over *which variable* it is averaged.
		1. Total variance: $\text{var}(\Theta) = \text{var}(\mathbf{E}[\Theta|X]) + \mathbf{E}[\text{var}(\Theta|X)]$, the LHS is *unconditional* and a number, RHS arguments are both functions of $X$ basing on $f_{\theta|X}$, then $f_X$.
	2. if it's a calculation of a number (conitional) or a R.V. (unconditional)
6. Why observation $X$ needs to be a R.V. as well?
	1. This is demanded by Bayesian inference. I.e., the joint distribution of $\Theta, X$. In theory, $X$ being deterministic could be a special case.
	2. Shows how $X$ is affected by $\Theta$.
## Interesting/Challenging problems and tricks
1. Example 8.3 and 8.4, P415 & 417. 
	1. The same family of posterior and prior distributions enables *recursive inference*
	2. Beta functions
2. Example 8.2, P414
	1. With one measurement ($x$) the posterior distribution is updated; with multiple measurements ($x_1, ..., x_n$), the range becomes more confined.![[Pasted image 20240708084253.png|400]]
	2. Actually, Examples 8.2, 8.7, 8.12 and 8.15 makes a good sequel. 
3. Problem set 10, 5(d)
	1. $T_1$ and $T_2$ are NOT independent because they both depend on $Q$. In other words, knowing $T_1$ changes belief on $T_2$. However, once $Q$ is given, they become (conditionally) independent.
	2. $\mathbf{E}[T_1 T_2] = \mathbf{E}[\mathbf{E}[T_1 T_2|Q]] = \mathbf{E}[\mathbf{E}[T_1|Q]\ \mathbf{E}[T_2|Q]]$ 