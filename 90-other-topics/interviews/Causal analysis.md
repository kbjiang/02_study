1. [Causal Inference - EXPLAINED!](https://youtu.be/Od6oAz1Op2k) by CodeEmporium
	1. Logic
		1. Randomized Controlled Test (RCT) is good but not always feasible
			1. Why controlled? Because of *confounders*! 
			2. When designing RCT, we can deliberately minimize confounders, i.e., other factors are equivalent between two treatment and control groups
		2. Therefore we need to rely on existing observed data, where we have no control over confounders
			1. Causal inference *simulates* RCT with observed data by dealing with *confounder* and *selection bias*.
	2. Assumptions
2. MIT 6.S897, lecture [14](https://youtu.be/gRkUhg9Wb-I) and 15
	1. https://youtu.be/gRkUhg9Wb-I
3. *What if* Chapter 1
	1. "association is defined by a different risk in *two disjoint subsets* of the population determined by the *individuals’ actual treatment* value (A = 1 or A = 0), whereas causation is defined by a different risk in the *same population* under two *different treatment values* (a = 1 or a = 0)." ![[Pasted image 20260214092217.png]]

## Causal Inference for the Brave and True
### 01 - Introduction to causality
1. $Y$ is the outcome, $Y_0 = Y(0)$ is the *potential outcome* without treatment, similar for $Y_1$.
	1. $Y$ can be dependent on $T$ (casual), while $Y_0$ and $Y_1$ are not (no bias). 
2. ATE and ATT 
	1. $ATE = \mathbf{E}[Y_{1} - Y_{0}],\ ATT = \mathbf{E}[Y_{1} - Y_{0}|T=1]$
		1. $\text{Association}=\mathbf{E}[Y|T=1] - \mathbf{E}[Y|T=0] = \mathbf{E}[Y_1|T=1] - \mathbf{E}[Y_0|T=0]$. See how I cannot drop the "$|T=1/0$" in the last equation--association is sample based, i.e., all factual.
	2. $ATE$ and $ATT$ contain counterfactual terms. 
### 02 - Randomized Experiments
1. $Y$ is dependent on $T$, while $Y_0, Y_1$ may not. ???
2. I.e., $\mathbf{E}[Y_0|T=0] = \mathbf{E}[Y_0|T=1]$ and similar for $Y_1$ ?. Here $T=1$ is the subset of population, and $Y_0|T=1$ is the counterfactual for this subset
3. When Randomized, $ATE$ becomes difference of sample means.
4. Randomization is on $T$! This will sever the link between $T$ and any confounders.
### 04 - Graphical Causal Models
1. Breaking down the population by feature $X$ is called *controlling for* or *conditioning on* $X$.
	1. E.g., break up severe and not severe patients and then treatment *within* each group
	2. non-conditioning means NOT taking this variable into account
2. ==Conditional independence==
	1. The whole point is to make treatment *==ASSIGNMENT==* conditionally independent of the potential outcomes so that causal analysis can be done.
		1. because ==*dependency = bias*==!
	2. ???This reads "$T$ becomes conditionally independent of the potential outcome $Y_0$ (same for $Y_1$) by conditioning on $X$". $$Y_0 \perp T|X \Leftrightarrow \mathbf{E}[Y_{0}|T=0]=\mathbf{E}[Y_{0}|T=1]$$
	3. Three rules to (un)block dependency. See more [here](https://ai.stanford.edu/~paskin/gm-short-course/lec2.pdf)     
		1. intermediary, fork, collider ![[Pasted image 20260215080114.png|500]]
3. Confounder vs Selection bias
	1. *Common cause to the treatment and the outcome*. 
		1. When not controlled, manifests as *confounding bias*, i.e., $\mathbf{E}[Y_0|T=1] - \mathbf{E}[Y_0|T=0] \ne 0$.
	2. *Selection bias* arises when we control for *more variables than we should*.
	3. ==Exercise explaining the two situations in words and in terms of bias!== ![[Pasted image 20260215085940.png]]
## 05 - The unreasonable effectiveness of linear regression
### All you need is regression
1. Makes sense for randomized experiments.
### Regression Theory
1. first two section assumes random data and the results are just beautiful. 
	1. When random, $ATE$ reduces to difference of *sample means* (therefore only factual data) of treated/untreated groups
### Non-random data
> Why does regression work?
1. multivariate to bivariate
	1. "if we include IQ in our model, then  becomes the return of an additional year of education while keeping IQ fixed." how ???
### Omitted variable bias (OVB) or confounding bias
1. the additional term in estimate for $\kappa$.
	1. blah blah...
2. Causal graphical explanation of RCT and Regression
	1. Regression, on the other hand, does so by comparing the effect of  while maintaining the confounder  set to a fixed level. ???
### 06 - Grouped and Dummy Regression
1. Use `weighted OLS` for grouped data
	1. different groups have different variance because of difference in size, i.e., heteroskedasticity.
2. Great explanation of the coefficients and the inclusion of *interaction* term
	1. "It tells us how much IQ increases the effect of graduating 12th grade"
	2. "interaction term allows the treatment effect to change by levels of the features"
3. Dummify `educ`
	1. "Treating education as a category, we no longer restrict the effect of education to a single parameter. Instead, we allow each year of education to have its own distinct impact. By doing so, we gain flexibility, since the effect of education is no longer parametric."
		1. **Non-parametric**: Each education level gets its own free coefficient. No functional form is imposed. The model just computes the mean wage at each education level independently.
		2. So "non-parametric" here doesn't mean "no parameters." It means **no assumption about the shape** of how education maps to wages. The model is just a lookup table of group means — it doesn't assume linearity, monotonicity, or any other structure.
	2. The trade-off the author mentions is exactly this: you gain flexibility (any shape is possible) but *lose statistical power* (many more parameters to estimate, so bigger standard errors and larger p-values for each one).
4. Dummify `educ` and `IQ`
	1. the coef for `educ` becomes a weighted average of the effect in each `IQ_bins` group
	2. the group weight is proportional to *treatment variance* (think entropy) in that group
### 13 - Difference-in-Difference
> Diff-in-diff is commonly used to assess the effect of *macro* interventions, i.e., a *period before and after the intervention* and you wish to untangle the impact of the intervention from a general trend.
1. Great explanation of the coefficients and the inclusion of interaction term
	1. Why regression works? Read 5 again
	2.  The interaction term $\beta_3$ is exactly the **difference-in-differences**: it lets the POA-July cell deviate from what you'd predict by just adding the city effect and the time effect. That deviation is the treatment effect. See [[temp-did]] for more detail.
## Code
1. multivariate OLS
2. DiD
	1. poa\*jul  →  poa + jul + poa:jul`
3. `pd.assign`
	1. new copy with new cols
4. `groupby` and `agg`
	```python
	wage.assign(count=1).groupby("educ").agg({"lhwage": "mean", "count":"sum"}).reset_index()
	```
## MISC
1. Simpson's paradox
	1. Subgroup trends differ from overall trend
2. ML vs Causal ML![[Pasted image 20260214065248.png]]