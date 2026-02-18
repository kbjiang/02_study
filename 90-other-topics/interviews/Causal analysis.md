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
3. Domain knowledge
	1. Just from the following data frame, there's no way to tell if `sex` is a confounder that causes the skewed distribution of both `drug` and `days`, or it is just correlated with them. The causation has to come from *domain knowledge*.
		```python
		drug_example = pd.DataFrame(dict(
		    sex= ["M","M","M","M","M","M", "W","W","W","W"],
		    drug=[1,1,1,1,1,0,  1,0,1,0],
		    days=[5,5,5,5,5,8,  2,4,2,4]
		))
		```
### 02 - Randomized Experiments
1. Under randomization, $Y_0, Y_1$ are not, i.e. $(Y_0, Y_1) \perp T$. This is ==independence==.
	1. Here $T=1$ is the subset of population, and $Y_0|T=1$ is the counterfactual for this subset
	2. No matter which subset, the potential outcomes are the same  $$\begin{align} E[Y_0|T=0] - E[Y_0|T=1] =0\\
	E[Y_1|T=0]-E[Y_1|T=1]=0 \end{align}$$
	3. *==TWO INTERPRETATIONS==*
		1. Treatment and control groups are comparable
			1. This means no EXPLICIT confounders, i.e., some missing feature that makes the two groups not comparable. E.g., ?? 
			2. To remedy, just include that feature
		2. Or that knowing the treatment assignment doesn’t give any information on how the outcome was previous to the treatment
			1. This means treatment was not designed fairly, and the identity of the groups implicitly introduced confounding. We need to MAKE IT EXPLICIT by including the identity dummies. This is exactly what happened in Panel data and Fixed Effects!
2. Under randomization, $Y$ is dependent on $T$. *This is the arrow between treatment and outcome in causal graph.*
	1. This is also called association. Here NO counterfactual data! $$\mathbf{E}[Y|T=1] - \mathbf{E}[Y|T=0] = \mathbf{E}[Y_1|T=1] - \mathbf{E}[Y_0|T=0] \ne 0$$
3. When Randomized, $ATE$ equals the difference of sample means.
4. Randomization is on $T$! This will sever the link between $T$ and any confounders.
### 04 - Graphical Causal Models
1. Breaking down the population by feature $X$ is called *controlling for* or *conditioning on* $X$.
	1. E.g., break up severe and not severe patients and then treatment *within* each group
	2. non-conditioning means NOT taking this variable into account
2. ==Conditional independence==
	1. The whole point is to make treatment *==ASSIGNMENT==* conditionally independent of the potential outcomes so that causal analysis can be done.
		1. because ==*dependency = bias*==!
	2. This reads "$T$ becomes conditionally independent of the potential outcome $Y_0$ (same for $Y_1$???) by conditioning on $X$". $$Y_0 \perp T|X \Leftrightarrow \mathbf{E}[Y_{0}|T=0]=\mathbf{E}[Y_{0}|T=1]$$
	3. Three rules to (un)block dependency. See more [here](https://ai.stanford.edu/~paskin/gm-short-course/lec2.pdf)     
		1. intermediary, fork, collider ![[Pasted image 20260215080114.png|500]]
3. Confounder vs Selection bias
	1. *Common cause to the treatment and the outcome*. 
		1. When not controlled, manifests as *confounding bias*, i.e., $\mathbf{E}[Y_0|T=1] - \mathbf{E}[Y_0|T=0] \ne 0$.
	2. *Selection bias* arises when we control for *more variables than we should*.
	3. ==Exercise explaining the two situations in words and in terms of bias!== ![[Pasted image 20260215085940.png]]
4. Two layers
	1. **Layer 1 (causal):** You draw the arrows based on domain knowledge. This is an untestable assumption.
	2. **Layer 2 (statistical):** Given those arrows, the d-separation rules (what the crash course teaches) tell you which variables are (conditionally) independent. These are testable predictions.
## 05 - The unreasonable effectiveness of linear regression
### All you need is regression
1. Makes sense for randomized experiments.
### Regression Theory
1. first two section assumes random data and the results are just beautiful. 
	1. When random, $ATE$ reduces to difference of *sample means* (therefore only factual data) of treated/untreated groups
2. see intuition in [[temp-ols]]
3. How is confounder handled
	1. RCT correct for confounders by severing the connection between them and the *treatment*
	2. Regression correct for confounders by fixing their value when calculating the effect of treatment. ![[Pasted image 20260218060624.png|500]]
### TBD
1. **Include confounders** → prevents over-crediting the treatment (some of the apparent effect is really due to the confounder) 
2. **Exclude mediators** → prevents under-crediting the treatment (blocking a real causal pathway would hide part of the true effect)
### Non-random data
> Why does regression work?
1. multivariate to bivariate
	1. "if we include IQ in our model, then  becomes the return of an additional year of education while keeping IQ fixed." how ???
		1. $X_1$: regress it on $X_2$, take the residual. Now both $\tilde{Y}$ and $\tilde{X}_1$ are "clean" of $X_2$'s influence. When you regress $\tilde{Y}$ on $\tilde{X}_1$, you get the effect of $X_1$ **holding $X_2$ constant** — which is exactly what multiple regression does.
### Omitted variable bias (OVB) or confounding bias
1. the additional term in estimate for $\kappa$ proves association with both treatment and outcome.
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
### 07 - Beyond Confounders
1. Confounders
	1. should be included! Otherwise wrong $\beta$.
2. Good controls
	1. A good predictor explain away a lot of variance of the ==*outcome*.== When included in the model, it becomes a good control. 
	2. Inclusion of good predictors are optional but recommended. It make the estimate of causal effect more significant (i.e., smaller std err of $\beta$.)
3. Harmful controls
	1. Variables predict the treatment but not the outcome. They reduce the variability of the treatment, making it hard to find the causal effect (i.e., greater std err of $\beta$.)
	2. They are ancestors of the treatment?
	3. Should not include in regression
4. Selection bias
	1. variables in the causal path from treatment to outcome (mediator), or common effect of treatment and outcome.
		1. They are children of the treatment
	2. should be excluded. Otherwise some of the variability of treatment would be explained by them therefore wrong $\beta$.
		1. E.g., when a mediator is fixed, treatment can only vary on a slice of data.
### 10 - Matching
1. find the similar subsets in treated and untreated and calculate ATE between them.
### 11 - Propensity score
1. Instead of control $(Y_1, Y_0) \perp T|X$ , use $(Y_1, Y_0) \perp T|e(x)$, where $e(x)$ is the propensity function/score which maps $X$ into $T$.
	1. So that we do not need to condition on entire $X$, which can have a lot of variables, but just one variable, $e(x)$.
	2. Notice propensity function is applied on each $x \in X$. 
		1. $e(x)$ is learnt via ML, such as logistic regression, to predict $P(T|X=x)$.
		2. this model predic
	3. (inverse of ) Propensity score is used to weight different $x$ when calculating ATE.
### 13 - Difference-in-Difference #DiD 
> Diff-in-diff is commonly used to assess the effect of *macro* interventions, i.e., a *period before and after the intervention* and you wish to untangle the impact of the intervention from a general trend.
1. Great explanation of the coefficients and the inclusion of interaction term
	1. Why regression works? Read 5 again
	2.  The interaction term $\beta_3$ is exactly the **difference-in-differences**: it lets the POA-July cell deviate from what you'd predict by just adding the city effect and the time effect. That deviation is the treatment effect. See [[temp-did]] for more detail.
2. The FWL mapping (chap 14) is:
	 1. . **$X_2$** = {`POA`, `Jul`} — the entity fixed effect (1 city dummy) + time fixed effect (1 period dummy). These get "partialled out."
	 2. **$X_1$** = {`POA × Jul`} — the interaction term, i.e., the treatment indicator. $\beta_3$ is the coefficient we care about (the DiD causal estimate).
### 14 - Panel Data and Fixed Effects
> - Panel data is a data structure applicable to a lot of problems; its structure makes it possible to control *time-invariant* confounders (measured or not!)
> - FE is one method to exploit this leverage
> - DiD is a FE with 2 groups and 2 periods
1. A *panel* is when we have repeated observations of the same unit over multiple periods of time.
	1. this structure allows us to deal with *fixed but unmeasurable*  confounding variables such as a person's beauty (vs income)
2. Instead of conditional un-confoundedness/independence, 
3. The Frisch-Waugh-Lovell (FWL) theorem
	1. **Step 1:** Regress $Y$ (lwage) on $X_2$ (individual dummies) → this just computes each person's mean $\bar{Y}_i$ 
	2. **Step 2:** Regress $X_1$ (married, expersq, etc.) on $X_2$ (individual dummies) → this just computes each person's mean $\bar{X}_i$
4. Exclude "occupation" as a potential mediator!
	1. Controlling for "occupation" would've taken away some effect because of "marriage".
5. Understand the examples
	1. **Wage example:** Thousands of individuals → demeaning *trick* is used because creating thousands of dummy columns is impractical.
	2. **City example:** Only 4 cities → `C(city)` dummies are included explicitly because it's cheap and lets you visualize the separate intercepts (the parallel lines in the plot).
	3. The notebook uses both approaches for pedagogical reasons: the explicit dummies make the "one regression line per entity" intuition visually clear, while the demeaning approach shows the scalable computation. Same math, different implementation choices based on the number of entities.
6. Time effects
	1. About effects that are changing with time but same for all entities. E.g., inflation.
	2. Partialling out both means you're looking at variation that's neither explained by "which person" nor by "which year" — only the truly idiosyncratic within-person, within-year variation remains.
	3. See [[temp-time-effects]]
7. ==???== [[temp-panel-data]]
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