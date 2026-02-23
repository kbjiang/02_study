## Coding round
### Stop:
- Early regression
- Jumping to statistical tests too fast
- Assuming single cause
### Start:
- Metric decomposition
- Segment-first thinking/controlling confounding
- Business implications/Cost
	- more shoe sizes, higher manufacture cost

### Metrics
1. ==guardrail==
2. novelty effect
3. heterogeneity - slice by country, device, user tenure...
#### North Star metric
- Answers: **“Is this feature achieving its primary goal?”**
- You _want this to move in the right direction_
- Used for **success justification**
#### Guardrail metric
- Answers: **“Are we breaking something important?”**
- You _do not want this to regress_
- Used for **safety / veto power**

## Testing
1. Use $\chi^2$ for counts, $t$-test for continuous!
2. borderline p-value, say 0.047
	1. Procedure
		1. interpretation in plain English, to shareholders. Also mention this is weak evidence.
		2. should be interpreted with caution: small change in data/assumptions/design may flip it above 0.05
		3. Mention ==effect size, CI includes zero, low/high risk== of rollout to non-technical audience
	2. replication tests (*for same effect*)
		1. if new p-value 0.03, then the evidence becomes much stronger
		2. if new p-value 0.07, still consistent directionally. Combined with 1st result, together support a real effect, albeit likely small
	3. see [[temp-multi-tests]] for more discussion, including multi-tests for multi-effects.
3. survey response dropped from 4.3 to 3.7 after a new feature was introduced, how do you attribute the cause for this drop? Is this feature harmful?
	1. alternative hypotheses
		1. different cohort
		2. seasonality/trend change
	2. better tests
		1. AB test. Most reliable, also expensive
		2. DiD. Can be realized by *staggered rollout*
	3. Is this feature harmful
		1. even if better tests attribute the drop to the feature, still need to check guardrail metrics to see if this new feature is really harmful
	4. Alternative hypotheses**  
		1. Composition effects (different cohorts responding)  
		2. Seasonality or underlying time trends
	5. Better tests
		1. A/B test: most reliable but expensive and sometimes infeasible
		2. Difference‑in‑Differences using staggered rollout to control for time effects
	6. Is this feature harmful?
		1. Even if causal analysis attributes the rating drop to the feature, that alone doesn’t determine whether it’s harmful  
		2. I’d evaluate guardrail metrics, such as revenue, and cohort‑level impacts to see ==whether the effect represents unacceptable or sustained harm versus a tolerable trade‑off== 
> 	Attribution answers _why the metric moved_; guardrails answer _whether we should act on it_.

## ML
### K-means
1. when not suitable
	1. non-spherical data
	2. outlier
	3. high-dimension
2. How to fix
	1. multiple restart/initialization
	2. choose K: elbow
	3. preprocessing

### Feature reduction
- Stepwise picks variables; PCA changes the basis. 
- Stepwise hurts inference (see below), PCA hurts interpretability.
	- In stepwise, it uses the same data twice: first to _choose_ the model, then to _pretend the model was fixed_ when computing p‑values and CIs.

“While many Maps searches are navigational, search still creates a lot of value for exploratory intents like dining or activities. The mistake is treating all searches the same — category features should be intent‑aware, otherwise they help discovery users but hurt navigational users.”