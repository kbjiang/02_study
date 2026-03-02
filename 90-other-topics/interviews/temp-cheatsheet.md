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


## 1) Your 60–90 second “modeling approach” spine (use every time)
When they give you a product problem, start with this structure (you can literally say it out loud):
**A. Clarify goal + decision**
- “What decision are we enabling: ranking, classification, forecasting, policy/targeting, or automation?”
- “What’s the success metric + guardrails (latency/cost/fairness/safety)?”
**B. Define prediction target + unit**
- “Unit of prediction: user, session, account, ticket, document…”
- “Horizon: real-time vs next day vs next month”
**C. Data generating process**
- “Do we have labels? Are they delayed/noisy? Any leakage risks?”
- “Is intervention involved (will the model _change_ behavior)?”  
    → If yes, consider causal/uplift rather than pure prediction. [[arxiv.org]](https://arxiv.org/html/2309.12036v1)
**D. Candidate model families + trade-offs**
- Baseline → simple/logistic/linear; then tree/GBDT; then deep/sequence; then causal/uplift; then LLM/agentic.
- You must justify **why this class**, not just “XGBoost is good”.
**E. Evaluation**
- Offline: proper metrics + calibration + segment checks
- Online: A/B, staged rollout, or quasi-experiment if needed
**F. Deployment constraints**
- Latency/throughput/cost
- Monitoring & retraining triggers
- Enterprise controls: identity, auditing, access boundaries

### modeling approach spine
- **Decision & risk** – what are we deciding and what’s the worst mistake?
- **Decomposition** – what sub‑decisions exist?
- **Model choice** – simplest thing that works
- **Evaluation** – offline ≠ online
- **Failure handling** – what happens when it’s wrong?

### The 3 anchors
1. risk asymmetry: some mistakes are worse than others
	1. thresholds, escalation, conservative rollout, rollback decisions...
2. segment before concluding: before conclude, slice the data
	1. Simpson's paradox, cohort analysis, DiD
3. fail safely: when uncertain, degrade to human
	1. fallback paths, confidence gating

### Feature importance
1. Feature importance to answer *"Is the model now paying attention to the wrong things?"*
	1. a single keyword suddenly dominates routing
	2. a feature that correlates with severity becomes a shortcut
2. *Feature importance explains model behavior, not causality*
	1. "I’d compare importance before and after the regression, and also slice by ticket type or severity. I’d treat this as evidence about model behavior, not causal proof, and validate it with segmentation and counterfactual comparisons."

## 2) Model choice cheat-sheet (what they really want to hear)
### 2.1 If the goal is “predict who will do X”
Good default: **regularized logistic/GBDT**, with careful feature hygiene.
- Strengths: strong tabular performance; handles nonlinearity; fast iteration.
- Watchouts: drift, leakage, calibration, and spurious correlations.
### 2.2 If the goal is “what should we do to change X” (targeting / interventions)
Call out the distinction explicitly:
- “A propensity model predicts _who converts_; an uplift/causal model predicts _who converts because of the intervention_.”  
    Then propose: [[arxiv.org]](https://arxiv.org/html/2309.12036v1)
- **Experiment-first** if feasible (A/B, staged rollout).
- Otherwise, causal approaches (DiD, matching, doubly robust, meta-learners) with assumptions and sensitivity checks.

### 2.3 If the goal is “forecast / capacity / time series”

Talk about:

- Seasonality, leakage, hierarchy, intermittent demand
- Backtesting, rolling windows, and what decisions the forecast drives

### 2.4 If the goal is “agentic AI / automation”

You’ll be evaluated on whether you think beyond “LLM prompt”. A practical enterprise framing (from 10 - Building Scalable AI Solutions From Agentic Architecture to Enterprise Deployment.pdf):

- An agent is a micro-service that takes messages, can invoke tools/APIs, and returns messages. [[10 - Build...Deployment | PDF]](https://microsofteur.sharepoint.com/sites/InnovationHub-Partners/Shared%20Documents/Presentations/10%20-%20Building%20Scalable%20AI%20Solutions%20From%20Agentic%20Architecture%20to%20Enterprise%20Deployment.pdf?web=1)
- There’s a spectrum: **retrieval → task → autonomous** agents. [[10 - Build...Deployment | PDF]](https://microsofteur.sharepoint.com/sites/InnovationHub-Partners/Shared%20Documents/Presentations/10%20-%20Building%20Scalable%20AI%20Solutions%20From%20Agentic%20Architecture%20to%20Enterprise%20Deployment.pdf?web=1)
- Core architecture elements include orchestration, tools, governance/identity, observability/evals, and a landing zone. [[10 - Build...Deployment | PDF]](https://microsofteur.sharepoint.com/sites/InnovationHub-Partners/Shared%20Documents/Presentations/10%20-%20Building%20Scalable%20AI%20Solutions%20From%20Agentic%20Architecture%20to%20Enterprise%20Deployment.pdf?web=1)
- Key design questions: data/knowledge grounding, tool permissions, human-in-loop steps, orchestration pattern, and UX channels. [[10 - Build...Deployment | PDF]](https://microsofteur.sharepoint.com/sites/InnovationHub-Partners/Shared%20Documents/Presentations/10%20-%20Building%20Scalable%20AI%20Solutions%20From%20Agentic%20Architecture%20to%20Enterprise%20Deployment.pdf?web=1)

If you can fluently discuss those, you’ll sound “enterprise-ready” (which is exactly what they’re signaling).
### Things I miss
1. quantify
	1. "I’d define an explicit acceptance criterion before rollout, like *X% reduction in low‑severity escalations...*"


### A. The **modeling approach spine** (your default flow)
You don’t say this out loud — you _walk it_.
1. **Decision & risk**  
    _What decision is being made? What’s the worst mistake?_
2. **Decomposition**  
    _What sub‑decisions exist?_
3. **Model choice**  
    _What’s the simplest thing that works?_
4. **Evaluation**  
    _Offline vs online — what can mislead me?_
5. **Failure handling**  
    _What happens when it’s wrong?_
If you hit these implicitly, interviewers feel safe.

---
### B. The **3 anchors** (your reasoning engines)
You only ever need to remember **three sentences**:
1. **Risk asymmetry**
    > _Some mistakes are much worse than others._
2. **Segment before concluding**
    > _Before drawing conclusions, slice the data._
3. **Fail safely**
    > _When uncertain, degrade to humans._
Everything you answered — tickets, thresholds, ranking, retention — was derived from these.


### Offline vs online
1. Offline evals told us the model became safer and more correct, but online metrics revealed a misalignment with user utility
2. 


### Safety vs utility
1. Big picture
	1. "safety is a constraint, not an optimization target"
	2. "once required safety is met, I focus on maximizing user utility within that envelope"
2. segmentation/localization
	1. "for which cohort/query type is performance hurting"
3. policy tuning before retraining
4. controlled rollout