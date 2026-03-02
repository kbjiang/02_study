Great question, Simon. You're right that when both `poa=1` and `jul=1`, you're in POA in July. But the interaction term `poa:jul` captures something the main effects **cannot**: the *extra* change that happens specifically to POA in July, beyond what you'd predict from the sum of the two individual effects.

Here's the intuition with the four group means:

|                   | May                 | July                                    |
| ----------------- | ------------------- | --------------------------------------- |
| **Florianópolis** | $\beta_0$           | $\beta_0 + \beta_2$                     |
| **POA**           | $\beta_0 + \beta_1$ | $\beta_0 + \beta_1 + \beta_2 + \beta_3$ |
- $\beta_1$ = how much POA differs from Florianópolis (in general)
- $\beta_2$ = how much July differs from May (in general, i.e., the time trend)
- $\beta_3$ = the **additional** effect specific to POA in July — **this is the causal effect (DiD estimate)**

Without $\beta_3$, you'd be forced to model POA-in-July as exactly $\beta_0 + \beta_1 + \beta_2$. That assumes the treatment effect is zero — it assumes the only reasons POA's July deposits differ from Florianópolis's May deposits are (1) being POA and (2) being July. There's no room for the billboard campaign to have had any effect.

The interaction term $\beta_3$ is exactly the **difference-in-differences**: it lets the POA-July cell deviate from what you'd predict by just adding the city effect and the time effect. That deviation is the treatment effect.

In short: `poa` and `jul` alone give you an **additive** model. The interaction `poa:jul` lets you detect whether the treatment **broke** that additive pattern — which is the whole point of DiD.