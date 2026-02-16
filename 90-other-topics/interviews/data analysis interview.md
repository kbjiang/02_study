### Confounder and A/B test
1. *underlying logic*
	1. Correlations might be spurious because of confounding variables; need to control for it.
	2. Even if correlation is true, does not mean causality; need to run A/B test to prove/disprove.
		1. the direction of causality can be either or even loop. Need 'instrumental variable approach'?
2. The Mathematical View
	In statistics, we often express the difference between observation and experimentation using the **Selection Bias** formula:
	$$Observed \text{ } Difference = \text{Average Treatment Effect} + \text{Selection Bias}$$
	*   **Correlations** include the Selection Bias (the confounding variables).
	*   **A/B Tests** use randomization to make Selection Bias equal **zero**, leaving you with only the true Treatment Effect.
3.  **Causality Validation Workflow**

| Step            | Action                                                                                     | Outcome                               |
| :-------------- | :----------------------------------------------------------------------------------------- | :------------------------------------ |
| **Observation** | Notice that users who receive "Discount Emails" spend more.                                | **Correlation found.**                |
| **Control**     | Filter data by "Loyalty Level" to remove the confounder that big spenders get more emails. | Correlation weakens but still exists. |
| **A/B Test**    | Randomly send "Discount Emails" to 50% of a group and "No Email" to the other 50%.         | **Causality proven** (or disproven).  |
