## Metrics

| Metric                         | What it measures                                 | When to use                                              | Comment                                                 |
| ------------------------------ | ------------------------------------------------ | -------------------------------------------------------- | ------------------------------------------------------- |
| **Recall (TPR)**               | Fraction of actual positives detected            | Missing positives is costly                              | Threshold‑dependent; ignores false positives            |
| **Precision (PPV)**            | Fraction of predicted positives that are correct | False positives are costly                               | Threshold‑dependent; sensitive to prevalence            |
| **F1 / Fβ**                    | Balance of precision and recall                  | Need a single trade‑off metric                           | Threshold‑dependent; encodes cost trade‑off via β       |
| **ROC‑AUC**                    | Ability to rank positives above negatives        | Compare models independent of cutoff; moderate imbalance | **Threshold‑free; based on ranking (TPR vs FPR)**       |
| **PR‑AUC (Average Precision)** | Precision–recall trade‑off over all thresholds   | Severe class imbalance                                   | **Threshold‑free; ranking‑based, focuses on positives** |
### One‑line takeaway
- **Threshold‑free metrics (ROC‑AUC, PR‑AUC)** evaluate the _model’s ranking ability_ independent of any decision cutoff.
- **Threshold‑based metrics** evaluate performance at a _specific operating point_ chosen by business constraints.