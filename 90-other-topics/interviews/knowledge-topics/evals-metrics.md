# Evaluation Metrics — Selection Table
Choose by three questions, in order: what type is the output, how balanced are the classes, and what do errors cost?

| Metric                               | Output type                                    | Suitable when                                                                    | Watch out                                                                                 |
| ------------------------------------ | ---------------------------------------------- | -------------------------------------------------------------------------------- | ----------------------------------------------------------------------------------------- |
| Accuracy                             | binary / categorical labels                    | classes roughly balanced, error costs symmetric                                  | collapses under imbalance — a majority-only classifier scores high                        |
| Balanced accuracy                    | binary / multiclass labels                     | imbalanced classes; average of per-class recall                                  | ignores precision; can flatter when negatives are huge                                    |
| Precision / recall / F1              | binary labels                                  | imbalanced, the positive (minority) class is what matters, detection-style tasks | threshold-dependent; precision depends on prevalence, recall doesn't                      |
| PR-AUC                               | scored binary (ranked)                         | rare positives — retrieval, detection, triage                                    | values not comparable across datasets with different prevalences                          |
| ROC-AUC                              | scored binary (ranked)                         | ranking quality overall, classes not extremely skewed                            | flatters under heavy imbalance — the huge negative pool dilutes FPR                       |
| MCC                                  | binary labels                                  | imbalanced and one honest summary number is required                             | less intuitive to explain; still threshold-dependent                                      |
| Cohen's kappa                        | categorical ratings (e.g. judge vs human)      | chance-corrected agreement, imbalanced label distributions                       | kappa paradox under extreme skew — report the marginals with it                           |
| Weighted kappa                       | ordinal ratings (e.g. 1–5 scales)              | near misses should cost less than far misses                                     | the weighting scheme (linear vs quadratic) changes the number — state which               |
| Fleiss' kappa / Krippendorff's alpha | ratings from 3+ raters                         | inter-annotator agreement; alpha also handles missing data and mixed scales      | alpha is the safer default when raters or data are messy                                  |
| Spearman / Kendall tau               | continuous or ordinal scores vs human rankings | judge scores should rank like humans do, absolute values secondary               | says nothing about calibration — a judge can rank perfectly and score everything too high |
| Pearson correlation                  | continuous scores                              | linear relationship between predicted and reference scores                       | sensitive to outliers and to range restriction                                            |
| MAE / RMSE                           | continuous values                              | regression-style error, magnitude matters                                        | scale-dependent; RMSE punishes large errors disproportionately — choose deliberately      |
| Brier score / ECE                    | predicted probabilities                        | the probabilities themselves are consumed downstream (thresholding, risk)        | ECE depends on binning choices; Brier mixes calibration and discrimination                |
| Recall@k / MRR / nDCG                | ranked lists (retrieval)                       | RAG and search quality; nDCG when relevance is graded                            | needs labelled relevance judgements; recall@k ignores ranking within k                    |

---
## Precision, Recall, Specificity, Sensitivity, ROC and what the numbers mean
### Definitions
- Sensitivity/Recall/TPR = $\frac{\text{TP}}{\text{TP}+\text{FN}}$
	- only about the positive class   
- Specificity/Recall for negative class = 1 - FPR = $\frac{\text{TN}}{\text{TN}+\text{FP}}$
	- only about the negative class
- Precision = $\frac{\text{TP}}{\text{TP}+\text{FP}}$
	- mixes both *positive and negative* classes
	- *Prevalence dependent*: one model at 80% sensitivity and 99% specificity has ~7% precision at 0.1% prevalence and ~90% at 10% prevalence. (To calculate, express every variable in terms of TP.)
- Prevalence = $\frac{TP+FN}{TP+FN+TN+FP}$
	- percentage of positive class
- Accuracy = sensitivity · prevalence + specificity · (1 − prevalence).
	- Change the mix and the same model gets a different accuracy — and at 1% prevalence the specificity term carries 99% of the weight, so a model that finds _no_ positives still scores ~99% if it just says "negative" everywhere. The number looks good while the model does nothing.
- F1 = $\frac{2P*R}{P+R}$, harmonic mean of Precision and Recall
	- the single number you use when both failure modes on the positive class matter — missing real positives (recall) and crying wolf (precision). It is dominated by whichever of the two is worse —  precision 1.0 and recall 0.1 leads to F1 0.18.
#### Example
### The Bayesian frame
- Precision and recall are the posterior and likelihood of Bayes' rule: $$P(Y=1 | Ŷ=1) = P(Ŷ=1 | Y=1) · P(Y=1) / P(Ŷ=1).$$Precision is the posterior (conditions on the prediction — the user's direction of inference), recall is the likelihood (conditions on the truth — the model's direction), prevalence is the prior, and FPR is the likelihood under the negative class. 
- This is why precision moves with prevalence and recall doesn't: posteriors depend on priors, likelihoods don't. The ROC curve is traced by the two likelihoods alone; the PR curve needs the prior as well.
### Precision and recall: what does it mean to the users
- **Recall = P(predicted positive | actually positive)** — conditions on the truth. It only exists as a rate over a population of known positives: a dataset-level property of the model. It *cannot* be stated for an individual item, whose true class is the unknown.
- **Precision = P(actually positive | predicted positive)** — conditions on the prediction. Measured on a representative offline set (at the shipped threshold), it is the likelihood that a positive prediction the user encounters is accurate.
### How the ROC/PR curves are traced
- Both traced by sweeping the decision threshold over the model's scores.
	- The difference in the tracing: as the threshold lowers, FPR and TPR both rise monotonically, but precision zigzags — each newly admitted item is either a true or false positive — which is why PR interpolation needs care and average precision (AP) is the usual discrete summary. 
	- Average precision(AP): average, over all positives, of the precision at that positive's rank — pick a random positive, and AP is the expected fraction of items ranked at or above it that are truly positive. ( #TODO definition of AP)
- For PR-AUC:
	- high threshold → few flags, high precision, low recall; low threshold → recall climbs toward 1 and precision trends down toward the prevalence.
### What the AUCs mean
- ROC-AUC = P(*a random positive scores above a random negative*) — a property of the classifier alone, a measure of its separation capability. ( #TODO link proof?)
- PR-AUC has no equally clean probabilistic reading; it is a property of the classifier *in a population* (it moves with prevalence). The closest physical reading belongs to AP. AP ≠ PR-AUC exactly: AP is the area under the step version of the curve, with no interpolation between recall levels.  ![ROC and PR curves for the same model at 0.1%, 1%, and 10% prevalence](pr-vs-roc-auc-prevalences.png|600)
### Average Precision (AP)
- Definition — an average over thresholds, weighted by recall gained:   $\text{AP} = \sum_k (R_k − R_{k−1}) · P_k$, where $k$ runs over every distinct threshold and $P_k, R_k$ are precision and recall there. Since recall only increases at step size ΔR = 1/#positives, the definition collapses to: `AP = (1/#positives) · Σ over each positive of (precision at that positive's rank)`.
	- negatives only get a say via precision when they're ranked __above__ a positive. A negative sitting below every positive affects nothing at all.
- Worked example
	- ranked output `+ − + − +` (3 positives): precisions at the positives' ranks are 1/1 = 1.00, 2/3 ≈ 0.67, 3/5 = 0.60, so  AP = (1.00 + 0.67 + 0.60)/3 ≈ 0.76. See how it only updates at positives.
- Geometrically, the sum is an area: each term is a rectangle of width  ΔR and height $P_k$, tiling the region under the __step__ version of the PR curve — precision holds flat between observed recall levels, because between them nothing was measured (no interpretation). This suits information retrieval (per query; averaged over queries to get MAP).
- Why it is a good measure — for the question retrieval-style systems ask ("how well does this ranking surface positives, in this population, when consumed from the top?"):
	- It samples the ranking exactly where it can fail: precision at a positive's rank counts the negative mass ranked above that positive.
	- It punishes top-of-list errors repeatedly — a false positive at  rank 2 degrades the precision recorded for every positive below it — matching how rankings are actually consumed, from the top.
	- Its extremes are calibrated: perfect ranking → AP = 1; random ranking → AP = prevalence (precision ≈ base rate everywhere).
	- It is threshold-free: no credit for a flattering threshold choice.
- Limits:
	- it entangles prevalence (chance level = prevalence), so AP is not comparable across populations;
	- it is blind to calibration (any monotone rescaling of scores leaves it unchanged — it sees only the order);
	- it weights all recall levels equally, so if the application lives at one operating point (e.g. recall ≥ 0.95),   precision at that recall is the better measure.
### ROC-AUC vs PR-AUC — when ROC is the better choice
- Rule of thumb: "does the model separate the classes?" → ROC-AUC. "At deployment prevalence, how trustworthy are its positive calls?" → PR-AUC. Report both when the stakes justify it.
- Comparing models across populations with different prevalences: ROC-AUC is prevalence-invariant, so it isolates the model; PR-AUC moves with the base rate even for an identical model.
- Both classes matter symmetrically, or the "negative" class is itself a category of interest — PR only looks at the positive class; ROC treats the classes symmetrically.
- Deployment prevalence or operating threshold not yet known: ROC summarizes ranking quality across all operating points without baking in the eval set's base rate.
- Very few positives in the eval set: precision jumps wildly between thresholds, so PR-AUC estimates are high-variance; ROC-AUC is more stable (report CIs either way).
- Clean interpretation: ROC-AUC = P(a random positive scores above a random negative); PR-AUC has no equivalent reading, and PR curves need care to interpolate.
### ROC vs PR — side by side

| Aspect                       | ROC                                                                                              | PR                                                                                                                         |
| ---------------------------- | ------------------------------------------------------------------------------------------------ | -------------------------------------------------------------------------------------------------------------------------- |
| Axes                         | FPR (x) vs TPR/recall (y)                                                                        | recall (x) vs precision (y)                                                                                                |
| Conditioning                 | both axes are per-class rates — the two likelihoods                                              | precision mixes both classes — the posterior                                                                               |
| Prevalence dependence        | invariant — same curve at any class mix                                                          | strongly dependent — the curve itself moves with prevalence                                                                |
| Random baseline              | 0.5, always                                                                                      | the prevalence itself (0.001 at 0.1%)                                                                                      |
| Perfect score                | 1.0                                                                                              | 1.0                                                                                                                        |
| Interpretation               | P(a random positive scores above a random negative) — a property of the classifier alone         | no equally clean reading; AP = expected precision at a random positive's rank — a property of classifier-in-population     |
| Role of negatives            | all negatives counted (FPR's denominator) — bulk easy negatives dilute FPR and flatter the curve | only false positives ranked above positives matter; true negatives are invisible — a million easy negatives change nothing |
| Under heavy imbalance        | flatters: ROC point (0.009, 0.80) can hide 8% precision                                          | exposes it: precision shows the false-alarm flood directly                                                                 |
| Class symmetry               | symmetric — swapping the labels mirrors the problem                                              | asymmetric — defined w.r.t. the positive class; swapping labels gives a different curve                                    |
| Curve shape                  | both axes monotonic as the threshold sweeps                                                      | recall monotonic, precision zigzags — interpolation needs care                                                             |
| Standard summary             | AUC (trapezoidal integration is fine)                                                            | AP — area under the step curve; trapezoidal PR-AUC overestimates                                                           |
| Stability with few positives | more stable                                                                                      | high-variance — precision jumps between thresholds                                                                         |
| Answers the question         | "does the model separate the classes?"                                                           | "at deployment prevalence, how trustworthy are its positive calls?"                                                        |