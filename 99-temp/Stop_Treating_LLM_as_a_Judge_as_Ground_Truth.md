# Stop Treating LLM-as-a-Judge as Ground Truth

*What if we thought of an LLM judge as a measuring instrument instead of an oracle?*

## Introduction

As LLM-as-a-judge becomes increasingly common, we often see evaluation pipelines that look like: run the model, ask another LLM to judge it, compute a metric, and compare models. This workflow scales well, but it hides a dangerous assumption: the judge is treated as ground truth.

I propose a different framing: **an LLM judge is a measurement instrument, not an oracle.** Once we adopt this perspective, decades of work from measurement science (metrology) become directly applicable.

## Every measurement has two components

A ruler measures the length of a table, but the ruler itself has uncertainty. Engineers therefore calibrate instruments, quantify their uncertainty, measure repeatability, and detect systematic bias. We should ask the same questions of LLM judges.

## LLM judges are measuring devices

Let θ denote the true model quality and J the observed score from the judge.

```
J = θ + ε
```

where ε is the measurement error introduced by the judge.

## Measuring the model versus measuring the measurement

A model's reported accuracy has sampling uncertainty because we evaluate only a finite number of examples. Separately, the judge itself has measurement uncertainty because it disagrees with humans. These are distinct sources of error.

## Calibrating the measuring instrument

Compare the judge against trusted human annotations. Report metrics such as accuracy, precision/recall, F1, Cohen's kappa, MCC, correlation, or Brier score, together with confidence intervals. Recalibrate periodically as models, prompts, and domains evolve.

## Sampling uncertainty is not measurement uncertainty

Evaluating more examples reduces sampling uncertainty, but it does not fix a biased or noisy judge. Measuring more tables with a bent ruler does not make the ruler more accurate.

## Propagating uncertainty

A reported model score should acknowledge both sampling uncertainty and uncertainty introduced by the judge. This mirrors how uncertainty is propagated in experimental science.

## Borrowing from metrology

Concepts including calibration, repeatability, reproducibility, bias estimation, and uncertainty propagation have been refined over centuries. Rather than inventing new evaluation methodologies, we can adapt these established principles.

## Proposed workflow

1. Define the property to measure.
2. Build a human-labeled calibration set.
3. Measure judge reliability.
4. Evaluate the model.
5. Report sampling and judge uncertainty.
6. Recalibrate continuously.

## Conclusion

The right question is not *"Is this judge good enough?"* Instead ask: *"What are the measurement properties of this instrument?"* That shift turns LLM evaluation into a principled measurement problem.
