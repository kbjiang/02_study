# GEPA in the SPEC2CAL Paper

## 1. What GEPA does

SPEC2CAL calibrates an LLM judge by improving its natural-language rubric while keeping the judge model fixed.

- The judge is represented as `J_{θ,r}`: `θ` is the fixed model and `r` is the editable rubric.
- GEPA is an **iterative rubric/prompt optimization backend**. It does not fine-tune the judge or update its weights.
- The goal is to improve grading decisions by making the rubric more precise and better aligned with the intended specification.

## 2. Where GEPA gets its signal

SPEC2CAL creates coverage-controlled examples:

- A positive example satisfies the criterion.
- A negative example is created by introducing a targeted corruption into the positive example, anchored to the same input.
- Examples carry intended labels and are split into source-disjoint train, development, and synthetic-test sets.

The signal flow is:

1. Run the judge with its current rubric on the examples.
2. Compare its predictions with the intended labels.
3. Use judge errors and execution feedback/traces to expose weaknesses or ambiguity in the rubric.
4. GEPA's reflection model proposes textual rubric revisions based on that feedback.
5. Evaluate candidate rubrics and retain/select candidates according to the task-specific grading metric.

**Nuance:** GEPA does not simply compare a positive and negative example and automatically infer the right edit. The pairs provide contrastive supervision; the judge's observed behavior and resulting feedback guide the reflection model's revisions.

## 3. Optimization and evaluation

The paper describes GEPA as evaluating candidate rubrics on minibatches, reflecting on execution traces, proposing textual revisions, and retaining complementary candidates on a Pareto frontier.

The objective is matched to the judge task:

- **LLMBar (pairwise):** exact preferred-response choice.
- **BigCodeBench (binary):** thresholded binary correctness / exact class agreement.
- The implementation also describes support for numeric-score judges using normalized score preference, but the reported experiments cover pairwise and binary tasks.

The training split supplies reflection minibatches; the development split scores and selects candidates. The synthetic test split is evaluated after optimization. External natural/adversarial benchmark labels are withheld from optimizer and candidate selection and used for transfer evaluation.

## 4. Reported results

Across five independent runs:

| Benchmark | One-shot rubric rewrite | GEPA |
|---|---:|---:|
| LLMBar | +3.58 ± 0.58 percentage points | +2.46 ± 2.25 all-run mean; three runs revised the rubric, with +4.10 ± 0.25 conditional mean |
| BigCodeBench | +1.10 ± 0.75 percentage points | +1.91 ± 0.32 percentage points |

For LLMBar, GEPA made no rubric change in two of five runs. For BigCodeBench, all five runs revised the rubric. The reported `±` values are sample standard deviations, not confidence intervals.

The BigCodeBench GEPA and one-shot results are **not data-matched**: GEPA used the full synthetic training split, while the one-shot rewrite used 30 stratified pairs. The paper therefore does not establish a clean apples-to-apples GEPA advantage over one-shot rewriting.

## 5. Representative rubric changes

The appendix summarizes, rather than reproduces verbatim, representative changes:

- **LLMBar:** revisions emphasized instruction following, requested format, scope, constraints, correctness, completeness, and relevance—not surface polish or verbosity.
- **BigCodeBench:** revisions required concrete, externally observable functional evidence for rejection and discouraged speculative hidden failures or style-based rejection.

The point is sharper decision boundaries, not simply longer rubrics.

## 6. Limitations and interpretation

- Synthetic negative examples can be noisy and are not guaranteed to be true violations.
- GEPA sometimes produces no-op runs, as in two of five LLMBar runs.
- The reported experiments cover two task families and two judge forms: pairwise and binary.
- The BigCodeBench one-shot and GEPA data budgets differ.
- External evaluation is still needed to test transfer beyond synthetic examples.
- Rubric length, token count, and inference cost are not explicit GEPA optimization objectives in this paper. Length differences are descriptive.

## Interview takeaway

> In SPEC2CAL, GEPA iteratively optimizes the judge's rubric while keeping the judge model fixed. Positive/negative pairs provide contrastive supervision. The current judge is evaluated against their intended labels; its errors and execution feedback give GEPA's reflection model signals about where the rubric is underspecified. GEPA proposes revised rubrics, which are evaluated and selected using a task-specific grading objective. This is prompt/rubric optimization—not model fine-tuning.
