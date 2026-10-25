# Kebei’s VLM Fine-Tuning — Interview Story Bank

Use this as a cue sheet, not a script. Lead with the section that matches the question; add detail only when asked.

## Project snapshot

- **Context:** Microsoft project for a fashion-brand customer adding hundreds of products to its catalogue weekly.
- **Goal:** classify product images into a predefined fashion taxonomy and automate catalogue updates.
- **My role:** technical lead for taxonomy design and the end-to-end training pipeline.
- **Team:** PM, data-platform engineering, applied science; junior scientist collected public images, generated training pairs, and ran final evaluation.
- **Deadline:** fixed Ignite delivery date.

## Major challenges

- Limited time before the fixed Ignite deadline.
- Customer catalogue was a messy product catalogue, not a usable taxonomy.
  - Example: “two pairs” treated as a class even though it is an attribute.
  - Product-specific attributes overlapped and were repeatedly defined.
- Limited labelled data; synthetic data was required.
- Need to make taxonomy terms usable by GPT-4o, which generated synthetic examples.
- Class imbalance: some categories had too little data and underperformed.

## Taxonomy: the central design decision

- **Problem:** using the raw catalogue would transfer its inconsistent structure into generation, training, and evaluation.
- **Decision:** create a proper taxonomy rather than reuse the catalogue as-is.
- **Design:** 18 top-level categories; about 100 classes and subclasses.
- **Structure:** separate product classes from attributes; use hierarchical inheritance and overrides for shared attributes.
- **GPT-4o alignment:** choose fashion-product terminology GPT-4o reliably understood.
  - Validate distinctions such as “sneaker” versus “trainer.”
  - This was essential because GPT-4o generated the synthetic training examples; the taxonomy had to be meaningful both to the business and the generator.
- **Why it mattered:** reusable product concepts, cleaner classification/evaluation, and less redundant text in generation, training, and inference.
- **Trade-off:** invested extra effort upfront—including overtime—to establish a scalable foundation without delaying delivery.
- **Caveat:** no separate downstream evaluation of the taxonomy’s long-term impact; the project was later paused.

## Data and training pipeline

- Used GPT-4o to generate synthetic image-and-taxonomy training examples.
- Built Azure Machine Learning workflow:
  - register datasets;
  - create training environment;
  - prepare training and inference scripts;
  - run training and evaluation;
  - deploy endpoint.
- Compared full supervised fine-tuning (SFT) with LoRA-based fine-tuning.
- Broader evidence also mentions SFT and DPO.
  - State exactly which method belonged to which experiment.
  - Do not imply all methods were in one final run.

## Data imbalance and delivery judgment

- Some classes remained data-poor and performed worse; down-sampling did not solve it.
- Could not wait for a perfectly balanced dataset without missing the deadline.
- Trained and delivered the first model with available data.
- Asked the other scientist to continue data generation in parallel.
- Preserved a retraining path if the project continued.
- Important: taxonomy and initial model were completed on time; later retraining did **not** occur because the project was paused.

## Result and factual caveats

- Delivered the first model on time.
- **Evaluation metrics:**
  - Use **accuracy** for single-valued classifications, such as product category.
  - Use **F1** for multi-valued attributes, such as color patterns.
- Project details report **approximately 90% accuracy against GPT-4o-generated ground truth**.
- Project was put on hold shortly after delivery.
- Additional data generation and retraining were not completed.

## Technical insight: memorization vs. generalization

- Diagnostic: rename a class from “sneaker” to “trainer.”
- Observation: fine-tuned model continued to predict “sneaker”; base model adapted to “trainer.”
- Interpretation: fine-tuning may have learned the original taxonomy too rigidly—shortcut learning/memorization rather than flexible domain understanding.
- Proposed direction:
  - continual pretraining for fashion-domain knowledge;
  - zero-shot, few-shot, or lightweight fine-tuning for taxonomy/output alignment.
- **Caveat:** evidence describes preliminary experiments and a proposed approach—not a completed production improvement.

## Question-to-evidence map

| Question area | Reminders |
|---|---|
| Ownership | Identified a foundational taxonomy problem; designed and delivered the fix under deadline. |
| Technical judgment | Did not replicate a messy catalogue; designed a reusable taxonomy aligned to GPT-4o’s fashion understanding. |
| Difficult trade-off | Upfront taxonomy investment and training with imperfect data to meet the deadline. |
| Challenge a decision | Explain why direct catalogue reuse would create inconsistent synthetic data and brittle evaluation. |
| Scientific depth | Hierarchy/inheritance, GPT-4o terminology alignment, SFT vs. LoRA, and sneaker-to-trainer diagnostic. |
| Model failure / learning | Rigid taxonomy behavior; separate domain adaptation from label alignment. |
| Cross-functional work | Coordinate PM, engineering, and junior scientist; manager took RAI work to protect the critical path. |

## Facts to verify before the interview

1. **90% accuracy:** exact test set, evaluation stage, and relationship to GPT-4o-generated ground truth.
2. **~1M synthetic examples:** full project total, later experiment, or multiple stages?
3. **SFT, LoRA, DPO:** which methods were actually run in each experiment?
4. **Endpoint:** demonstration, handoff, or production deployment?
5. **Multi-stage approach:** what was tested versus proposed?

## Delivery reminder

- Start with the decision relevant to the question; do not narrate every project detail.
- State outcomes and caveats precisely.
- Taxonomy design = strongest ownership / judgment thread.
- Generalization diagnostic = strongest science-depth thread.
