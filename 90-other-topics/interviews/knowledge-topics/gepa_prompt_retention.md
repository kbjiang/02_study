# GEPA (Generative Prompt Optimization) - Prompt Selection & Pareto Domination

## How GEPA Decides Which Update to Keep
GEPA decides which prompt update to keep using a **two-tiered evaluation process** designed to optimize performance while conserving compute. The framework evaluates candidate prompt updates in two distinct stages:

### 1. Minibatch Gating Filter (Initial Screening)
* **Purpose:** Quickly prunes poorly performing updates without wasting compute on large datasets.
* **Mechanism:** A newly generated prompt update is first evaluated on a small, representative subset (minibatch) of the training data. 
* **Decision:** If the update fails to meet a baseline performance threshold relative to the current best prompt on this small sample, it is immediately discarded.

### 2. Full-Dataset Pareto Frontier Filter (Final Retention)
* **Purpose:** Ensures that only globally optimal prompts are kept for the next iteration.
* **Mechanism:** Candidates that pass the minibatch gate are evaluated across the full validation dataset. GEPA maps the results across multiple objective dimensions (such as accuracy, cost, or response length).
* **Decision:** The framework calculates a **Pareto frontier**. An update is retained if it is "non-dominated"—meaning it improves at least one metric without worsening others compared to the current baseline. Inferior or dominated prompt candidates are pruned, and the winning set forms the foundation for the next iteration.

---

## Understanding Dominated Prompt Candidates
In multi-objective optimization, a **dominated prompt candidate** is an updated prompt that is **strictly worse than another available prompt across all performance metrics, or worse in some metrics and equal in the rest.** 

In simpler terms, it is a prompt that has absolutely no competitive advantage, making it mathematically obsolete.

### How Domination Works (An Analogy)
Imagine you are comparing two prompts based on two goals: **High Accuracy** and **Low Cost (Tokens Used)**.

* **Prompt A (The Baseline):** Has 85% accuracy and costs $0.05 per 1,000 tokens.
* **Prompt B (Candidate 1):** Has 80% accuracy and costs $0.07 per 1,000 tokens. 
  * *Result:* **Prompt B is dominated by Prompt A.** It is less accurate *and* more expensive. There is zero reason to keep it.
* **Prompt C (Candidate 2):** Has 90% accuracy but costs $0.08 per 1,000 tokens.
  * *Result:* **Prompt C is non-dominated.** Even though it costs more, it offers higher accuracy. It represents a valid trade-off, so it is placed on the Pareto frontier.

### Why GEPA Prunes Them
If GEPA kept dominated prompts, the system would waste memory and computational power tracking prompts that no human or algorithm would ever choose. By filtering them out, GEPA ensures it only passes forward prompts that offer a genuine upgrade or a meaningful trade-off.
