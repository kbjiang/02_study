# Long-Term ML / DL / Post-Training Curriculum

**Goal:** Build a deep, long-term intellectual foundation in machine learning, deep learning, optimization, language modeling, and LLM post-training.

This is designed as a self-directed graduate-style curriculum. It is not optimized for the shortest path to running post-training libraries; it is optimized for durable understanding, research intuition, and visible evidence of mastery.

**Assumed pace:** 10–12 hours/week  
**Estimated duration:** ~12–18 months  
**Main strategy:** learn theory deeply, solve real homework, synthesize through written notes, and produce portfolio artifacts that demonstrate first-principles understanding.

---

## 0. Guiding Philosophy

The goal is not simply to learn DPO, PPO, GRPO, or RLVR as isolated algorithms. The goal is to build the intellectual stack that makes modern post-training methods feel natural:

```text
statistical learning
+ probabilistic modeling
+ convex optimization
+ learning theory
+ deep learning theory
+ language modeling systems
= principled post-training research intuition
```

The expected end state is not just “I know more algorithms.” It is:

> I can read new ML / DL / post-training papers, identify the underlying objective, connect it to probability and optimization, reason about assumptions, and explain the method clearly.

---

## 1. Curriculum Overview

| Semester | Duration | Theme | Primary Materials | Practice Layer | Main Portfolio Artifact |
|---|---:|---|---|---|---|
| 1 | 12–16 weeks | ML foundations | Cornell CS4780 + Murphy MLAPP | CS4780 homeworks; optional CS229 problem sets | Machine Learning Foundations Notes |
| 2 | 10–12 weeks | Optimization + learning theory | Boyd EE364A + CS229M | Boyd / CS229M exercises | Optimization & Learning Theory Notes |
| 3 | 10–12 weeks | Deep learning theory + research mindset | Simon Prince UDL + NYU Introduction to Deep Learning Research + selected papers | Mini-experiments and research-style writeups | Deep Learning Theory Notes |
| 4 | 10–12 weeks | Language modeling | Stanford CS336 second pass | Reuse/refine existing CS336 implementations | Language Modeling from Scratch Report |
| 5 | 3–4 weeks | Targeted RL for post-training | Berkeley CS285 selected lectures | REINFORCE/PPO exercises | RL for Post-Training Note |
| 6 | 8–12 weeks | Post-training specialization | SFT, DPO, RLHF, RLVR papers | DPO/RLVR experiments | Post-Training Theory & Experiments Report |

Recommended sequence:

```text
CS4780 + Murphy + CS4780 homeworks
        ↓
Boyd Convex Optimization + CS229M
        ↓
Prince Understanding Deep Learning + NYU Deep Learning Research + selected papers
        ↓
CS336 second pass + Language Modeling Report
        ↓
Berkeley CS285 selected RL module
        ↓
Post-training: SFT → Reward Modeling → DPO → PPO essentials → RLVR
```

---

## 2. Semester 1 — Machine Learning Foundations

**Duration:** 12–16 weeks  
**Primary course:** Cornell CS4780 / CS5780, *Machine Learning for Intelligent Systems*  
**Primary textbook:** Kevin Murphy, *Machine Learning: A Probabilistic Perspective*  
**Primary homework:** CS4780 homeworks  
**Supplemental practice:** Stanford CS229 problem sets, optional enrichment  
**Optional companion:** Caltech CS156, *Learning From Data*  
**Intuition layer:** 3Blue1Brown linear algebra, calculus, neural network, and probability visualizations  
**Portfolio artifact:** *Machine Learning Foundations Notes*, 20–30 pages

### Why this semester matters

CS4780 and Murphy form the conceptual spine. Since you found the original CS4780 homework archive, those assignments should be the primary practice vehicle because they are aligned with the course and textbook.

CS229 problem sets are no longer the main homework source. Use them selectively as extra practice for high-value topics such as logistic regression, GDA, Newton methods, and ML debugging.

### Core topics

| Topic | What You Should Be Able to Do |
|---|---|
| MLE / MAP | Derive estimators and explain probabilistic assumptions. |
| Linear / logistic regression | Derive objectives, gradients, Hessians, and Newton updates. |
| GDA / Naive Bayes | Compare generative and discriminative modeling assumptions. |
| SVMs / kernels | Explain margins, primal/dual views, and kernel trick. |
| EM / latent variables | Explain EM as lower-bound optimization / coordinate ascent. |
| PCA | Derive PCA from reconstruction error and variance maximization. |
| Regularization | Interpret regularization as both penalty and prior. |
| Generalization | Explain bias, variance, model complexity, and overfitting. |

### Required practice: CS4780 homeworks

Complete the original CS4780 homework sequence first. These assignments should be treated like graduate homework:

1. Attempt problems before looking anything up.
2. Write full derivations.
3. Implement only when required, and keep code readable.
4. Produce plots and short explanations.
5. Revisit key derivations one week later from memory.

If you use the homework archive you found, keep it in your repo under:

```text
01_ml_foundations/cs4780_homeworks/
```

### Optional enrichment: CS229 problem sets

| Resource | Link | Use |
|---|---|---|
| Official CS229 handouts | [Stanford CS229 Course Handouts](https://cs229.stanford.edu/materials.html-full) | Notes and handouts for topic review. |
| CS229 Summer 2020 syllabus | [Stanford CS229 Summer 2020 Syllabus](https://cs229.stanford.edu/summer2020/syllabus-summer2020.html) | Topic-to-assignment map. |
| Problem Set 1 | [CS229 Summer 2020 PS1](https://cs229.stanford.edu/summer2020/ps1.pdf) | Logistic regression and GDA. |
| Problem Set 2 | [CS229 Summer 2020 PS2](https://cs229.stanford.edu/summer2020/ps2.pdf) | Logistic regression stability and debugging. |
| Problem Set 3 | [CS229 Summer 2020 PS3](https://cs229.stanford.edu/summer2020/ps3.pdf) | Use selectively. The Summer 2020 version includes an RL/control exercise. |
| 2018 materials mirror | [CS229 2018 Autumn GitHub Mirror](https://github.com/maxim5/cs229-2018-autumn) | Useful for locating older problem-set folders and notes. Do not use solutions as shortcuts. |

Suggested CS229 usage:

| Topic | Use CS229? | Reason |
|---|---|---|
| Logistic regression | Yes | Good derivation and debugging practice. |
| GDA | Yes | Good generative vs. discriminative contrast. |
| Newton method | Yes | Useful optimization review. |
| EM / PCA | Maybe | Use if CS4780 coverage feels light. |
| Neural network backprop | Optional | Good refresh before deep learning theory. |
| RL assignment | Usually skip | Better covered later via targeted CS285. |

### Optional companion: Caltech CS156, Learning From Data

Use CS156 only if you want extra intuition about learning itself: generalization, overfitting, model complexity, VC dimension, and why learning from finite data is possible.

Do **not** add it as a full separate semester. Treat it as optional viewing during Semester 1, especially if you want a smoother bridge into CS229M.

### Semester 1 deliverables

| Deliverable | Description |
|---|---|
| Machine Learning Foundations Notes | 20–30 page polished note synthesizing probability, objectives, assumptions, and optimization. |
| CS4780 homework writeups | Clean solutions and derivations. |
| Selected CS229 enrichment writeups | Only where useful. Do not overdo this. |
| Derivation notebook | MLE/MAP, logistic regression, GDA, SVM, EM, PCA. |

---

## 3. Semester 2 — Optimization and Learning Theory

**Duration:** 10–12 weeks  
**Primary optimization course:** Stanford EE364A, *Convex Optimization*  
**Primary text:** Boyd & Vandenberghe, *Convex Optimization*  
**Primary theory course:** Stanford CS229M / STATS214, *Machine Learning Theory*  
**Portfolio artifact:** *Optimization & Learning Theory Notes*, 20–30 pages

### Track A — Convex Optimization

| Topic | Why It Matters |
|---|---|
| Convex sets and functions | Language for understanding tractable objectives. |
| Lagrangians and duality | Essential for constrained optimization viewpoints. |
| KKT conditions | Shows up in SVMs, constrained learning, and KL-regularized objectives. |
| Least squares, logistic regression, SVMs | Connects classical ML objectives to optimization structure. |
| Constrained optimization | Helps reinterpret DPO, TRPO, PPO, and KL-regularized training. |

### Track B — Machine Learning Theory

| Topic | Why It Matters |
|---|---|
| Empirical risk minimization | Core abstraction behind supervised learning. |
| Uniform convergence | Classical view of generalization. |
| Rademacher complexity | More refined capacity/generalization notion. |
| Stability | Alternative way to reason about generalization. |
| Overparameterization | Bridge to modern neural networks. |
| NTK / implicit regularization | Helps explain why deep models break classical intuitions. |

### Semester 2 deliverables

| Deliverable | Description |
|---|---|
| KKT / duality cheat sheet | Practical reference for constrained optimization. |
| Optimization derivation set | Logistic regression, SVMs, ridge/lasso, KL-constrained objectives. |
| Learning theory notes | Uniform convergence, Rademacher complexity, stability, overparameterization. |
| Essay | “Why constrained optimization keeps reappearing in ML and post-training.” |

---

## 4. Semester 3 — Deep Learning Theory and Research Mindset

**Duration:** 10–12 weeks  
**Primary textbook:** Simon Prince, *Understanding Deep Learning*  
**Research-mindset course:** NYU, *Introduction to Deep Learning Research* by Alfredo Canziani  
**Reference text:** Goodfellow, Bengio, and Courville, *Deep Learning*, selected chapters  
**Supporting papers:** ResNet, BatchNorm, LayerNorm, Adam/AdamW, double descent, Lottery Ticket, Grokking, Scaling Laws, Chinchilla  
**Portfolio artifact:** *Deep Learning Theory Notes*, 20–30 pages

### Why Prince becomes the primary DL text

Use Simon Prince’s *Understanding Deep Learning* as the main modern deep learning text. It is better aligned with your current goal than reading Goodfellow cover-to-cover because it is more contemporary and more digestible as a structured learning path.

Goodfellow remains valuable as a canonical reference, especially for optimization, regularization, and methodology.

### Why add the NYU course

NYU’s *Introduction to Deep Learning Research* is valuable because it emphasizes research reasoning: using mathematics, diagrams, graphs, physics-style simplification, and coding to form and test hypotheses. Use it as a companion to Prince when you want to strengthen research taste and explanatory ability.

### Textbook backbone

| Source | Role |
|---|---|
| Simon Prince, *Understanding Deep Learning* | Primary modern DL text. Read sequentially or semi-sequentially. |
| Goodfellow Ch. 6 | Deep feedforward networks reference. |
| Goodfellow Ch. 7 | Regularization reference. |
| Goodfellow Ch. 8 | Optimization reference. |
| Goodfellow Ch. 11 | Practical methodology reference. |
| Goodfellow Ch. 12 | Sequence modeling context. |

### Paper clusters

| Cluster | Suggested Topics | Questions to Answer |
|---|---|---|
| Optimization | SGD implicit bias, Adam, AdamW, sharp vs. flat minima | Why does optimization work in huge non-convex networks? |
| Architectures | ResNet, BatchNorm, LayerNorm, Attention | Why do residuals, normalization, and attention stabilize training? |
| Generalization | Double descent, NTK, Lottery Ticket, Grokking | Why do overparameterized models generalize? |
| Scaling | Scaling laws, Chinchilla | How do loss, data, compute, and model size trade off? |
| Research method | NYU DL research lectures, projects, oral-exam style | How do we reason, hypothesize, test, and communicate? |

### Semester 3 deliverables

| Deliverable | Description |
|---|---|
| Deep Learning Theory Notes | A polished, tutorial-style note explaining optimization, generalization, normalization, residuals, scaling, and modern DL intuition. |
| Prince chapter notes | Each chapter: key idea, math, implementation implications, open questions. |
| NYU research notes | Short notes on research reasoning and hypothesis-testing habits. |
| Mini-experiments | Small experiments showing optimizer differences, normalization effects, or generalization behavior. |
| Oral exam bank | 30–50 questions answered from memory. |

---

## 5. Semester 4 — Language Modeling From Scratch

**Duration:** 10–12 weeks  
**Primary course:** Stanford CS336, *Language Modeling from Scratch*  
**Approach:** second pass, not first exposure  
**Portfolio artifact:** *Language Modeling from Scratch Report*

Since you already completed CS336 and have from-scratch implementations, this semester should focus on refinement, synthesis, and documentation rather than rebuilding everything from zero.

### Focus areas

| Area | Goal |
|---|---|
| Tokenization | Understand tradeoffs and implementation details. |
| Transformer architecture | Derive and explain attention, MLP, residuals, LayerNorm, RoPE. |
| Training loop | Understand optimizer, scheduler, batching, logs, instability. |
| Inference | Understand KV cache, decoding, throughput, and memory. |
| Scaling | Connect scaling laws and Chinchilla to practical training decisions. |
| Evaluation | Understand benchmark design, contamination, calibration, and failure analysis. |

### Suggested papers

| Area | Papers / Topics |
|---|---|
| Architecture | Attention Is All You Need, GPT-2, RoPE, FlashAttention. |
| Scaling | Neural Scaling Laws, Chinchilla. |
| Inference | KV cache, speculative decoding, vLLM-style serving concepts. |
| Evaluation | LM evaluation, contamination, calibration, judge reliability. |

### Semester 4 deliverables

| Deliverable | Description |
|---|---|
| Language Modeling from Scratch Report | Technical write-up of your implementation and main lessons. |
| Polished existing repo | Improve README, diagrams, experiments, and failure analysis. |
| Scaling note | Small-scale scaling simulation or experiment. |
| Debug diary | Record shape bugs, instability, numerical issues, and fixes. |

---

## 6. Semester 5 — Targeted RL for Post-Training

**Duration:** 3–4 weeks  
**Primary course:** Berkeley CS285, selected lectures only  
**Portfolio artifact:** *RL for Post-Training Note*, 10–15 pages

Do not drop Berkeley CS285 entirely, but do not treat it as a full semester unless you want to become an RL researcher.

For LLM post-training, the useful subset is policy gradients, advantage estimation, PPO, KL regularization, and LLM RL.

### Suggested CS285 subset

| Topic | Why It Matters for Post-Training |
|---|---|
| RL basics | Gives MDP/reward-optimization vocabulary. |
| Policy gradients | Foundation for REINFORCE, PPO, and RLHF. |
| Actor-critic intuition | Helps understand variance reduction and advantage estimation. |
| PPO / advanced policy gradients | Direct bridge to PPO-style RLHF. |
| LLM RL material | Most directly relevant to modern post-training. |

### Deliverables

| Deliverable | Description |
|---|---|
| REINFORCE derivation | Derive policy-gradient estimator and baseline trick. |
| PPO toy implementation | Implement PPO on a toy setup before mapping intuition to LMs. |
| RLHF note | Explain reward model, KL penalty, clipping, advantages, and instability modes. |
| RLVR bridge note | Explain why verifier rewards simplify the environment compared with general RL. |

---

## 7. Semester 6 — Post-Training Specialization

**Duration:** 8–12 weeks  
**Portfolio artifact:** *Post-Training Theory & Experiments Report*

Treat post-training methods as objective functions and optimization procedures, not disconnected recipes.

### Stage 1 — Supervised Fine-Tuning

| Material | Focus |
|---|---|
| FLAN | Instruction tuning and task mixtures. |
| Self-Instruct | Synthetic instruction data generation. |
| Alpaca | Lightweight instruction-tuning replication. |
| LIMA | Data quality and small curated datasets. |

Deliverables:

- Build or curate an instruction dataset.
- Run SFT baseline.
- Compare full fine-tuning, LoRA, and QLoRA if compute allows.
- Evaluate with custom rubrics and model-based judging.

### Stage 2 — Reward Modeling and Preference Learning

| Material | Focus |
|---|---|
| InstructGPT | SFT → reward model → PPO pipeline. |
| Constitutional AI | Preference data and critique/revision. |
| DPO | Direct preference optimization. |
| IPO / ORPO / SimPO | DPO-family alternatives. |

Deliverables:

- Derive Bradley–Terry reward modeling objective.
- Derive DPO from KL-regularized preference optimization.
- Implement DPO from scratch or polish an existing implementation.
- Compare DPO / IPO / ORPO / SimPO on a small dataset.

### Stage 3 — RLVR

| Material | Focus |
|---|---|
| DeepSeekMath | Verifiable rewards for math reasoning. |
| GRPO | Group-relative policy optimization. |
| DeepSeek-R1 | Reasoning-oriented RL/RLVR pipeline. |

Deliverables:

- GSM8K-style verifier.
- Verifiable reward function.
- Small RLVR training loop.
- Comparison: SFT vs. DPO vs. PPO-style RLHF vs. RLVR.

---

## 8. Evidence Portfolio / Knowledge Distillation Track

Because you already have several from-scratch implementations, the strongest signal is not another toy repo. The stronger signal is visible evidence of research-level understanding.

Every semester should produce two outputs:

1. **Learning artifact:** derivations, homework, experiments, code.
2. **Public-facing synthesis artifact:** polished notes, essays, reports, or repos that show you can connect ideas across fields.

### Portfolio artifacts by semester

| Semester | Artifact | Target Length / Format | Purpose |
|---|---|---|---|
| 1 | Machine Learning Foundations Notes | 20–30 pages | Show systematic ML foundations. |
| 2 | Optimization & Learning Theory Notes | 20–30 pages | Show maturity with duality, constraints, and generalization. |
| 3 | Deep Learning Theory Notes | 20–30 pages | Show modern DL conceptual understanding. |
| 4 | Language Modeling from Scratch Report | 15–25 pages + repo | Show implementation and debugging fluency. |
| 5 | RL for Post-Training Note | 10–15 pages | Show targeted RL literacy for RLHF/RLVR. |
| 6 | Post-Training Theory & Experiments Report | 25–40 pages + repo | Show direct relevance to target roles. |

### Suggested essay topics

| Essay | Core Question |
|---|---|
| MLE, MAP, and Regularization | Why is regularization both penalty and prior? |
| EM as Coordinate Ascent | What is EM really optimizing? |
| KKT Conditions for ML People | Why do constrained optimization problems keep reappearing in ML? |
| DPO as Constrained Optimization | How does DPO relate to KL-regularized preference optimization? |
| Why Residual Connections Help | What optimization problem do residuals solve? |
| LayerNorm vs. BatchNorm | Why did transformers settle on LayerNorm? |
| Why Overparameterized Models Generalize | Why are classical capacity bounds insufficient? |
| Scaling Laws and Chinchilla | What do scaling laws say about compute, data, and model size? |
| PPO for LLMs | Which parts matter for RLHF, and which are historical baggage? |
| RLVR vs. RLHF | Why do verifiable rewards simplify post-training? |

### Evidence quality rubric

| Dimension | Weak Evidence | Strong Evidence |
|---|---|---|
| Notes | Fragmented, personal, hard to follow | Polished and readable by another ML practitioner. |
| Code | Course homework only | Clean README, derivations, tests, experiments, and analysis. |
| Math | Equations copied from papers | Step-by-step derivations with assumptions explicit. |
| Experiments | One-off runs | Controlled comparisons, plots, failure analysis, and ablations. |
| Research synthesis | Paper summaries only | Cross-paper synthesis, limitations, and proposed extensions. |
| Interview readiness | Fact recall | First-principles explanations and cross-topic connections. |

### Priority ranking for your portfolio

Given your existing implementation background, prioritize:

1. Deep Learning Theory Notes.
2. Post-Training Theory & Experiments Report.
3. DPO / preference optimization derivation and implementation.
4. RL for Post-Training Note.
5. Language Modeling from Scratch Report, polishing existing CS336 work.
6. Machine Learning Foundations Notes.

Rule of thumb: for every 10 hours of learning, spend 1–2 hours turning it into visible evidence.

---

## 9. Foundational Visualization Resources: 3Blue1Brown

Use 3Blue1Brown as an intuition layer, not as a replacement for derivations.

| Topic | 3Blue1Brown Resource | When to Use |
|---|---|---|
| Linear algebra | Essence of Linear Algebra | Before and during CS4780. |
| Calculus | Essence of Calculus | Before optimization and backprop derivations. |
| Neural networks | Neural Networks series | During deep learning theory and CS336 review. |
| Backpropagation | Backpropagation video | Before manual neural-net derivations. |
| Gradient descent | Gradient descent visualizations | During Boyd and deep learning optimization. |
| Probability intuition | Probability-related visual explanations | Alongside Murphy when ideas feel abstract. |

Recommended pattern:

- Use 30–60 minutes of visual intuition before dense Murphy or Boyd chapters.
- Then do the formal derivation yourself.
- Then write a short explanation in your own words.

---

## 10. Weekly Rhythm

| Day | Time | Focus | Output |
|---|---:|---|---|
| Monday | 2 hrs | Lectures / course notes | One-page concept summary. |
| Tuesday | 2 hrs | Math derivations | Derivation notebook update. |
| Wednesday | 2 hrs | Reading | Chapter or paper summary. |
| Thursday | 3 hrs | Homework / implementation / experiments | Code, proofs, or problem-set progress. |
| Friday | 2 hrs | Experiments / debugging / synthesis | Plots, logs, observations, or a short essay draft. |
| Saturday | 1 hr | Write-up / oral exam | Weekly synthesis + 5 questions answered from memory. |

---

## 11. Readiness Criteria Before Moving On

| Dimension | Ready-to-Move-On Standard |
|---|---|
| Theory | Can explain the core concept without notes. |
| Math | Can derive the important objective, gradient, or proof at a whiteboard. |
| Implementation | Can implement the core algorithm without copying reference code. |
| Experiments | Can explain plots and failure modes. |
| Research | Can critique assumptions and suggest plausible extensions. |
| Interview | Can answer 80%+ of oral questions clearly and concisely. |

Suggested pass threshold before moving on: **85/100**.

---

## 12. Suggested Repository Structure

```text
ml_dl_post_training_portfolio/
  01_ml_foundations/
    notes.md
    cs4780_homeworks/
    cs229_optional_enrichment/
    derivations/
    implementations/
    experiments/

  02_optimization_learning_theory/
    convex_optimization_notes.md
    learning_theory_notes.md
    proof_sketches/

  03_deep_learning_theory/
    prince_udl_notes.md
    nyu_dl_research_notes.md
    deep_learning_theory_notes.md
    mini_experiments/

  04_language_modeling/
    gpt_from_scratch/
    language_modeling_report.md

  05_rl_for_post_training/
    rl_for_post_training.md
    ppo_toy/

  06_post_training/
    reward_modeling/
    dpo_family/
    rlvr/
    final_report.md

  paper_reviews/
  oral_exam_questions/
```

---

## 13. Final Capstone

**Title idea:** *From Probabilistic ML to Post-Training: A Reproduction-Oriented Study of SFT, Preference Optimization, and RLVR*

| Component | Description |
|---|---|
| Background | Explain ML, optimization, and language-modeling foundations. |
| Theory | Derive SFT, reward modeling, DPO, PPO-style KL regularization, and RLVR. |
| Implementation | Include reproducible code for SFT, DPO, PPO/GRPO mini-module, and evaluation. |
| Experiments | Compare methods under controlled assumptions. |
| Ablations | Dataset quality, reward style, KL strength, LoRA rank, learning rate, model size. |
| Evaluation | Rubrics, model-based judging, task metrics, and failure analysis. |
| Discussion | What worked, what failed, what assumptions matter, and what to try next. |

---

## 14. One-Line Summary

This is a self-directed graduate curriculum designed to make post-training feel like a natural consequence of probability, optimization, deep learning, and language modeling — not a collection of disconnected recipes.
