# Airbnb Senior MLE (Query Intelligence) — 10-Day Interview Study Plan

## 1. Background & Candidate Context

* **Target Role:** Senior Machine Learning Engineer, Query Intelligence — Airbnb (US - Remote Eligible)
* **Candidate Profile:** Kebei Jiang (PhD in Physics, 11+ years of applied ML & scientific computing experience across Senior Applied Scientist at Microsoft and Applied Scientist at Viridien).
* **Core Strengths:** Deep expertise in LLMs/VLMs, fine-tuning (LoRA, SFT, DPO), Multimodal RAG, NER/sequence labeling (U.S. patent holder), agentic systems, and automated LLM-as-a-judge evaluation frameworks.
* **Key Prep Objectives:**
  1. **Bridge the Domain Gap:** Translate enterprise AI/agentic experience into consumer two-sided marketplace search vocabulary (query understanding, two-stage retrieval/ranking, position bias, online A/B experimentation, and strict `<30ms` serving latency budgets).
  2. **Solidify End-to-End Production Intuition:** Review modern end-to-end MLOps workflows (data validation, experiment tracking/model registry, serving APIs, and drift monitoring) to pair architectural theory with concrete production plumbing.
  3. **Full-Loop Readiness:** Prepare systematically across all onsite rounds reported in recent 2026 Airbnb MLE interview loops—including ML Experience/Trade-offs, Product Classification Design, Two-Stage Search & Ranking Design, GenAI/RAG Design, and Generalist Coding (practiced separately on a daily basis).
* **Study Cadence:** **4 hours/day for 10 days** (40 hours total), excluding daily LeetCode practice.

---

## 2. 10-Day Intensive Study Plan

| Day | Focus Area | Time Breakdown & Exact Materials | Execution Callouts & Tactics |
| :--- | :--- | :--- | :--- |
| **Day 1** | **Retrieval & Classification Foundations** | • **2.5h:** *Machine Learning System Design Interview* — **Ch. 2** (*Visual Search System*) & **Ch. 5** (*Harmful Content Detection*)<br>• **1.5h:** Draft outline for Airbnb's **Family-Friendly Listing Classifier** (June 2026 Reddit prompt) | • **Apply Ch. 1's 7-Step Framework + the 1Point3Acres method:** Stop right after reading the problem statement in each chapter, open **Excalidraw**, and sketch your own 7-step solution *before* reading the book's answer.<br>• Map Ch. 5's multimodal fusion (text, reviews, images) and imbalanced classification directly to Airbnb's search filter problem. |
| **Day 2** | **Search Architecture & Query Intelligence** | • **2.5h:** *Machine Learning System Design Interview* — **Ch. 4** (*YouTube Video Search*) & **Ch. 9** (*Similar Listings on Vacation Rentals*)<br>• **1.5h:** **Shusen Wang (王树森)** YouTube series on *Search Engine Architecture* & *Query Understanding* | • **Ch. 4 and Ch. 9 are your highest-ROI chapters:** Ch. 4 covers text/query search and ranking, while Ch. 9 is modeled directly on vacation rentals like Airbnb.<br>• While watching Shusen Wang, focus on **query spell-check, NER/slot tagging, query expansion, and `<30ms` latency budgets** for Query Intelligence. |
| **Day 3** | **Personalization, Position Bias & Marketplace Mechanics** | • **1.5h:** *Machine Learning System Design Interview* — **Ch. 10** (*Personalized News Feed*) *(optional: skim **Ch. 8** Ad Click Prediction for calibration/CTR)*<br>• **2.5h:** **Shusen Wang** YouTube videos on *Two-Tower Retrieval* & *Position Bias* + Marketplace Q&A notes | • **Close the "Senior" gap here:** Alex Xu's book under-covers bias and marketplace dynamics. Write crisp, 3-sentence answers to the June 2026 Reddit probes: *position bias (IPW / position features)*, *cold-start listings*, *two-sided marketplace A/B cannibalization*, and *offline-vs-online metric drops*. |
| **Day 4** | **End-to-End ML Production Exposure (Part 1)** | • **3.0h:** **Made With ML** (Goku Mohandas) — *Data Validation, Modeling, Experiment Tracking (MLflow), and Reproducibility* modules<br>• **1.0h:** Excalidraw mapping of data-to-registry lifecycle | • **Keep your momentum:** Now that you have the architectural blueprint from Days 1–3, use *Made With ML* to see how a real **NLP/text classification pipeline** is wired in production.<br>• Focus on *why* each component exists (preventing training/serving skew, versioning data/code hashes) rather than memorizing syntax. |
| **Day 5** | **End-to-End ML Production Exposure (Part 2)** | • **2.5h:** **Made With ML** — *Serving (FastAPI/Ray), Testing (behavioral/data tests), and Monitoring (EvidentlyAI drift)*<br>• **1.5h:** Map these tools onto an **Airbnb Query Intelligence** pipeline | • **Build your "Rollout & Monitoring" cheat sheet:** Note how `@champion` / `@challenger` aliases, shadow deployments, and input/output drift alerts work.<br>• You will use this exact mental model in Step 7 (Deployment & Monitoring) of your 7-step design framework. |
| **Day 6** | **Timed Mock Design #1: Query Intelligence & Two-Stage Search** | • **40m:** Strict timed Excalidraw mock on **Airbnb Query Understanding & Two-Stage Home Search**<br>• **3h 20m:** Self-critique, drill follow-ups, and redraw a cleaner v2 diagram | • **Simulate real interview pacing:** 5m clarify -> 5m metrics/framing -> 10m data/features -> 15m retrieval/ranking architecture -> 10m serving, latency fallbacks, & monitoring.<br>• Stress-test your design against tail queries, multimodal inputs, and strict `<30ms` latency budgets. |
| **Day 7** | **Timed Mock Design #2: Product Classification** | • **40m:** Strict timed Excalidraw mock on **Family-Friendly Listing Classifier** (June 2026 Reddit prompt)<br>• **3h 20m:** Deep-dive on noisy label generation, multimodal fusion trade-offs, and threshold tuning | • **Inject your own strengths:** Use your Microsoft experience to propose **LLM-as-a-judge / weak supervision** for scalable training labels, and discuss distilling a heavy VLM into a fast lightweight classifier for low-cost inference. |
| **Day 8** | **Timed Mock Design #3: GenAI / RAG Assistant** | • **40m:** Strict timed Excalidraw mock on **RAG Assistant for Airbnb Customer Support** (Sept 2026 Reddit prompt)<br>• **3h 20m:** Refine hybrid retrieval, reranking, latency/token cost trade-offs, and LLM evaluation | • **Own this round:** Leverage your Microsoft *Multimodal RAG* and *Specification-to-Calibration* work.<br>• Explicitly cover chunking, BM25 + dense fusion, cross-encoder reranking, PII redaction, and automated LLM-as-a-judge regression suites. |
| **Day 9** | **Microsoft Narratives, Core Values & CodeSignal Check** | • **2.5h:** Write & polish 3-minute Senior MLE pitches for *Account Reconciliation Agent*, *VLM Fine-Tuning*, & *Multimodal RAG* + 2 Core Values stories<br>• **1.5h:** Run Airbnb's **CodeSignal Free-Coding Question** | • **Use the 1Point3Acres "Doc-to-Story" method:** Write out every Microsoft project using **Constraints -> Simpler Baseline -> Production Failure Mode -> Trade-off -> Guardrail**.<br>• Prep 2 behavioral stories on **technical conflict resolution** and **adapting to shifting scope** (Sept 2026 Reddit loop). |
| **Day 10** | **Full-Loop Dress Rehearsal & Taper** | • **2.0h:** Out-loud verbal dress rehearsal of your 3 Microsoft pitches + 15-minute speed-sketching of your 3 Excalidraw designs<br>• **2.0h:** Final review of CodeSignal's 5-step coding rubric & early rest | • **Speak out loud without notes:** Practice defending *"Why didn't you use a simpler model?"* and *"What happened when offline and online metrics disagreed?"*<br>• Stop studying at least 4 hours before bedtime so you enter your first round sharp and rested. |

---

## 3. Loop Intelligence & Coding Rubric Summary

### A. Recurring Themes from 2026 Airbnb MLE Loops
* **ML Experience & Trade-Offs:** Interviewers skip high-level summaries and probe *why* a modeling decision was chosen over simpler alternatives, how latency/cost/precision/recall were balanced, and how offline vs. online metric disagreements were resolved.
* **Three Distinct ML System Design Flavors:**
  1. *Two-Stage Search & Ranking:* Query understanding, semantic candidate retrieval (two-tower/ANN), deep ranking, position bias, and booking conversion metrics.
  2. *Product Classification:* Problem framing, noisy/weak label acquisition, multimodal features (text, reviews, images), and threshold tuning for user-facing filters.
  3. *GenAI / RAG Systems:* Hybrid retrieval, reranking, latency/token trade-offs, guardrails, and offline/online evaluation.
* **Generalist Coding (CodeSignal):** Conducted by a Software Engineer (45 mins). Focuses on practical string manipulation, parsing, hash maps, tries, heaps, and edge-case handling (e.g., floating-point precision, overlapping replacements) rather than obscure algorithms.

### B. Official Airbnb CodeSignal Grading Flow
1. **Clarify the problem** (bounds, edge cases, tie-breakers, precision requirements) before coding.
2. **Define interfaces** (clean function signatures, input/output types, modular helper methods).
3. **Think out loud** continuously while implementing.
4. **Test & debug proactively** (write and execute unit tests in CodeSignal).
5. **Discuss limitations & trade-offs** (time/space complexity, scaling bottlenecks, alternative approaches).

---

## 4. References & Study Materials

### Official Role & Recruiter Documents
* **Job Posting:** Airbnb — Senior Machine Learning Engineer, Query Intelligence (`careers.airbnb.com/positions/8065789/`)
* **Recruiter Coding Guide:** Airbnb Engineering CodeSignal Preparation Guide (`airbnb.app.box.com/s/3hfbkqt1xhjczfpfk8u2fma35p1owfnc`)

### Interview Loop Reports & Strategy
* **Reddit (Sept 2026 MLE Loop):** *Airbnb Machine Learning Engineer (MLE) Interview* (`reddit.com/r/OfferEngineering/comments/1wl3n04/airbnb_machine_learning_engineer_mle_interview/`) — Covers String Table Formatting, Document Redaction, Home-Search Ranking, Customer-Support RAG Assistant, and Project Deep-Dive probes.
* **Reddit (June 2026 Senior MLE Loop):** *Airbnb Senior MLE Interview Experience Jun 2026* (`reddit.com/r/OfferEngineering/comments/1w5pahq/airbnb_senior_mle_interview_experience_jun_2026/`) — Covers Family-Friendly Listing Classifier, User Embeddings & Two-Stage Search, Position Bias, Modeling Trade-offs, and Floating-Point Coin Change.
* **1Point3Acres (2026 PhD + Senior AS/MLE Retrospective):** Thread `#1173541` (`1point3acres.com/bbs/thread-1173541-1-1.html`) — Source of the Excalidraw active-diagramming workflow and structured Project Narrative Doc technique.

### Books, Courses & Technical Videos
* **Primary System Design Book:** *Machine Learning System Design Interview* by Ali Aminian & Alex Xu (ByteByteGo) — Focus on **Ch. 1** (7-Step Framework), **Ch. 2** (Visual Search), **Ch. 4** (YouTube Video Search), **Ch. 5** (Harmful Content Detection), **Ch. 9** (Similar Listings on Vacation Rentals), and **Ch. 10** (Personalized News Feed).
* **Search & Ranking Depth:** Shusen Wang (王树森) YouTube Series on *Search Engine Architecture, Query Understanding, Two-Tower Retrieval, and Position Bias*.
* **End-to-End Production MLOps:** *Made With ML — MLOps Course* by Goku Mohandas (`madewithml.com`) — Focus on Data Validation, MLflow Experiment Tracking/Registry, FastAPI/Ray Serving, Testing, and Drift Monitoring.
* **Supplemental MLOps Reference:** *Future-Proof DS — MLOps for Data Scientists* (`futureproofds.com/blog/mlops-for-data-scientists/3-experiment-tracking-mlflow`) — Optional reference for `@champion` / `@challenger` model registry promotion and rollback patterns.
* **Optional ML Coding Warm-Up:** `MLE_Coding_Interview_Guide.ipynb` (`github.com/jyotipmahes/ML-Interview-Prep/blob/main/MLE_Coding_Interview_Guide.ipynb`).
