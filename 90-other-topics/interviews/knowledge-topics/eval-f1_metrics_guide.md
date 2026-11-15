# Deep Dive: Micro vs. Macro F1 Score & Confusion Matrices in Scikit-Learn

This document compiles our discussion on multi-class classification metrics in `scikit-learn`, tracking how **Micro-F1**, **Macro-F1**, **Accuracy**, and **Balanced Accuracy** handle class imbalances, followed by step-by-step mathematical and code derivations.

---

## 1. Conceptual Overview: The Core Dilemma

When expanding from binary to multi-class classification, evaluating your model requires a choice between prioritizing individual **samples (instances)** or prioritizing individual **classes**.

### Micro-F1 vs. Macro-F1
*   **Micro-F1 (`average='micro'`)**: Pools the global counts of True Positives (TP), False Positives (FP), and False Negatives (FN) across all classes first, then computes a single global F1 score.
    *   *Behavior:* Every individual instance carries equal weight. Larger, high-frequency classes naturally dominate the final score.
    *   *Single-Label Note:* In standard single-label multi-class setups, Micro-F1 is mathematically identical to overall **Accuracy**.
*   **Macro-F1 (`average='macro'`)**: Calculates the F1 score independently for each class first, then takes the unweighted arithmetic mean of those per-class scores.
    *   *Behavior:* Every class carries equal weight. A tiny minority class with 5 samples impacts the metric identically to a majority class with 5,000 samples.

### The Analogy: Accuracy vs. Balanced Accuracy
The relationship between Micro and Macro metrics mirrors the difference between standard accuracy and balanced accuracy:
*   **Global Instance View (Majority Wins):** Accuracy & Micro-F1
*   **Balanced Class View (Minority Matters):** Balanced Accuracy & Macro-F1

> **The Technical Distinction:** Balanced Accuracy strictly averages per-class **Recall** (ignoring cross-class False Positives for individual class evaluation). Macro-F1 averages per-class **F1-Scores**, thereby penalizing both False Negatives (Recall drops) and False Positives (Precision drops) within every class before averaging.

---

## 2. Step-by-Step Numerical Example

To see how these metrics diverge under severe class imbalance, let's look at a 10-sample dataset split across 3 classes:
*   **Class 0 (Rare):** 2 samples
*   **Class 1 (Common):** 4 samples
*   **Class 2 (Common):** 4 samples

### The Data Setup
*   **True Labels (`y_true`):** `[0, 0, 1, 1, 1, 1, 2, 2, 2, 2]`
*   **Predicted Labels (`y_pred`):** `[1, 2, 1, 1, 1, 1, 2, 2, 2, 2]`

*Analysis of the predictions:* The model is completely blind to the rare Class 0, misclassifying one as Class 1 and the other as Class 2. However, it performs perfectly on the common classes.

---

## 3. Deriving the Confusion Matrix

Mapping this exact scenario yields a 3x3 grid where **Rows** represent the **True classes** and **Columns** represent the **Predicted classes**:

| | Predicted Class 0 | Predicted Class 1 | Predicted Class 2 |
|---|:---:|:---:|:---:|
| **True Class 0** | **0** *(TP for 0)* | **1** *(FP for 1)* | **1** *(FP for 2)* |
| **True Class 1** | **0** | **4** *(TP for 1)* | **0** |
| **True Class 2** | **0** | **0** | **4** *(TP for 2)* |

### Visual Breakdown:
*   **The Main Diagonal `[0, 4, 4]`:** Represents the True Positives (TP) for each class. Summing these gives $0 + 4 + 4 = 8$ correct predictions out of 10.
*   **The Off-Diagonal Rows:** Visually demonstrates the error bleed. True Class 0 samples spill into Columns 1 and 2, which artificially inflates the False Positives for those common categories.

---

## 4. Complete Python Implementation

The following complete python script derives all four metrics from raw counts to match `scikit-learn` outputs exactly.

```python
import numpy as np
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score, confusion_matrix

# 1. SETUP THE DATA
y_true = np.array([0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
y_pred = np.array([1, 2, 1, 1, 1, 1, 2, 2, 2, 2])

# 2. RAW PER-CLASS COUNTS (Derived from our matrix)
# Class 0: TP=0, FP=0, FN=2
# Class 1: TP=4, FP=1, FN=0
# Class 2: TP=4, FP=1, FN=0
tp = {0: 0, 1: 4, 2: 4}
fp = {0: 0, 1: 1, 2: 1}
fn = {0: 2, 1: 0, 2: 0}

# 3. HELPER FUNCTION FOR F1 SCORE
def calc_f1(tp_val, fp_val, fn_val):
    if tp_val == 0:
        return 0.0
    precision = tp_val / (tp_val + fp_val)
    recall = tp_val / (tp_val + fn_val)
    return 2 * (precision * recall) / (precision + recall)

# Per-class Recalls (For Balanced Accuracy)
recall_0 = tp[0] / (tp[0] + fn[0])  # 0 / 2 = 0.0
recall_1 = tp[1] / (tp[1] + fn[1])  # 4 / 4 = 1.0
recall_2 = tp[2] / (tp[2] + fn[2])  # 4 / 4 = 1.0

# Per-class F1-Scores (For Macro F1)
f1_0 = calc_f1(tp[0], fp[0], fn[0])  # 0.0
f1_1 = calc_f1(tp[1], fp[1], fn[1])  # 2 * (0.8 * 1.0) / (0.8 + 1.0) = 0.8888...
f1_2 = calc_f1(tp[2], fp[2], fn[2])  # 2 * (0.8 * 1.0) / (0.8 + 1.0) = 0.8888...

# 4. METRIC MATHEMATICAL COMBINATIONS

# A. Accuracy: Total correct / Total samples
manual_accuracy = (tp[0] + tp[1] + tp[2]) / len(y_true)

# B. Micro F1: Total global TP, FP, FN accumulated first
global_tp = sum(tp.values())
global_fp = sum(fp.values())
global_fn = sum(fn.values())
manual_micro_f1 = calc_f1(global_tp, global_fp, global_fn)

# C. Balanced Accuracy: Arithmetic mean of per-class recalls
manual_balanced_accuracy = (recall_0 + recall_1 + recall_2) / 3

# D. Macro F1: Arithmetic mean of per-class F1-scores
manual_macro_f1 = (f1_0 + f1_1 + f1_2) / 3

# 5. PRINT AND VERIFY WITH SKLEARN
print("--- INSTANCE-FOCUSED METRICS (Majority Wins) ---")
print(f"Manual Accuracy:      {manual_accuracy:.3f} | Sklearn: {accuracy_score(y_true, y_pred):.3f}")
print(f"Manual Micro F1:      {manual_micro_f1:.3f} | Sklearn: {f1_score(y_true, y_pred, average='micro'):.3f}")

print("\n--- CLASS-FOCUSED METRICS (Minority Punished) ---")
print(f"Manual Balanced Acc:  {manual_balanced_accuracy:.3f} | Sklearn: {balanced_accuracy_score(y_true, y_pred):.3f}")
print(f"Manual Macro F1:      {manual_macro_f1:.3f} | Sklearn: {f1_score(y_true, y_pred, average='macro'):.3f}")

print("\n--- CONFUSION MATRIX VERIFICATION ---")
print(confusion_matrix(y_true, y_pred))
```

### Result Summary Analysis
*   **Accuracy / Micro-F1 (0.800):** Reflects that 80% of data points were correctly classified. Completely masks the systemic failure on the minority class.
*   **Balanced Accuracy (0.667):** Suffers a penalty because one of the three recall terms drops to 0.0.
*   **Macro-F1 (0.593):** Suffers the heaviest penalty. Not only does Class 0 score a 0.0 F1, but the spilling errors drag down the precisions of Class 1 and Class 2, driving the average down further.