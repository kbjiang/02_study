# RAG Retrieval Evaluation: Precision@k, Recall@k, MRR@k, NDCG@k, and RRF

> Obsidian note. Math uses native Obsidian/MathJax delimiters (`$...$` and `$$...$$`); Python examples use fenced code blocks.

## 1. Running example

Imagine a RAG system answering:

> What is the annual maximum employer contribution to an employee's HSA?

The retriever returns chunks from an employee benefits handbook. Human evaluators assign graded relevance:

- **3** = highly relevant; directly contains the answer
- **2** = relevant; useful supporting information
- **1** = marginally relevant; useful context
- **0** = not relevant

| Rank | Chunk | Content (illustrative) | Relevance |
|---:|---|---|---:|
| 1 | C17 | General explanation of an HSA | 1 |
| 2 | C04 | Employer HSA contribution amounts and annual limits | 3 |
| 3 | C22 | How to submit HSA reimbursements | 0 |
| 4 | C09 | Employee contribution limits and eligibility | 2 |
| 5 | C31 | Dental insurance coverage | 0 |

The relevance vector is `[1, 3, 0, 2, 0]`. In a real evaluation, calculate metrics for each query and aggregate across a fixed query set.

---

## 2. Precision@k

**Question:** Of the top *k* retrieved chunks, what fraction is relevant?

For binary relevance, a chunk is relevant if its grade is greater than zero.

$$
\operatorname{Precision@k} =
\frac{\text{relevant chunks in top }k}{k}
$$

For the top three chunks, grades are `[1, 3, 0]`. Two of three are relevant:

$$
\operatorname{Precision@3} = \frac{2}{3} \approx 0.667
$$

```python
def precision_at_k(relevance, k):
    """Binary Precision@k; any grade > 0 counts as relevant."""
    if k <= 0:
        raise ValueError("k must be positive")

    top_k = list(relevance[:k])
    # Missing result positions are treated as non-relevant.
    top_k += [0] * (k - len(top_k))
    return sum(grade > 0 for grade in top_k) / k


relevance = [1, 3, 0, 2, 0]
print(precision_at_k(relevance, 3))  # 0.666666...
```

**Interpretation:** Precision@k rewards keeping irrelevant chunks out of the top results. It does not distinguish grade 1 from grade 3 unless you choose a stricter binary relevance threshold.

---

## 3. Recall@k

**Question:** Of all relevant chunks for the query, what fraction did the retriever find in the top *k*?

$$
\operatorname{Recall@k} =
\frac{\text{relevant chunks retrieved in top }k}
{\text{total known relevant chunks for the query}}
$$

Suppose the handbook contains four relevant chunks in total. The top three retrieved chunks contain two relevant chunks:

$$
\operatorname{Recall@3} = \frac{2}{4} = 0.5
$$

```python
def recall_at_k(relevance, total_relevant, k):
    """Binary Recall@k. total_relevant is the known total for this query."""
    if k <= 0:
        raise ValueError("k must be positive")
    if total_relevant < 0:
        raise ValueError("total_relevant cannot be negative")
    if total_relevant == 0:
        return 0.0  # Alternatively, exclude such queries by policy.

    retrieved_relevant = sum(grade > 0 for grade in relevance[:k])
    return retrieved_relevant / total_relevant


relevance = [1, 3, 0, 2, 0]
print(recall_at_k(relevance, total_relevant=4, k=3))  # 0.5
```

**Interpretation:** Recall@k measures coverage. It requires knowing the total set of relevant chunks for each query; incomplete relevance judgments can make recall misleading.

---

## 4. MRR@k — Mean Reciprocal Rank

**Question:** How high is the first relevant result, on average across queries?

For one query, let $r$ be the rank of the first relevant result:

$$
RR@k =
\begin{cases}
\frac{1}{r}, & \text{if the first relevant result is at rank } r \leq k \\
0, & \text{if no relevant result appears in the top } k
\end{cases}
$$

Across $N$ queries:

$$
MRR@k = \frac{1}{N}\sum_{i=1}^{N} RR_i@k
$$

In the running example, the first relevant chunk is C17 at rank 1, so `RR@5 = 1`. If the first relevant result were at rank 2, `RR@5 = 0.5`.

### Python implementation

```python
def mrr_at_k(relevance_lists, k):
    """
    MRR@k across queries.
    Each inner list contains ranked relevance grades; grade > 0 is relevant.
    """
    if k <= 0:
        raise ValueError("k must be positive")
    if not relevance_lists:
        return 0.0

    reciprocal_ranks = []
    for relevance in relevance_lists:
        rr = 0.0
        for rank, grade in enumerate(relevance[:k], start=1):
            if grade > 0:
                rr = 1.0 / rank
                break
        reciprocal_ranks.append(rr)

    return sum(reciprocal_ranks) / len(reciprocal_ranks)


relevance_lists = [
    [1, 3, 0, 2, 0],  # first relevant at rank 1 -> RR = 1
    [0, 0, 2, 1, 0],  # first relevant at rank 3 -> RR = 1/3
    [0, 0, 0, 0, 0],  # none in top 5 -> RR = 0
]

print(f"MRR@5 = {mrr_at_k(relevance_lists, 5):.3f}")
# (1 + 1/3 + 0) / 3 = 0.444...
```

**Interpretation:** MRR@k closer to 1 means the first relevant result tends to appear near the top across queries.

**Limitation:** MRR only considers the first relevant result. It does not reward additional relevant chunks after the first one. A system can have high MRR but poor overall coverage.

---

## 5. NDCG@k — Normalized Discounted Cumulative Gain

**Question:** Are the most relevant chunks ranked near the top, considering all retrieved results and graded relevance?

NDCG is useful when relevance has degrees rather than only relevant/not relevant.

### DCG

A common formulation is:

$$
DCG@k = \sum_{i=1}^{k}
\frac{2^{rel_i}-1}{\log_2(i+1)}
$$

Here, $rel_i$ is the relevance grade at rank $i$. The gain rewards higher relevance grades, while the logarithmic discount reduces the contribution of lower-ranked results.

### IDCG and normalization

IDCG is the DCG of the ideal ranking: the same relevance grades sorted from highest to lowest.

For the running example:

- Retrieved: `[1, 3, 0, 2, 0]`
- Ideal: `[3, 2, 1, 0, 0]`

$$
NDCG@k = \frac{DCG@k}{IDCG@k}
$$

NDCG is between 0 and 1 when IDCG is positive. A score of 1 means the ranking is ideal under the chosen labels and gain formula.

### Python implementation

```python
import math


def dcg_at_k(relevance, k):
    """DCG@k with exponential gain: (2^relevance - 1)."""
    if k <= 0:
        raise ValueError("k must be positive")

    score = 0.0
    for rank, grade in enumerate(relevance[:k], start=1):
        gain = (2 ** grade) - 1
        discount = math.log2(rank + 1)
        score += gain / discount
    return score


def ndcg_at_k(relevance, k):
    """NDCG@k for one query with graded relevance labels."""
    if k <= 0:
        raise ValueError("k must be positive")

    actual_dcg = dcg_at_k(relevance, k)
    ideal_relevance = sorted(relevance, reverse=True)
    ideal_dcg = dcg_at_k(ideal_relevance, k)

    if ideal_dcg == 0:
        return 0.0
    return actual_dcg / ideal_dcg


relevance = [1, 3, 0, 2, 0]
print(f"DCG@5  = {dcg_at_k(relevance, 5):.3f}")
print(f"NDCG@5 = {ndcg_at_k(relevance, 5):.3f}")
```

**Why useful for RAG:** Suppose Retriever A returns `[3, 0, 0, 2, 1]` and Retriever B returns `[1, 3, 2, 0, 0]`. Both retrieve three relevant chunks, but B ranks useful chunks more prominently overall. NDCG captures the quality and ordering of multiple relevant results.

**Important:** Gain is a convention. Some implementations use linear gain (`rel`) rather than exponential gain (`2^rel - 1`). State which definition you use.

---

## 6. RRF — Reciprocal Rank Fusion

**RRF is not an evaluation metric. It is a ranking-fusion method.** It combines ranked lists from multiple retrievers without requiring their raw scores to be on the same scale.

In hybrid RAG, the sources might be:

- **BM25:** keyword-based retrieval.
- **Dense retrieval:** embedding/vector similarity.

Their raw scores have different meanings and scales. RRF combines their ranks instead.

### Formula

$$
RRF(d) = \sum_{m=1}^{M}\frac{1}{c + rank_m(d)}
$$

Where:

- $d$ is a document/chunk.
- $M$ is the number of ranked lists.
- $rank_m(d)$ is its rank in list $m$.
- $c$ is a smoothing constant, commonly 60.
- A document absent from a list contributes zero for that list.

### Running example

BM25 ranking: `C04, C17, C09`

Dense ranking: `C17, C22, C04`

With $c=60$:

$$
RRF(C04) = \frac{1}{61} + \frac{1}{63}
$$

$$
RRF(C17) = \frac{1}{62} + \frac{1}{61}
$$

$$
RRF(C09) = \frac{1}{63}, \qquad
RRF(C22) = \frac{1}{62}
$$

The fused ranking is C17, C04, C22, C09. C17 ranks highly because it appears near the top in both lists.

### Python implementation

```python
from collections import defaultdict


def reciprocal_rank_fusion(rankings, c=60):
    """
    Fuse ranked document-ID lists using Reciprocal Rank Fusion.
    Returns (document_id, fused_score) pairs in descending score order.
    """
    scores = defaultdict(float)

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] += 1.0 / (c + rank)

    return sorted(scores.items(), key=lambda item: item[1], reverse=True)


bm25_results = ["C04", "C17", "C09"]
dense_results = ["C17", "C22", "C04"]

fused = reciprocal_rank_fusion([bm25_results, dense_results], c=60)

for doc_id, score in fused:
    print(f"{doc_id}: {score:.6f}")
```

RRF creates a new ranking. Evaluate that ranking with Precision@k, Recall@k, MRR@k, or NDCG@k to determine whether fusion improves retrieval.

---

## 7. Metric comparison and practical choices

| Metric / method | Main question | Graded relevance? | Emphasis |
|---|---|---:|---|
| Precision@k | How many top-k results are relevant? | Usually no | Relevance density |
| Recall@k | How many known relevant results were retrieved? | Usually no | Coverage |
| MRR@k | Where is the first relevant result? | Usually no | Early discovery of one relevant result |
| NDCG@k | Are highly relevant results ranked near the top? | Yes | Ordering of all top-k results |
| RRF | How can multiple rankings be combined? | No | Rank-based fusion |

For a typical RAG retrieval evaluation, consider reporting:

- **Recall@k:** Are we retrieving enough relevant context?
- **NDCG@k:** Are the most useful chunks ranked near the top?
- **MRR@k:** Does at least one relevant chunk appear early?
- **Precision@k:** Are the top results polluted with irrelevant chunks?

Use RRF when combining multiple retrieval methods, then evaluate the fused ranking with the same metrics.

No single metric tells the whole story. The right choice depends on whether the application needs broad evidence coverage, one highly relevant passage, or several well-ranked supporting passages.

---

## 8. Interview-ready summary

- **Precision@k:** relevance density in the top-k results.
- **Recall@k:** coverage of all known relevant results.
- **MRR@k:** how early the first relevant result appears, averaged across queries.
- **NDCG@k:** ranking quality across multiple results, accounting for graded relevance and rank position.
- **RRF:** a method to combine rankings from different retrievers; evaluate its output with retrieval metrics.

A concise interview explanation:

> I would use Recall@k to measure whether the retriever finds enough relevant context, NDCG@k to assess whether the most useful chunks are ranked near the top, and MRR@k to measure how early the first useful result appears. If I combine BM25 and dense retrieval, I might use RRF to fuse their rankings, then evaluate the fused results using the same metrics.
