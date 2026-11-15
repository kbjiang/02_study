# ANN Retrieval: Senior MLE Interview Prep

**Estimated study time: 20 minutes.** The goal is to understand the concepts, explain the trade-offs, and handle one or two interviewer follow-ups.

## 1. Dynamic ANN Index Updates (8 minutes)

### The problem

Airbnb listings have different types of data that change at different frequencies.

| Data | Examples | Update strategy |
|---|---|---|
| Semantic attributes | Description, photos, neighborhood | Generate embeddings and update the ANN index |
| Transactional attributes | Price, availability, dates | Retrieve from a real-time database or online feature service |

**Why separate them?**

Embedding-based retrieval is useful for semantic similarity, but price and availability are dynamic constraints. Rebuilding the entire ANN index whenever a host changes a price or calendar would be inefficient.

### Proposed architecture

```text
                    Listing data
                         |
              +----------+----------+
              |                     |
       Semantic data          Transactional data
      (description, photos)   (price, availability)
              |                     |
        Embedding model       Online KV / inventory
              |                  service
          ANN index                 |
              |                     |
              +----------+----------+
                         |
                  ANN candidates
                         |
              Filter by current constraints
                         |
                       Rank
```

### Important edge cases

- **New listing:** Generate an embedding and insert it into the index.
- **Updated description/photos:** Regenerate the embedding and update the index.
- **Updated price:** Update the transactional store, not necessarily the ANN index.
- **Deleted listing:** Remove or invalidate the candidate.
- **Stale index:** Use versioning and freshness checks.

One additional concept: **post-filtering can reduce candidate recall.** If 100 candidates are retrieved but 90 fail availability checks, retrieve more candidates or use filter-aware retrieval.

### Interview-ready answer

> I'd separate relatively static semantic attributes from dynamic transactional attributes. Semantic features would be embedded and maintained in an incrementally updated ANN index. Price and availability would be retrieved from an online inventory service at query time. After candidate retrieval, I'd enforce the latest booking constraints before ranking. I'd also handle stale entries through versioning and freshness checks.

---

## 2. HNSW: Latency vs. Recall (7 minutes)

### What is HNSW?

**Hierarchical Navigable Small World (HNSW)** is an approximate nearest-neighbor search algorithm that builds a multi-layer graph.

- Upper layers: sparse graph, enabling long-distance navigation.
- Lower layers: denser graph, enabling more precise local search.
- Query: navigate from upper layers to lower layers to find approximate nearest neighbors.

### Three parameters to know

| Parameter | When used | What it controls |
|---|---|---|
| `M` | Index construction | Maximum graph connections per node |
| `efConstruction` | Index construction | Search breadth when building the graph |
| `efSearch` | Query time | Number of candidates explored during search |

**Most important:** `efSearch`.

Increasing `efSearch` generally improves recall but increases query latency.

### Trade-off

| Lower `efSearch` | Higher `efSearch` |
|---|---|
| Lower latency | Higher recall |
| Less computation | More computation |
| Potentially lower Recall@K | Higher latency, potentially higher p99 |

### How would you tune it?

1. Establish a target, e.g. Recall@100.
2. Benchmark several `efSearch` values.
3. Measure recall and p50/p95/p99 latency.
4. Select a value that meets the recall target within the latency budget.
5. Validate with representative production queries.

### Interview-ready answer

> At index construction time, I'd tune M and efConstruction to balance graph quality, memory, and build cost. At query time, efSearch is the main latency-recall control. I'd benchmark Recall@K against p95/p99 latency and choose the operating point that meets our product requirements.

---

## 3. HNSW vs. IVF-PQ vs. LSH (5 minutes)

You don't need to know the mathematical details. Understand why different algorithms exist.

| Algorithm | Basic idea | Main advantage | Main trade-off |
|---|---|---|---|
| HNSW | Graph-based navigation | Strong recall/latency performance | Memory overhead |
| IVF-PQ | Cluster vectors, then search compressed vectors | Memory efficiency and scalability | Approximation/quantization error |
| LSH | Hash similar vectors into buckets | Conceptually simple approximate search | Hash-table and recall trade-offs |

### When would you choose each?

- **HNSW:** When low-latency, high-recall retrieval is important and memory is sufficient.
- **IVF-PQ:** When the vector collection is very large and memory efficiency matters.
- **LSH:** When its hashing approach fits the similarity metric and workload; worth considering, but not automatically the best choice for dense embedding retrieval.

### Interview question: Why not just use LSH?

> LSH is a valid approximate nearest-neighbor approach, but for dense semantic embeddings I'd initially evaluate HNSW and IVF-PQ as well. HNSW often offers a strong recall-latency trade-off, while IVF-PQ can reduce memory requirements. I'd choose based on actual recall, latency, memory, and index-update benchmarks rather than assuming one algorithm is universally superior.

---

## Final Revision Card

| Question | Short answer |
|---|---|
| How do you handle updated listing prices and availability? | Keep dynamic transactional attributes outside the semantic ANN index; check them through an online service. |
| What happens when a new listing is created? | Generate its embedding and incrementally insert it into the ANN index. |
| What happens when `efSearch` increases? | Usually higher recall and higher query latency. |
| What do `M` and `efConstruction` control? | Graph connectivity and index construction search quality. |
| When might you choose IVF-PQ over HNSW? | When memory efficiency at large scale is a major constraint. |
| Why not always use LSH? | Different algorithms have different recall, latency, memory, and workload trade-offs. Benchmark them. |

**Study recommendation:** Spend 20 minutes on these topics, then move on. These are supporting retrieval-engineering concepts, not the central focus of the Airbnb Query Intelligence interview.
