# Chap 2, Visual Search System
## Clarifying Requirements
1. Display to the user: more similar results at top
2. User interaction: allow other operation than clicking?
3. Additional info: metadata available?
4. Safety: image moderation?
5. Training data: online (data) + user interactions (label)?
6. Speed/scope: 100B images, fair assumption ?

## Evaluation for ranking (offline)
1. P43
2. MRR@k, 
3. nDCG considers relevance score of each retrieved item

### Serving
1. Fig 2.19
2. Reranking -- business logic
### ANN Indexing & Retrieval Cheat Sheet (Airbnb Senior MLE Prep)

### 1. ANN Algorithm Comparison Table
When discussing candidate retrieval in an ML System Design interview, you do not need to memorize dozens of indexing variants. Use this comparison table to anchor your default choice (**HNSW**, Hierarchical Navigable Small World ) and pivot cleanly when memory or scale constraints arise.

| Algorithm                                | Core Mechanism                                                                                          | Memory Footprint                                              | Query Latency & Recall                                                  | Typical Interview Use Case                                                                         |
| :--------------------------------------- | :------------------------------------------------------------------------------------------------------ | :------------------------------------------------------------ | :---------------------------------------------------------------------- | :------------------------------------------------------------------------------------------------- |
| **Flat (`IndexFlatIP` / `IndexFlatL2`)** | Exact brute-force vector scan across all items                                                          | High (stores full uncompressed vectors)                       | 100% recall, but too slow (`O(N)`) for catalogs > 100K items            | Small catalogs or computing exact ground-truth neighbors offline                                   |
| **HNSW (`IndexHNSW`)**                   | Multi-layer proximity graph (coarse long-range links on top, fine-grained local links at bottom)        | High (stores full vectors + multi-layer graph edges in RAM)   | **Fastest online latency (<10ms) & highest recall (>95%)**              | **Default industry choice** for low-latency online retrieval when catalog fits in RAM (<50M items) |
| **IVF + PQ (`IndexIVFPQ`)**              | Partitions space into k-means Voronoi cells (`IVF`) + compresses sub-vectors into centroid codes (`PQ`) | **Very Low** (10x–30x memory compression)                     | Good recall, cleanly tunable via `nprobe` (number of clusters searched) | Massive catalogs (100M–1B+ items) or strict RAM/GPU memory budgets                                 |
| **LSH (`IndexLSH`)**                     | Hashes similar vectors into the same buckets via random hyperplane projections                          | High (requires many hash tables to achieve acceptable recall) | Lower recall and slower speed vs. HNSW/IVF on dense neural embeddings   | Rarely chosen in modern dense embedding systems                                                    |

---
### 2. Your 30-Second Senior MLE Answer
If an interviewer asks *"How would you serve embedding-based retrieval?"*, deliver this concise, structured response:
* **Default Choice (HNSW):** *"For online retrieval across millions of listings, I would start with **HNSW** because its multi-layer proximity graph delivers sub-10ms query latency and high recall (>95%)."*
* **Memory & Scale Trade-Off (IVF + PQ):** *"The main trade-off with HNSW is RAM overhead, since it holds both full vectors and graph edges in memory. If memory footprint or index build time becomes a bottleneck at larger scale, we can switch to **IVF + Product Quantization (IVF-PQ)** to compress the vectors and tune recall versus latency via `nprobe`."*
* **Airbnb Marketplace Catch (Metadata Filtering):** *"In Airbnb search, guests also apply hard constraints like **city, check-in/out dates, and guest count**. Standard HNSW does not natively handle arbitrary boolean filters, so we must account for pre-filtering versus post-filtering."*

---

## 3. The Airbnb Marketplace Catch: Metadata & Availability Filtering

In a two-sided travel marketplace, pure semantic similarity is useless if a retrieved home is located in the wrong city or is already booked on the guest's dates.

* **Why Post-Filtering Fails:** If you run HNSW first to fetch the top 500 semantic matches and *then* filter by check-in/out dates and guest count, popular dates (like holiday weekends) can wipe out 95% of your candidates, leaving almost nothing for the downstream ranker.
* **Why Naive Pre-Filtering Fails:** If you filter all listings by exact dates first and then try to traverse an HNSW graph, broken edges in the graph can trap the search in disconnected subgraphs, hurting recall and latency.
* **Production Solutions to Mention:**
  1. **Geographic Sharding / Partitioning:** Partition ANN indexes by city, region, or travel market (`market_id`) since 99% of travel queries are geographically bounded. This drastically shrinks the search space per query and keeps HNSW indexes small and fast.
  2. **Hybrid Filtered HNSW:** Use modern vector engines (e.g., Qdrant, Pinecone, Milvus, or filtered HNSW implementations) that evaluate bitmap metadata masks *during* graph traversal while dynamically expanding candidate exploration (`efSearch`) when filters are highly selective.
  3. **Over-Fetching + Lexical Hybrid Retrieval:** Combine dense ANN retrieval with an inverted index (BM25 / Elasticsearch) that natively handles exact boolean constraints (dates, price caps, pet-friendly flags), then merge and deduplicate candidates before the ranking stage.

---
## 4. Rapid-Fire Interview Follow-Up Probes
* **Q: How do you handle newly created listings or updated prices/availability in the ANN index?**
  * **A:** Separate static/slow-changing semantic attributes (listing description, photos, neighborhood vibe -> embedded into the ANN index via near-real-time streaming updates or incremental HNSW inserts) from fast-changing transactional state (real-time calendar availability and dynamic nightly pricing -> checked via a fast key-value availability service or inverted index).
* **Q: How do you tune the latency vs. recall trade-off in HNSW?**
  * **A:** At index build time, tune `M` (max bi-directional links per node) and `efConstruction`. At online query time, tune `efSearch` (size of the dynamic candidate list explored)—raising `efSearch` increases recall at the cost of higher p99 latency.

## Chap 4
1. inverted index -- BM25, TF-IDF
2. elastic search. Ref 1.
3. query understanding Ref 17.
4. Real-world search Ref 22, 23, 24, 25

## [Shuseng Wang](https://www.youtube.com/@ShusenWang)
1. search query CTR vs document CTR? document exposure? (Lec 1)
2. User satisfaction (on search result)
	1. *Relevance*: 
		1. Semantic, independent of user
		2. single/two tower BERT
	2. *quality*
		1. EAT: Google's Search Quality Evaluator Guidelines
	3. 时效性
		1. 取决于 查询词 意图
		2. 识别方法： 数据挖掘(突发时效性) 