### [SBERT](https://www.sbert.net/index.html) repo
1. Very nice documentation/section on *Semantic Search* and *Retrieve & Re-rank*.
2. Workflow for information retrieval
	1. Use `Bi-Encoder` to do initial/full search in index and return top `n` results
	2. Pair these `n` results with `query` and rerank with `Cross-Encoder`. Keep top `k` as final search results. ![[Pasted image 20230721094739.png]]
	3. Nice notebook [examples](https://www.sbert.net/examples/applications/semantic-search/README.html)
3. Some details
	1. *symmetric search* vs *asymmetric search*. Different models for different lengths of query and corpus.
	2.  `cross-encoder` is more accurate, but cannot be used for initial search coz it only works on pair of query and corpus, which is combinatorically in number of inferences. Assuming `m` queries and `n` corpus, `bi-encoder` needs `O(m+n)` inferences while `x-encoder` needs `O(m*n)`.![[Pasted image 20230721100135.png]]

### [Paper](https://arxiv.org/pdf/1908.10084.pdf) on Sentence-BERT
1. Big idea: 
	1. Introduced `Bi-Encoder` as a cheaper alternative to `cross-encoder`. The contribution is a more accurate sentence embedding in terms of similarity measures such as cosine-similarity.
	2. Me: 
		1. think of `x-encoder` as a single BERT and the pair was just like next sentence prediction (NSP); `bi-encoder` is two separate BERT inferences.
		2. a simple but effective improvement over things like GloVe embedding or last hidden-state of BERT. 
2. Data
	1. SNLI as major training dataset
3. Training, section 3.
	1. multiple objective functions
		1. classification objective requires another `softmax` layer. Otherwise just same as BERT
		2. `triplet` loss function is popular
		3. how do they prepare the labels?
	2. summarizing figure  ![[Pasted image 20230802100528.png]]
1. Evaluation
	1. Here *BERT* is `cross-encoder` while *SBERT* the `bi-encoder`. 
	2. Table 2 shows that `cross-encoder` outperforms. *It's because it attends to both sentences directly.*

### [Paper](https://cdn.openai.com/papers/Text_and_Code_Embeddings_by_Contrastive_Pre_Training.pdf) on OpenAI embedding
1. "In generative models, the information about the input is typically distributed over multiple hidden states of the model." vs "learning a single representation (embedding) of input"
2. Data
	1. Internet data
	2. *in-batch negatives*: "For each example in a mini-batch of M examples, the other (M − 1) in the batch are used as negative examples." I.e., only need to find/label $(x_i, y_i)$.
3. Loss
	1. `cross_entropy` with labels `np.arange(M)` with $M$ being batch size.
	2. loss along both row and col directions of $M \times M$ , coz $(x_i, y_j) \neq (x_j, y_i)$ .
4. log-prob and x entropy [machine learning - Log probabilities in reference to softmax classifier - Cross Validated (stackexchange.com)](https://stats.stackexchange.com/questions/289369/log-probabilities-in-reference-to-softmax-classifier)
	1. it simplifies the gradient calculation
5. Blog post: https://openai.com/blog/new-and-improved-embedding-model


### [Paper](https://arxiv.org/pdf/2212.03533.pdf) on E5
1. It's mostly about preparing high quality training data
2. Training: contrastive learning

### [Paper]([2401.00368 (arxiv.org)](https://arxiv.org/pdf/2401.00368)) on Embedding via LLM
1. Synthetic data over 93 embedding tasks
2. LLM (decoder-only) hidden state as embedding


### Resources
1. https://huggingface.co/blog/mteb