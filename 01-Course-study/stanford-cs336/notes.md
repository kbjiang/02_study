# Lecture 1: Overview and Tokenization
## What can we learn in this class that transfers to frontier models?
- There are three types of knowledge:
	- **Mechanics**: how things work (what a Transformer is, how model parallelism leverages GPUs)
	- **Mindset**: squeezing the most out of the hardware, taking scale seriously (scaling laws)
	- **Intuitions**: which data and modeling decisions yield good accuracy
## Course components
1. Basics: how things work
	1. Tokenization, Architecture, Training
	2. also latest updates: RoPE, State-space model...
2. Systems: squeeze the most out of the hardware
	1. GPU Kernels, parallelism, inference...
3. Scaling laws: do experiments at small scale, predict hyperparameters/loss at  large scale
4. Data
	1. Curation, processing...
5. Alignment
## Tokenization
1. Why not character/word-level tokenizer
	1. Unicode codespace is huge $\Rightarrow$ very large vocab
		1. For word, it's unbounded, and `UNK/OOV` is inevitable 
	2. rare chars/words do not 'deserve' being in the vocab, otherwise very inefficient
2. Why not byte-level
	1. The token sequence length would be huge
3. BPE and why
	1. *Intuition*
		1. common sequences of chars are represented by a single token, rare ones by many tokens.
	2. why *byte*-level
		1. Limited base (256 choices) but comprehensive (covers all, no need for `UNK`!)
	3. unlike char/word tokenizer with predefined vocab, here vocab is fluid and requires training

# Lecture 2: Pytorch, Resource Accounting
## Overview of this lecture
- We will discuss all the **primitives** needed to train a model.
- We will go bottom-up from tensors to models to optimizers to the training loop.
- We will pay close attention to efficiency (use of **resources**).
## Tensor basic, model basic...
1. Pytorch tensors are just *pointers* into allocated memory
2. tensor storage
	1. ![[Pasted image 20250705162749.png|600]]
3. [tensor view](https://docs.pytorch.org/docs/stable/tensor_view.html#tensor-view-doc)
	1. Views are free; no duplicated storage.
	2. Taking a view of *contiguous* tensor could potentially produce a non-contiguous tensor.
		```
		Original tensor:
		tensor([[0, 1], [2, 3]])
		Strides: (2, 1)   # t.stride()
		Storage: [0, 1, 2, 3]  # list(t.storage())
		Is contiguous: True  # t.is_contiguous()
		
		Transposed view:
		tensor([[0, 2], [1, 3]])
		Strides: (1, 2)
		Storage: [0, 1, 2, 3]
		Is contiguous: False
		
		Memory layout explanation:
		Original: stride=(2,1) means only 1 step to immediate next element in memory, e.g., from '0' to '1'
		Transposed: stride=(1,2) means 2 steps from '0' to '1', no longer contiguous
		```
## Memory accounting
tags: #memory #gpu
1. Precision `float32`, `bfloat16`
	1. depends on layer. E.g., *QKV* can be half precision
	2. depends on phase. E.g., quantization at inference time
2. For the `Cruncher` example memory breakdown
	1. Parameters: num_parameters = (D * D * num_layers) + D 
	2. Activations: num_activations = B * D * num_layers
		1. need for calculating gradients
	3. Gradients: num_gradients = num_parameters 
	4. Optimizer states: num_optimizer_states = num_parameters 
	5. total_memory (float32):  4 * (num_parameters + num_activations + num_gradients + num_optimizer_states)
3. For Transformers memory breakdown, see [[02_study/01-Course-study/GPU and CUDA/GPU#^gpt2mem|GPU]]
	1. Shows memory usage at different phase of model training, i.e., *steady* vs *peak*.
4. What is optimizer state?
	1. it contains all info for *resuming model training*, including learning rate, momentum, model weights etc.
	2. `optimizer.state` vs `optimizer.params_group`?
## Compute accounting
### tensor_operations_FLOPs
1. floating-point operation (FLOP) is a basic operation like addition or multiplication 
	1. not `FLOP/s`: floating-point operations per second (also written as FLOPS), which is used to measure the speed of hardware.
2. Matrix multiplication `mat_mul`
	1. In general, no other operation that you'd encounter in deep learning is as expensive as matrix multiplication for large enough matrices.
	2. Interpretation
		1. B is the number of data points; (D K) is the number of parameters 
		2. FLOPs for forward pass is ==2 (# tokens) (# parameters)==; 2 is because of both multiplication and addition.
3. Model FLOPs utilization (MFU)
	1. (actual FLOP/s) / (promised FLOP/s); measures how good we are squeezing the hardware
	2. depends on hardware; $<0.5$ is considered bad
### gradients_FLOPs
1. FLOPs in backward pass
	1. ==(2+2) (# tokens) (# parameters)==
		1. (2+2): one for activation derivatives $h.\text{grad}$, the other for weights derivatives $w.\text{grad}$ 
		2. Doubles forward pass:  $f=WX$ is one mat mul; $\partial{f}/\partial{W}$ and $\partial{f}/\partial{X}$ are two mat mul.
	2. in lecture example
		1. set batch size $B=1$ will recover standard backpropagation 
		2. all activation are just identity function, i.e., $\sigma(x)=x$ for all $h$.
### total FLOPs 
1. Forward pass: 2 (# data points) (# parameters) FLOPS
2. Backward pass: 4 (# data points) (# parameters) FLOPS
3. 6 in total
## References
1. See [[02_study/01-Course-study/GPU and CUDA/GPU|GPU]] for more on GPU and its memory usage.

# Lecture 3: Architectures, Hyperparameters
## Recent trends
1. `pre-norm` instead of `post-norm`
	1. keeping normalization out of residual stream helps
2. `rmsnorm` instead of `layernorm`
	1. Normalization ops takes up 0.17% FLOPs, but 25% running time. `rmsnorm` is faster with less trainable params
3. dropping bias terms in `FFN`

## Activations
1. Gated Linear Units (GLU): $\text{GLU}(x, W_1, W_2) = \sigma(W_1x)\odot W_2 x$
	1. $\sigma$ can be any non-linear gating function, such as `ReLU`, `sigmoid`, `silu` etc., it controls the flow of $W_2 x$
	2. *Motivation*: the linear part $W_2 x$ "reduce the vanishing gradient problem by providing a linear path for the gradients", while the gating part $\sigma(...)$ "retaining non-linear capabilities."
	3. With the outgoing linear transformation, this becomes a `FFN`, i.e., $W_3 (\sigma(W_1x)\odot W_2 x)$
		1. as opposed to vanilla `FFN` as $W_3 (\sigma(W_1x))$
		2. $d_{ff}=8/3d_{model}$ instead of vanilla $4d_{model}$ to compensate for the extra params comes with gating. 
2. ==`FFN` and `MLP` are different!== The latter is one linear layer followed with one non-linear activation, the former has non-linearity sandwiched between two linear layers.

### Weight decay
1. Interesting discussion. See [[August#Why Do We Need Weight Decay in Modern Deep Learning?]]
## Other developments
### KV caching
1. At training time, $QK^{T}$ is chunky and done in one go, whiles at inference time, things needs to be done incrementally, meaning all $K_{i<t}$ are calculated multiple times. 
	1. Nice illustration from [Post]([Transformers KV Caching Explained | by João Lages | Medium](https://medium.com/@joaolages/kv-caching-explained-276520203249)) 
		1. ![[kv-caching.gif|800]]
	2. KV caching needs only the same amount of FLOPs as during training $bnd^2$, i.e., no repetition. 
	3. However it requires more memory access to move $KV$ in and out. See *arithmetic intensity* below.
### Multi-query attention (MQA)
1. [Multi-Query Attention is All You Need](https://fireworks.ai/blog/multi-query-attention-is-all-you-need)
	1. single $KV$, multiple $Q$, rest is same as KV caching.
	2. ![[Pasted image 20250803074723.png|700]]
### Arithmetic/operational intensity
#GPU #GPU/roofline
1. Think of it as a measurement of the efficiency of your code. The more efficient, the better usage of each memory access (`FLOPs/byte`), i.e., higher arithmetic intensity 
2. When arithmetic intensity is low, the total FLOPs is linear of memory bandwidth; high end got bounded by peak performance. See [Multi-Query Attention is All You Need](https://fireworks.ai/blog/multi-query-attention-is-all-you-need) for more detail.
	1. The slope of the first part of the curve is the *peak bandwidth*
	2. ![[Pasted image 20250803073700.png|800]]
3. Exemplary calculation for different inferences
	1.  $b$ batch size, $n$ seq length, $h$ num of heads, $d$ hidden dimension, $k=d/h$
	2. KV caching
		1. Total arithmetic operation: $O(bnd^2$), leading term is mat_mul.
		2. Total memory access: $O(bn^2d + nd^2)$, $n^2$ in 1st term is from  $\sum_{1}^{n}{i}$ due to moving $K\in \mathbf{R}^{i\times d}$ at each step of $n$ auto-generations; 2nd term is final projections of each $n$ generation.
		3. arithmetic intensity: $O\Bigl( (\frac{n}{d} + \frac{1}{b})^{-1} \Bigr)$, favors short sequence and big model dimension, which is not good
	3. MQA 
		1. Total arithmetic operation: $O(bnd^2$), leading term is mat_mul.
		2. Total memory access: $O(bn^2k + nd^2)$, similar to `KV caching`, except now we have single $k$ instead of $hk=d$
		3. arithmetic intensity: $O\Bigl( (\frac{n}{dh} + \frac{1}{b})^{-1} \Bigr)$, factor of $h$ improvement.
### Misc
1. numeric instability
	1. mostly from `softmax`
		1. ![[Pasted image 20250806182943.png|800]]
	2. `z-loss`
		1. to "encourage the softmax normalizer `log(Z)` to be close to 0, which we found increases the stability of training." (PaLM)
		2. ![[Pasted image 20250810082206.png|800]]
			1. Intuition: When normalizer $Z(x)$ is close to 1, as a prob. dist. should be, then loss is just $U_r(x)$, which is stable.

# Lecture 4: Mixture of experts #MoE 
1. sparse *FNN* layer (MOE) vs dense FNN layer (vanilla)
	1. ![[Pasted image 20250806075742.png|800]]
	2. For inference, same FLOP, more param returns better results
	3. For training, though more FLOP, much faster to train MOE to get to same loss
		1. still much less than dense models in memory
2. Routing function
3. Training objectives: non differentiable
	1. Switch Transformer
		1. ![[Pasted image 20250805151216.png|800]]
		2. $P$ in eqn (4) is differentiable and $p_i$ with largest $f_i$ got punished the most during training, i.e., balancing. ![[Pasted image 20250808104225.png|800]]
	2. OLMoE
		1. "getting rid of the load balancing loss is an important direction for future research as it constrains the flexibility of the model by forcing it to use all experts approximately equally... prior work has failed to find strong evidence of expert specialization...". *So really the benefit of MoE is more parameters but same FLOP at inferencing.*

## Training
1. Discrete routing is hard because it's non-differentiable. Options:
	1. auxiliary loss balancing
	2. RL
	3. ...
2. Cannot turn on all experts during training, too much FLOPs
3. fine-grained + shared experts
4. upcycling
5. DeepSeek MoE v3: MTP
6. OLMoE
	1. ![[Pasted image 20250808071442.png|800]]


## Misc
1. stochasticity of MoE models, 
	1. even when $t=0$.
2. DeepSeek MoE v3 latent attention
## References
1. [2401.06066](https://arxiv.org/pdf/2401.06066) deepseek
2. [2101.03961](https://arxiv.org/pdf/2101.03961) switch transformers
3. [2409.02060](https://arxiv.org/pdf/2409.02060) OLMoE
	1. A bunch of ablation (Sec. 4) and MoE analysis (Sec. 5)
4. DeepSeek v3?
5. ST-MoE? (https://arxiv.org/pdf/2202.08906)
	1. Good discussion on stability
6. Implementation Qwen3 MOE from Sebastian Raschka
	1. https://www.linkedin.com/feed/update/urn:li:activity:7357401606549655552/

### *TODO*: 
1. follow through this implementation from Sebastian Raschka
	1. https://www.linkedin.com/feed/update/urn:li:activity:7357401606549655552/


# Lecture 5: GPUs
## Part 1
1. CPU: large control, lots of branching, low latency
2. GPU: small control, high throughput
### GPU anatomy
1. memory and physical proximity
2. SM/SP
	1. SM: atomic unit working on its own data
3. threads, blocks and warp
	1. these compute units leveraged by CUDA applications, i.e., from the software side
4. GPUs operate in SIMT (single instruction multi-thread) model
5. Compute (and esp matmuls) have scaled faster than memory
	1. so need to respect the *memory hierarchy* to make things go fast
6. Memory coalescing
	1. ![[Pasted image 20250810210749.png|800]]
		1. Therefore when read a torch matrix, the thread should move along columns to take advantage of `burst section`.
## part 2
![[Pasted image 20250810213451.png|400]]

### Understand this graph
1. [What Shapes Do Matrix Multiplications Like? [medium]](https://www.thonking.ai/p/what-shapes-do-matrix-multiplications)
	1. ![[Pasted image 20250919220735.png|800]]
2. The graph
	1. It should be understood as such that the FLOP/s is jumping all over the place as $N$ increases
	2. Color-coded each dot by the highest power of 2 it’s divisible by. Lowest band is when $N$ is odd, rest bands with $K$ equals to 2, 8, 16 and 32. Also explains why from bottom to top, bands become more sparse.
		1. ![[Pasted image 20250919221255.png|600]]
	3. Tiling
	4. 
# Lecture 6: Kernels, Triton
### Reference
1. Horace He's [blog](https://horace.io/brrr_intro.html)
	1. The discussion on memory/computation/overhead bounded
2. [CUDA refresher](https://developer.nvidia.com/blog/tag/cuda-refresher/)
	1. some basic concepts about CUDA, very useful.
	2. "The CUDA programming model provides a heterogeneous environment where the host code is running the C/C++ program on the CPU and the kernel runs on a physically separate GPU device."
# Lecture 7: Parallelism 1
> big separation between `computation` and `comminucation`. As a GPU, I should be agnostic to communications.
### Memory
1. Memory breakdown
> | Concept | Stored as | Lifetime | Purpose | Memory scale |
> |----------|------------|-----------|-----------|---------------|
> | **Model parameters** | FP16 | Persistent | Define model weights | x |
> | **Master weights** | FP32 | Persistent | Used by optimizer for precise updates | x | 
> | **Optimizer states** (`m`, `v`) | FP32 | Persistent | Store moving averages (Adam) | 2x |
> | **Gradients** | FP16 | Temporary (per step) | Backpropagation | x |
> | **Activations** | FP16 | Temporary (per batch) | Backpropagation | 5–10x |
 2. Master weights
	 1. a **full-precision (FP32) copy** of each parameter for stable accumulation
	 2. not to confuse with activations: the `master weights` belong to the optimizer’s bookkeeping, while activations belong to the computational graph.
 3. Typical flow
	 1. Forward/backward uses FP16 parameters (for speed and memory savings).    
	 2. Gradients are computed in FP16 (then possibly converted to FP32).
	 3. Optimizer updates happen on **master FP32 weights**.
	 4. The updated master weights are cast back to FP16 and copied into the model for the next forward pass.
## Data parallel
 1. meaning each GPU a) gets a part of the data, and b) runs the same operation at the same time (SPMD)
 2. Even though ZeRO stage 3 (or FSDP) shards weight, but it's not considered model parallel. 
	 	1. model parallel never moves weights around while FSDP broadcast weight every step
	 	2. model parallel passes activations
 3. `pipelined`, `tensor`...?
 4. `batch size` as an important resource. See also [[An Empirical Model of Large-Batch Training]]
	 	1. For DP along, there exists a turning point for "good" scaling of BSs, beyond which training time gain diminishes with scaling. ![[Pasted image 20251029155543.png|500]]
		 	1. Intuition: within this limit, there are a lot of gradient noises and reducing this variance leads to faster training; beyond this limit, BS is big enough that batch gradient is close to true gradient, keep increasing BS won't help any more.
		 	2. *Therefore we need to "spend" batch sizes in dimensions other than just data.*
	2. [How to Parallelize a Transformer for Training | How To Scale Your Model](https://jax-ml.github.io/scaling-book/training/)
		 	1. Different dimensions of parallelism, e.g., data, tensor
		 	2. To remain compute-bound at moderate BS, use mixed FSDP + TP. ![[Pasted image 20251029155946.png]]
### ZeRO
1. ZeRO stage 1
	1. each GPU updates a part of the params *at the same time*.
	2. compared with naive DDP
2. ZeRO stage 2
	1. just stage 1 with an extra reduce to have the right (sum of) gradient on the right rank.
3. ZeRO stage 3
	1. contains stage 1 and 2 as special cases?
	2. ZeRO stage 3 is just FSDP. See [[02_study/02-Deep-Learning/Pytorch/notes#Distributed parallelism|notes]]
4. For stage 1 and 2, think of the sharding by model layer, while stage 3 shards on both layer (FSDP unit) and tensor (horizontal split)
5. Communication analysis (Chap 7 of [ZeRO: Memory Optimizations Toward Training Trillion Parameter Models](https://arxiv.org/pdf/1910.02054))
	1. Naive, stage 1 and 2
		1. $2\Psi$ communication/data movement per training step--one for `reduce-scatter` of gradients, the other for `all-gather` of parameters.
	2. stage 3
		1. $3\Psi$. The `broadcasting` of weights at each step, both forward and backward passes, can be deemed as a spread out `all-gather`; and the usual `reduce-scatter` of gradients.
		2. "Therefore, during the forward propagation it needs to receives the parameters for all the other partitions. However, *this can be pipelined to avoid the memory overhead.*"???

### Highlight
1. 26:35 and the example prior to that

### References
1. [How to Parallelize a Transformer for Training | How To Scale Your Model](https://jax-ml.github.io/scaling-book/training/)
	
# Lecture 8: Parallelism 2
### Hardware
1. Generalized hierarchy (from small/fast to big/slow):
	- Single node, single GPU: L1 cache / shared memory
	- Single node, single GPU: HBM
	- Single node, multi-GPU: NVLink
	- Multi-node, multi-GPU: NVSwitch
2. Modern GPU
	- Within a node: NVLink connects GPUs directly, bypass CPU (classic)
	- Across nodes: NVSwitch connects GPUs directly, bypass Ethernet (classic)
	- `nvidia-smi topo -m` shows the connections between GPUs
### Software
1. Collective communication terminology
	- Reduce: performs some associative/commutative operation (sum, min, max)
	- Gather: performs concatenation
	- Broadcast/scatter is inverse of gather
	- All: means destination is all devices
2. NCCL: NVIDIA collective communication library (low level )
	1. NCCL translates collective operations into low-level packets that are sent between GPUs.
	2. They are also GPU *kernels* highly optimized for communication
3. PyTorch distributed library 
4. Ring `AllReduce`
	1. https://youtu.be/rj-hjS5L8Bw?t=478


### From assignment
1. Each of these worker processes belong to a *process group*, which is initialized via `dist.init_process_group`. The process group represents multiple worker processes that will coordinate and communicate via a shared master. The master is defined by its IP address and port, and the master runs the process with rank 0. Collective communication operations like all-reduce operate on each process in the process group.
	```python
	def setup(rank, world_size):
	    os.environ["MASTER_ADDR"] = "localhost"  # these defines the group
	    os.environ["MASTER_PORT"] = "29500"
	    # Use gloo backend for CPU, nccl for GPU
	    backend = "nccl" if torch.cuda.is_available() else "gloo"
	    dist.init_process_group(backend, rank=rank, world_size=world_size)
	```
2. blah


> Did not spend too much time on scaling laws or assignment 3. 
# Lecture 9: Scaling laws 1
1. Scaling laws 
	1. To extrapolate learnings on smaller models to large models
	2. can be the leap from theoretical upper bounds, which is usually quite loose, to more empirical prediction on performance.
2. Model size vs data size (Chinchilla rule)
	1. with fixed computation budget (IsoFLOP), find the point of lowest loss and lcoate the optimal token/parameter ratio. For Chinchilla this is 20:1.
	2. Training efficiency (Chinchilla regime)
	    - Optimal ratio: ~20:1
	    - Reason: Minimizes loss for a fixed compute budget
	3. Inference performance (modern LLMs)
	    - Optimal ratio: 40–200:1
	    - Reason: Improves generalization, data coverage, and long-tail reasoning

# Lecture 11: Scaling laws 2
1. muP
	1. it's about setting good initialization and layer-wise learning rate so that activations are $\Theta(1)$.
	2. It leads to scale-invariant learning-rate
		1. ![[Pasted image 20251030205011.png]]
	3. detailed derivation skipped

| Goal                                        | Optimal Ratio | Reason                                               |
| ------------------------------------------- | ------------- | ---------------------------------------------------- |
| **Training efficiency** (Chinchilla regime) | ~20:1         | Minimize loss for fixed compute                      |
| **Inference performance** (modern LLMs)     | 40–200:1      | Better generalization, coverage, long-tail reasoning |
	2. This ratio has been pushed up dramatically in favor for inference

### Reference
1. Kaplan
2. Chinchilla Hoffman+ 2022


# Lecture 10: inference
> Inference have to be done sequentially, the name of the game is to increase efficiency without hurting accuracy
### Metrics
1. Time-to-first-token (interactive app), Latency (*seconds/token*, interactive app), Throughput (*tokens/second*, batch processing)
	1. High throughput $\neq$ low latency. May wait for a while for batching generating large number of tokens, which won't work for interactive apps.

### Accounting
1. review of arithmetic intensity

2. KV caching
	1. one query a step so each step is $O(t)$, where $t$ is current sequence length
		1. as opposed to full QK attention with $O(t^2)$.
	2. KV cache length grows as $t$ increases $O(t)$
		1. In local attention, KV cache length stays the same $o(1)$, i.e., the size of the sliding window, but hurts performance
3. Therefore the name of the game is to reduce the size of KV cache
	1. why: KV caching cannot be batched, i.e., each KV is specific to each sequence; therefore memory limited, i.e., high memory IO
	2. Tricks e.g.: GQA, local attention etc
4. Quantization, pruning
5. speculative sampling
	1. The goal is to sample with a faster but similar distribution $p_x$, and make sure the algorithm recovers the sampling from the target distribution $q_x$. 
		1. See proof at end of [paper]([2302.01318](https://arxiv.org/pdf/2302.01318)). 
	2. [Speculative Decoding: When Two LLMs are Faster than One](https://www.youtube.com/@EfficientNLP)
		1. Good intuition on $(q_x - p_x)_{+}$ for sampling after rejection ![[Pasted image 20251105180710.png]]
		2. Tricky part is that $\forall x \ s.t. \  p_x < q_x$, there is contribution from resampling after rejection from $\forall x \ s.t. \  q_x <= p_x$.
	3. [Accept-Reject Sampling : Data Science Concepts](https://www.youtube.com/@ritvikmath)
		1. Provides derivation and intuition on accepting with prob. $q_x/p_x$.
		2. This applies to the part where $p_x > q_x$ in speculative decoding.
6. vLLM
	1. small degradation on latency, but huge improve on throughput
	2. Explanation on how it works
		1. [E07 | Fast LLM Serving with vLLM and PagedAttention](https://www.youtube.com/@MLSysSingapore) (Chinese)
		2. [Fast LLM Serving with vLLM and PagedAttention](https://www.youtube.com/@anyscale

### References
1. [All About Transformer Inference | How To Scale Your Model](https://jax-ml.github.io/scaling-book/inference/)
2. vLLM
	1. [vLLM: Easy, Fast, and Cheap LLM Serving, Woosuk Kwon, UC Berkeley](https://www.youtube.com/@AMDDevCentral)



# Lecture 12: Evaluation
> Good review on benchmarks.
# Lecture 13: Data 1
1. Data by phase
	1. Pre-training, mid-training, post-training
2. Common Crawl

# Lecture 14: Data 2
## Data filtering
1. Given target $T$ and raw $R$, find subset of $R$ similar to $T$. ![[Pasted image 20251107135314.png|600]]
	1. Estimate some model based on $R$ and $T$ and derive a scoring function
	2. Keep examples in $R$ based on their score
2. Instantiations
	1. Generative model of $T$ (KenLM)
	2. Discriminative classifier (fastText)
	3. importance resampling (DSIR)
## Fuzzy deduplication
> To remove all pairs of documents that exceed a particular `Jaccard similarity` threshold
### Naively
1. each documents as a set of n-grams; then calculate `Jaccard similarity` of each pair of sets
2. not feasible in time and memory
### MinHash ([book3n.dvi](http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf))
2. `characteristic matrix` of sets
3. MinHash `signature matrix`
	1. `MinHash` function always looks for the 1st non-zero element of a column; but If think of permutation as part of hash function, then we have same number of hash function $h_i, i \in [0, n]$.
	2. The point is to replace permutation (expensive) with hashing functions. See 3.3.5 of the book.

### Locality-sensitive hashing (LSH) ([book3n.dvi](http://infolab.stanford.edu/~ullman/mmds/ch3n.pdf))
> *To quickly find similar items in a set*
1. How
	1. compare only similar pairs by using many different hash functions to avoid quadratic growth in computation time
	2. main concern is false negative, but can be reduced by careful tuning
2. How do `r` and `b` influence the slope and location of phase transition
	1. TODO: see the notebook for plot

### MISC
1. `Shingles` vs `bag of words`
	1. both are any substring of length $k$, i.e., a $k$-shingle, found within the document
	2. `Shingles` counts the 1st time a shingle showed up, while `bow` counts every time
		1. E.g., for `{abcab}`, `Shingles` is `{ab, bc, ca}` while `bow` is `{ab, bc, ca, ab}`
2. `NFD` unicode normalization #unicode 
	>TODO: Connect this to `unicode` section in first lecture	
	1. To separates pre-composed characters (like "é") into their base character ("e") and a combining accent (the acute accent mark).
	2. Useful for removing accents from text as part of normalization.
		```python
		# remove accents
		import unicodedata
		def strip_accents(string):
			# `normlize` separates, `combining` picks out combining chrs
		    return "".join(c for c in unicodedata.normalize("NFD", string)
		                   if not unicodedata.combining(c))
		print("Antoine de Saint-Exupéry")
		print(strip_accents("Antoine de Saint-Exupéry"))
		# Output: Antoine de Saint-Exupery
		```
	3. More tests
		```python
		text = "Exupéry"
		text_norm = unicodedata.normalize("NFD", text)
		print(text)
		# Exupéry
		print(text_norm)
		# Exupéry
		assert text == text_norm, "prints same, but decomposed"

		# unicode is now different
		print([ord(t) for t in text])
		# [69, 120, 117, 112, 233, 114, 121]
		print([ord(t) for t in text_norm])
		# [69, 120, 117, 112, 101, 769, 114, 121]
		
		# the acute accent is a combining char.
		chr(233), chr(101), chr(769)
		# ('é', 'e', '́')
		assert unicodedata.combining(chr(101))==0
		assert unicodedata.combining(chr(233))==0
		assert unicodedata.combining(chr(769))!=0
		
		# show in different encodings
		print(text.encode("ascii", "replace"))
		# b'Exup?ry'
		print(text.encode("utf-8"))
		# b'Exup\xc3\xa9ry'
		print(text_norm.encode("ascii", "replace"))
		# b'Exupe?ry'
		print(text_norm.encode("utf-8"))
		# b'Exupe\xcc\x81ry'
		```

# Lecture 15: Alignment - SFT/RLHF
## SFT
1. Knowledge extraction and alignment
	1. *Instruction fine-tuning a model on "facts it doesn't know", i.e., not included in pre-trianing, makes it hallucinate*--it encourages the model to make things up in order to match the depth of the SFT data.
		1. in other words, SFT works best when we are just extracting pre-training behaviors, not adding new ones.
2. Mid-training

## RLHF
1. *Lots of RL-related empirical results are highly contingent on the specifics of the experiment setup.* There's the risk of over-generalizing conclusions.
2. **SFT (*imitation*) vs RLHF (*Optimization*)**
	1. SFT: Fit $\hat{p}(y|x) \approx p^*(y|x)$ for some reference distribution $p^*$.
		1. *maximize likelihood* measured by KL
		2. requires examples from reference distribution
	2. RLHF: Find $\hat{p}(y|x)$ s.t. $\underset{p}{\max}E_p[R(y, x)]$ for a reward $R(y, x)$
		1. *maximize some reward* function that we can measure
		2. *LMs are policies, no longer a model of some distribution*
		3. requires on-policy sampling, i.e., exploration
	3. DPO can be a very illustrative example, where it leveraged the reward implicitly by linking it to policy in closed form(the Bradley-Terry model). Then it's no longer a RL problem (*maximizes reward*) but a SFT (*maximizes likelihood*).
	4. Cost comparison ![[Pasted image 20251122074049.png|600]]
		1. There are tasks that are much easier for experts to verify than to solve
3. RLHF factors to consider
	1. annotator demographic
	2. human vs AI annotation
	3. style: e.g. length
> from here on actually is covered in Lecture 16
4. Watch out for
	1. Overoptimization: where better proxy reward does not lead to better real human preference (e.g. win-rate)
			1. In other words, you can *succeed the RL* while fail the real task
	2. model collapse #LLM-calibration 
		1. RLHF makes models no longer 'probabilistic models' -- no calibration by default![[Pasted image 20251123073459.png]]
		2. This is by design, but *need to be careful when thinking of them as calibrated probabilistic models*.
		3. *TODO:* look up the papers he mention on slide 13; one of them is GPT4 release 
5. DPO
	1. Conceptually, it is *MLE on the pairwise **implied** rewards* $r(x, y_w) - r(x, y_l)$, under nonparametric assumption and alternative parametrization. 
		1. No reward, it's SFT: *to find a policy such that its implied reward is most likely to generate the pair-wise preference we have in the training data.*
		2. ![[Pasted image 20251123064345.png]]
	2. DPO update
		1. ![[Pasted image 20251123070957.png]]
	3. *TODO*: really understand the argument of $\sigma$ function. Compare with logistic regression.
### *Why is this not true RL*
1. No interaction with environment (like in a game setting)
 
# Lecture 16: Alignment - RLVR
1. Overall mindset (around 12:25)
	1. *TODO*: RLHF is hard to scale, needs more efficient RL... Find domains where RL works well
		1. “Overoptimization” means the model is pushed so hard toward a reward signal that it starts exploiting quirks of the reward model (*reward hacking*) instead of actually getting better, which can distort its behavior or reduce general capability. Using RL only in narrow, well-defined domains reduces this risk by limiting where that strong optimization pressure is applied.
2. PPO
	1. reward shaping
		1. per-token KL penalty, last-token full reward
		2. *TODO*: his comment on RL as a contextual bandit 22:55
## GRPO
1. Why not PPO or DPO
	1. PPO: complicated implementation, requires a value model as well
	2. DPO: data not inherently pairwise
2. replace advantage/value func
3. length bias and fix

### Case study
#### Deepseek
1. R1 zero: controlled setting/baseline
	1. base model (Deepseek-V3) + GRPO
	2. reward design: accuracy + format (thinking tags)
2. R1: all tricks
	1. base model (Deepseek-V3) + reasoning SFT + GRPO + SFT/RLHF
	2. reward design: + language consistency for COT (s.t. it's always in same language)
	3. others: 
		1. SFT initialization: so model knows how to do long COT without starting with RL?
		2. Non-verifiable rewards in last RLHF
	4. Process Reward Model (PRM) and Monte Carlo Tree Search (MCTS) did not work
#### Qwen 3
1. thinking mode fusion
# Lecture 17: Alignment - RLVR
### *RL step for language models*
- state $s$: prompt + generated response so far
- action $a$: generate next token
- Rewards $R$: how good the response is; here we focus on
	- outcome rewards, which depend on the entire response
	- verifiable rewards, whose computation is deterministic
	- notions of discounting and bootstrapping are less applicable (which are more for intermediate rewards)
- Transition probabilities $T(s'|s, a)$: deterministic $s'=s+a$
	- can do planning/test-time compute (unlike in robotics) **???**
	- states are really made up (different from robotics), so a lot of flexibility
- Policy $n(a|s)$: just a language model (fine-tuned)
- Rollout/episode/trajectory: $s \rightarrow a \rightarrow ... \rightarrow a \rightarrow R$.
- Objective: maximize expected reward $E[R]$ 
### Policy gradient
#### Naive policy gradient
- Given sample prompt $s$, sample response $a \sim \pi(a|s)$ 
- ==Update parameters based on $(\nabla \log \pi (a|s)) R(s,a)$; this is ==
	- `log_prob` (stems from the *log-derivative trick*) is same as SFT, but weighted by $R(s, a)$.
		- E.g., if $R(s, a) \in \{0, 1\}$ then only updates on correct responses
	- different to SFT where $a$ is generated by annotator and is fixed; in RL it is generated by policy/model and changing over time.
- Challenge: high noise/variance
	- sparse rewards: in contrast, in RLHF the reward model are more dense.
	- For variance, see example generated by GPT; see chat history in ChatGPT. https://chatgpt.com/share/6930dc98-c22c-8003-8361-b16b9025dea6
#### Baselines
- $\nabla \log \pi (a|s) R(s,a)$ is an unbiased estimate of $\nabla E[R]$, but it has high variance.
	- See section 6.5 of assignment 5
		- "the idea is to decrease the variance of the estimator by subtracting a term that is correlated with it, without introducing bias."
		- E.g., the value function can be a reasonable baseline
		- proof of baselined policy gradient being unbiased
- 

### Assignment 
- `compute_naive_policy_gradient_loss`
	- the `policy_log_probs` should *not be understood as loss* because there's no token-wise *label*, as in the case of SFT, only the *id* of the token generated next.
	- the advantage is on *rollout* level, therefore is the same for every token in the same rollout
- only `grpo_clip_loss` is *off-policy*, where `old_log_probs` is required
- What does `loss` mean in RL? why does its gradient work? `GRPO` loss is very close to zero?
- sequence length increase over training, why?
- reinforce vs grpo
	- ![[Pasted image 20251209192739.png]]
	- The key difference:
		- `reinforce_with_baseline` weights by log probability — tokens the model is uncertain about (low probability) contribute more to the gradient
		- grpo_clip (on-policy) is just advantage alone — every token in a good/bad response gets equal gradient signal
	- Why this matters for answer vs format:
		- Format is easy patterns (special tokens like <think>, </answer>) — the model quickly becomes confident, and both losses handle this fine
		- Answer correctness requires learning nuanced reasoning — reinforce_with_baseline gives stronger gradients on uncertain (likely wrong) tokens, helping the model learn which specific tokens to change. grpo_clip just says "this whole response was good/bad" without token-level credit assignment.
	- In short: The log π term in REINFORCE provides implicit credit assignment that helps learning. The PPO-style ratio objective loses this when on-policy.This is why grpo_clip is designed for off-policy (epochs_per_rollout_batch > 1) where the ratio deviates from 1 and clipping prevents destructive updates.








## RLHF
1. deeplearning.ai post-training course https://youtu.be/beMtcPK-ldU

## Mixed-precision
1. See one of the `submission.md` for how errors accumulate for different precision
2. when is `bfloat16` enough, when is it not? For e.g., last assignment SFT finetuning is done in `bfloat16`.

### Logistic regression
1. https://medium.com/@akankshaverma136/logistic-regression-in-ml-a6a1aa1874ff

### Posts in linkedin
### UV
1. just a `venv` vs a `project`. Former `uv venv` and `uv pip install` is enough.