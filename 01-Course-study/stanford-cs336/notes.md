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
3. For Transformers memory breakdown, see [[02_study/02-Deep-Learning/GPU#^gpt2mem|GPU]]
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
		1. (2+2): one for activation  derivatives $h.\text{grad}$, the other for weights derivatives $w.\text{grad}$ 
		2. Doubles forward pass:  $f=WX$ is one mat mul; $\partial{f}/\partial{W}$ and $\partial{f}/\partial{X}$ are two mat mul.
	2. in lecture example
		1. set batch size $B=1$ will recover standard backpropagation 
		2. all activation are just identity function, i.e., $\sigma(x)=x$ for all $h$.
### total FLOPs 
1. Forward pass: 2 (# data points) (# parameters) FLOPS
2. Backward pass: 4 (# data points) (# parameters) FLOPS
3. 6 in total
## References
1. See [[02_study/02-Deep-Learning/GPU|GPU]] for more on GPU and its memory usage.

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
	2. ![[Pasted image 20250803074723.png|800]]
### Arithmetic/operational intensity
1. Think of it as a measurement of the efficiency of your code. The more efficient, the better usage of each memory access (`FLOPs/byte`), i.e., higher arithmetic intensity 
2. When arithmetic intensity is low, the total FLOPs is linear of memory bandwidth; high end got bounded by peak performance. See [Multi-Query Attention is All You Need](https://fireworks.ai/blog/multi-query-attention-is-all-you-need) for more detail.
	1. ![[Pasted image 20250803073700.png|800]]
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
# Lecture 4: Mixture of experts #MoE 
1. sparse *FNN* layer (MOE) vs dense FNN layer (vanilla)
	1. ![[Pasted image 20250806075742.png|800]]
	2. For inference, same FLOP, more param returns better results
	3. For training, though more FLOP, much faster to train MOE to get to same loss
		1. still much less than dense models in memory
2. Routing function
3. Training objectives: non differentiable
4. ![[Pasted image 20250805151216.png|800]]

## Training
1. Discrete routing is hard because it's non-differentiable. Options:
	1. auxiliary loss balancing
	2. RL
	3. ...
2. Cannot turn on all experts during training, too much FLOPs
3. fine-grained + shared experts
4. upcycling
5. DeepSeek MoE v3: MTP


## Misc
1. stochasticity of MoE models, 
	1. even when $t=0$.
2. numeric instability, mostly from `softmax`
	1. ![[Pasted image 20250806182943.png|800]]
3. DeepSeek MoE v3 latent attention
## References
1. [2401.06066](https://arxiv.org/pdf/2401.06066) deepseek
2. [2101.03961](https://arxiv.org/pdf/2101.03961) switch transformers
3. [2409.02060](https://arxiv.org/pdf/2409.02060) OLMoE
4. DeepSeek v3?