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

## KV caching
1. At training time, $QK^{T}$ is done in one go, whiles at inference time, things needs to be done autoregressively, meaning all $K_{i<t}$ are calculated multiple times. Similarly for $V$s. 
	1. Nice illustration from [Post]([Transformers KV Caching Explained | by João Lages | Medium](https://medium.com/@joaolages/kv-caching-explained-276520203249)) 
		1. ![[kv-caching.gif|800]]
	2. KV caching needs only the same amount of FLOPs as during training $bnd^2$, i.e., no repetition. However it requires more RAM. This leads to *arithmetic intensity*.