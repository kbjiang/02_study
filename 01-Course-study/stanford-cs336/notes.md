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
## Memory accounting
1. When to `float32` or `bfloat16`
	1. Former for accumulating things like optimizer, latter for transitory things
## Compute accounting
1. Pytorch tensors are just *pointers* into allocated memory
2. tensor storage
	1. ![[Pasted image 20250705162749.png|600]]
3. [tensor view](https://docs.pytorch.org/docs/stable/tensor_view.html#tensor-view-doc)
	1. Views are free; no duplicated storage.
	2. Taking a view of contiguous tensor could potentially produce a non-contiguous tensor.
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
4. `tensor_matmul`
5. `einops`
	1. name each dimension
6. floating-point operation (FLOP) is a basic operation like addition or multiplication 
	1. not `FLOP/s`
	2. `mat_mul` dominates the cost; `O(n^3)` 
	3. MFU measures how good we are squeezing the hardware
7. total FLOPs 
	1. Forward pass: 2 (# data points) (# parameters) FLOPS
	2. Backward pass: 4 (# data points) (# parameters) FLOPS
	3. 6 in total; what about `activations`, `optimizers`?