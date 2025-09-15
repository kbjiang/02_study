## Lecture 1 How to profile CUDA kernels in PyTorch
1. [Time stamp](https://youtu.be/LuhJEEJQgUM?list=TLPQMjcwODIwMjVEfZMeBMdhXQ&t=2203): this is supposed to print the Triton kernel, good starting point
```python
# TORCH_LOGS = "OUTPUT_CODE" python square_compile.py

import torch
def square(a):
   a = torch.square(a)
   return torch.square(a)

opt_square = torch.compile(square)
opt_square(torch.randn(10000, 10000).cuda())
```

## Lecture 2 Ch1-3 PMPP book
1. Kernel coordinates
	1. 
	2. `blockIdx` and `threadIdx` allow threads to identify 
2. Telephone system analogy
	1. `blockIdx` as area code and `threadIdx` as the local phone number

### Indexing (First three exercises of Ch2)
1. Assume that we want to use each thread to calculate two adjacent elements of a vector addition. What would be the expression for mapping thread/block indices to the data index (i) of the first element to be processed by a thread?
	1. `i = (blockIdx.x * blockDim.x + threadIdx.x) * 2`
	2. The (`blockIdx`, `threadIdx`) identifies one thread, the RHS expression maps it to a certain data index `i`.

## Lecture 3: Getting Started With CUDA for Python Programmers
1. He basically showed even without knowing CUDA well, one can still write/use it with Pytorch.
2. How? *Write in Python, then convert to CUDA* ([lecture link](https://youtu.be/4sgKnKbR-WE?list=TLPQMjcwODIwMjVEfZMeBMdhXQ))
	1. Write CUDA-syntax `kernel` in Python, then ask GPT to convert it to `CUDA-C`
	2. `torch.load_inline` to convert `CUDA-C` into torch-compatible `module`.
3. we need allocation of memory to allow parallel computing
```python
def rgb2grey_py(x):
	c,h,w = x.shape
	n = h*w
	x = x.flatten()
	
	# need to allocate the memory before computing
	res = torch.empty(n, dtype=x.dtype, device=x.device)
	
	# all threads write to corresponding memory in parallel
	for i in range(n):
		res[i] = 0.2989*x[i] + 0.5870*x[i+n] + 0.1140*x[i+2*n]
	return res.view(h,w)
```

## Lecture 8: CUDA Performance Checklist
1. Calculation of Arithmetic Intensity (AI)
	1. Quantization increases AI
2. Thread coarsening
	1. Even a factor of *2* led to *10x* speed up
3. Different bounded and what to do?
	1. mem:
	2. compute: better algorithm, very hard to do.
### References
1. Mark's [deck](https://docs.google.com/presentation/d/1cvVpf3ChFFiY4Kf25S4e4sPY6Y5uRUO-X-A4nJ7IhFE/edit?slide=id.p#slide=id.p) and references within
2. 