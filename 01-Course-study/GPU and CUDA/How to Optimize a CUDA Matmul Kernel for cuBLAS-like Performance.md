# [How to Optimize a CUDA Matmul Kernel for cuBLAS-like Performance: a Worklog](https://siboehm.com/articles/22/CUDA-MMM)
#CUDA
### Kernel 1: Naive Implementation
1. The `thredIdx` corresponds to indices in *output* matrix $C$. Indexing is 3D.
	1. ![[Pasted image 20250817075127.png|800]]
2. Calculation visualized
	1. ![[Pasted image 20250817074815.png|800]]
### Kernel 2: Global Memory Coalescing
1. Nice introduction to `warp` too.
2. Coalescing visualized
	1. In example below, when coalescing kernel loads `(0, 2)` element of `B` into global memory, `(0,3)` and `(0,4)` will be loaded as well due to coalescing!
		1. Also notice "Make sure these threads end up in same warp...", again the threads corresponds to output matrix!
	2. ![[Pasted image 20250817074003.png|800]]
3. In [code](https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/2_kernel_global_mem_coalesce.cuh), the data was indexed as 1D, i.e. `threadIdx.x / BLOCKSIZE` is of range `(0, BLOCKSIZE-1)`.

### Kernel 3: Shared Memory Cache-Blocking
1. "load a chunk of A and a chunk of B from global memory into *shared memory*. Then we’ll perform *as much work as possible* on the two chunks, with each thread still being *assigned one entry of C*. We’ll move the chunks along the columns of A and the rows of B performing partial sums on C until the result is computed."
	1. ![[Pasted image 20250817083959.png|800]]
	2. [code](https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/3_kernel_shared_mem_blocking.cuh)
		1. line 21-22 defines buffer in *shared memory*; line 35-40 enables `coalescing` in column direction. ![[Pasted image 20250817093155.png|700]]
### Kernel 4: 1D Blocktiling for Calculating Multiple Results per Thread
1. The idea is that each thread now calculate `TM` entries of `C` in the column direction (therefore 1D blocktiling), to reduce SMEM accesses.
2. it's important to keep in mind the dimension of `idx`, and which ones are for thread, which for matrices `A` and `B`.
	1. Similarly, how to map indexing between `As/Bs` and `A/B`
3. the original post misses a lot of information, see [[1D_Blocktiling_Explanation]] for more detail

![[Pasted image 20250817181110.png|700]]
![[Pasted image 20250817181322.png|700]]