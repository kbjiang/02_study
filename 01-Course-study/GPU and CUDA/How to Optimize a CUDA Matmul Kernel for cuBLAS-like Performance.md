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
3. In [code](https://github.com/siboehm/SGEMM_CUDA/blob/master/src/kernels/2_kernel_global_mem_coalesce.cuh), the thread was indexed as 1D, i.e. `threadIdx.x / BLOCKSIZE` is of range `(0, BLOCKSIZE-1)` but block is indexed 2D. 
> [!NOTE]- Detailed explanation
> ### Setup
> - `BLOCKSIZE = 2` (each block handles a 2×2 tile)
> 	- Matrix C is 3×3 
> 	- We need a 2×2 grid of blocks (since ceil(3/2) = 2)
> 	- `M = 3, N = 3`
> - Corresponding `dim3` definition at calling time
> 	```
> 	dim3 gridDim(CEIL_DIV(M, 2), CEIL_DIV(N, 2));
> 	dim3 blockDim(2 * 2);
> 	```
> ### The Problem
>	```
>	Matrix C (3×3):           Grid of Blocks (2×2):
>	┌─────┬─────┬─────┐       ┌─────────┬─────────┐
>	│C[0,0]│C[0,1]│C[0,2]│       │Block(0,0)│Block(0,1)│
>	├─────┼─────┼─────┤       ├─────────┼─────────┤
>	│C[1,0]│C[1,1]│C[1,2]│       │Block(1,0)│Block(1,1)│
>	├─────┼─────┼─────┤       └─────────┴─────────┘
>	│C[2,0]│C[2,1]│C[2,2]│       
>	└─────┴─────┴─────┘       
>	```
>	Notice the mismatch: we have 4 blocks but only need to cover a 3×3 matrix!
>	
> ### Threads Within Each Block
>	Each block has 4 threads (BLOCKSIZE² = 2² = 4):
> 	```
> 	threadIdx.x:  0  1  2  3
> 	Row (x/2):    0  0  1  1  
> 	Col (x%2):    0  1  0  1
> 	
> 	Thread layout within each block:
> 	┌───┬───┐
> 	│ 0 │ 1 │  <- threadIdx.x = 0,1
> 	├───┼───┤
> 	│ 2 │ 3 │  <- threadIdx.x = 2,3  
> 	└───┴───┘
> 	```
>
> ### Index Calculation Formula
> ```cpp
> const int cRow = blockIdx.x * BLOCKSIZE + (threadIdx.x / BLOCKSIZE);
> const int cCol = blockIdx.y * BLOCKSIZE + (threadIdx.x % BLOCKSIZE);
> ```
> 
> ### Block(0,1) - Top-right
> **Thread 0** (`threadIdx.x = 0`):
> - `cRow = 0 * 2 + (0 / 2) = 0`
> - `cCol = 1 * 2 + (0 % 2) = 2`
> - `cRow < 3 && cCol < 3` → **C[0,2] ✓**
>
> **Thread 1** (`threadIdx.x = 1`):
> - `cRow = 0 * 2 + (1 / 2) = 0`
> - `cCol = 1 * 2 + (1 % 2) = 3`
> - `cRow < 3 && cCol < 3` → **0 < 3 && 3 < 3 is FALSE** ❌
> 
> ### Final Mapping: matrix C with thread assignments:
> ```
> ┌─────────┬─────────┬─────────┐
> │Block(0,0)│Block(0,0)│Block(0,1)│
> │Thread 0 │Thread 1 │Thread 0 │
> │  C[0,0] │  C[0,1] │  C[0,2] │
> ├─────────┼─────────┼─────────┤
> │Block(0,0)│Block(0,0)│Block(0,1)│
> │Thread 2 │Thread 3 │Thread 2 │
> │  C[1,0] │  C[1,1] │  C[1,2] │
> ├─────────┼─────────┼─────────┤
> │Block(1,0)│Block(1,0)│Block(1,1)│
> │Thread 0 │Thread 1 │Thread 0 │
> │  C[2,0] │  C[2,1] │  C[2,2] │
> └─────────┴─────────┴─────────┘
> ```

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