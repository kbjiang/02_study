# [Computer Architecture - Lecture 30: SIMD and GPU Architectures (Fall 2024)](https://www.youtube.com/@OnurMutluLectures)
## SIMD processor
1. Good review of vectorization at [timestamp](https://youtu.be/8B-Pqa2_LP0?list=TLPQMTAwOTIwMjVM3Pfk9fIh0Q&t=6205)
	1. `vector processor` is frugal, i.e., just need num of ops processors which is much less than num of elements (`array processor`)
	2. Usually it's the combination of the two
		1. The `vector processor` on the left, indicated by `pipelined`; then it's parallelly executed in batch of four, i.e. `array processor`.
		2. ![[Pasted image 20250914073440.png|700]]
2. Enough banks with interleaving to overlap enough memory operations to cover memory latency.
	1. See example starting [timestamp](https://youtu.be/8B-Pqa2_LP0?list=TLPQMTAwOTIwMjVM3Pfk9fIh0Q&t=3703)
3. Fine-grained multithread (FGMT)  can hide long latency operations (e.g., memroy accesses)
## GPU
1. The *major advantage* of GPU, comparing to SIMD, is that it's SIMT executed and easier to program
	1. Note SIMT is *multiple* streams* of *scalar instructions*. ![[Pasted image 20250914083326.png|700]]
	2. Warps are not exposed to GPU programmers
	3. Dynamic warp formation/merging
2. GPU is SPMD programmed, i.e., same code but potential different programs (branching for example), but executed on SIMD processors
	1. programming model vs execution model
	2. SPMD is not SIMD in that it's NOT *vector instruction* but *scalar instructions* (i.e. threads) like C. 
	3. But when the hardware receives threads, it automatically converts them into *vector instructions* in unit of *warps*.
		1. warp`: a set of threads that execute the same instruction (i.e., at the same PC). Essentially, a SIMD operation formed by hardware.
		2. Can also regroup threads doing the same op at the same PC into the same wrap
		3. See [timestamp](https://youtu.be/zfru8aHZ44M?t=1698)
	4. ***vector instruction* is always synced? Probably not true...**
		1. Threads in different wraps *can* be doing different ops?? That's why SPMD needs *barriers* to sync.
		2. Threads in same wrap *cannot* be doing different ops because they are *vector instruction*?? Here threads are synced at every time stamp.

# [HetSys Course: Lecture 5: GPU Performance Considerations (Spring 2023)](https://www.youtube.com/@OnurMutluLectures)
1. Occupancy
	1. ![[Pasted image 20250914211132.png|700]]
2. Memory coalescing
	1. Takes time to move a row to row buffer; ensuing reading of rest of the row is fast (burst)
	2. Multi-bank can hide this latency
3. Data reuse: Shared memory
4. SIMD (Warp) untilization: Divergence
5. Other considerations...