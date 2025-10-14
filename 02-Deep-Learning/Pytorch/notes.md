## Distributed parallelism
1. [How Fully Sharded Data Parallel (FSDP) works?](https://www.youtube.com/@ahmedtaha8848)
	1. [Fully Sharded Data Parallel -- Public - Google Slides](https://docs.google.com/presentation/d/1ntPSYg-Wphl8sErwjUl0AztOY1i4SZmQuvmGhkeRElA/edit?slide=id.g2318fd43235_0_214#slide=id.g2318fd43235_0_214)
	2. [PyTorch FSDP: Experiences on Scaling Fully Sharded Data Parallel]([2304.11277](https://arxiv.org/pdf/2304.11277))
	3. "The memory requirements for FSDP are proportional to the size of the sharded model plus the size of the largest fully-materialized FSDP unit". 
		1. "sharded model": It's the 3rd row situation, the thin bar for each gpus ![[Pasted image 20251013203057.png|800]]
		2. "fully materialized FSDP". During the entire forward/backward pass, FSDP only needs to fully materialize one unit (see units in `Figure 1` below) at a time, while all other units can *stay sharded*.
	4. FSDP algorithm overview
		1. vertical split: the FSDP units
		2. horizontal split: partial tensors shown as half-filled layers
		3. ![[Pasted image 20251013204637.png|700]]
		4. In words ([Ref]([Getting Started with Fully Sharded Data Parallel (FSDP2) — PyTorch Tutorials 2.8.0+cu128 documentation](https://docs.pytorch.org/tutorials/intermediate/FSDP_tutorial.html)))
			1. Outside of forward and backward computation, parameters are fully sharded    
			2. Before forward and backward, sharded parameters are all-gathered into unsharded parameters    
			3. Inside backward, local unsharded gradients are reduce-scatterred into sharded gradients    
			4. Optimizer updates sharded parameters with sharded gradients, resulting in sharded optimizer states
	5. The overhead is not that high because of 
		1. the overlapping between GPU computation and communication; it's like *prefetching*.
		2. ![[Pasted image 20251013203215.png|900]]
		3. The `RS1` probably should be placed after `BWD1`
		4. FSDP unit0 is special, it was not freed during forward/backward pass until the end.
2. 
3. 

## How PyTorch works
1. [PyTorch Autograd Explained - In-depth Tutorial](https://www.youtube.com/watch?v=MswxJw-8PvE)
2. [PyTorch internals : ezyang’s blog](https://blog.ezyang.com/2019/05/pytorch-internals/)