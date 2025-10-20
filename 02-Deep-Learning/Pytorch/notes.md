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

## `tensor.backward()`
### The Gradient Argument
- The `gradient` argument represents **∂(scalar_output)/∂(this_tensor)**. For when `this_tensor` is output itself, this argument is always `tensor(1.0)`.
	- comes handy during test when no loss is defined, can just do `o.backward(d_o)` where `d_o` is just a random tensor.
- By default, `backward()` assumes you're computing gradients of a **scalar** loss. But if your tensor is non-scalar, you need to specify what scalar you're actually differentiating.
#### Example 1: Scalar output - gradient argument defaults to 1.0
```python
import torch
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x.sum()  # y is scalar: y = x1 + x2 + x3
print(f"y = {y}")
y.backward()  # Equivalent to y.backward(torch.tensor(1.0))
print(f"∂y/∂x = {x.grad}")  # Should be [1, 1, 1]
```
#### Example 2: Non-scalar output - must provide gradient
```python
x = torch.tensor([1.0, 2.0, 3.0], requires_grad=True)
y = x * 2  # y is NOT scalar: y = [2, 4, 6]
print(f"y = {y}")
# This would error: y.backward()  # Error! y is not scalar
# We need to specify: if loss = y.sum(), what is ∂loss/∂y?
# ∂(y.sum())/∂y = [1, 1, 1]
y.backward(torch.tensor([1.0, 1.0, 1.0]))
print(f"∂(y.sum())/∂x = {x.grad}")  # Should be [2, 2, 2]
```
#### Key Insight:
This is used by chain rule to compute ∂(final_scalar_loss)/∂x:
```
∂loss/∂x = ∂loss/∂y * ∂y/∂x
           ↑           ↑
      gradient arg   computed by autograd
```

## How PyTorch works
1. [PyTorch Autograd Explained - In-depth Tutorial](https://www.youtube.com/watch?v=MswxJw-8PvE)
2. [PyTorch Hooks Explained - In-depth Tutorial](https://www.youtube.com/@elliotwaite)
	1. Mostly it's for accessing/editing gradients during backpropagation; `tensor.retain_grad` is considered a hook
	2. it can be for both tensor and module. 
	3. it's handles, which can be removed later.
3. [PyTorch internals : ezyang’s blog](https://blog.ezyang.com/2019/05/pytorch-internals/)