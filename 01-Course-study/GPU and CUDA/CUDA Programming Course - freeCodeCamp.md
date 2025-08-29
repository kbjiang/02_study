### What makes GPUs so fast for deep learning?
![[Pasted image 20250814110923.png|800]]

- CPU (host)
    - minimize time of one task
    - metric: latency in seconds

- GPU (device)
    - maximize throughput
    - metric: throughput in tasks per second (ex: pixels per ms)

### Typical CUDA program
1. CPU allocates CPU memory
2. CPU copies data to GPU
3. CPU launches kernel on GPU (processing is done here)
	1. Kernels are to GPUs as programs are to CPUs.
4. CPU copies results from GPU back to CPU to do something useful with it

### Some terms to remember
- kernels (not popcorn, not convolutional kernels, not linux kernels, but GPU kernels)
- threads, blocks, and grid (next chapter)
	- *These are just software (CUDA) abstraction. The underlying hardware do NOT organize in grid/block/thread.*
- GEMM = **GE**neral **M**atrix **M**ultiplication
- SGEMM = **S**ingle precision (fp32) **GE**neral **M**atrix **M**ultiplication
- cpu/host/functions vs gpu/device/kernels
	- CPU is referred to as the host. It executes functions. 
	- GPU is referred to as the device. It executes GPU functions called kernels.

### Atomic
1. No more than one thread (a piece of code) can access the same memory address at the same time. 
	1. You can think of atomics as a very fast, hardware-level mutex operation. It's as if each atomic operation does this:
		1. lock(memory_location)
		2. old_value = \*memory_location
		3. \*memory_location = old_value + increment
		4. unlock(memory_location)
		5. return old_value
	2. Interesting example at this [timestamp](https://youtu.be/86FAWCzIe_4?t=13002).
		1. the wrong result shows how multiple threads access the same memory at the same time.

### API calls
1. To leverage highly optimized/compiled/binary CUDA libraries (e.g., `.so` lib) via calling their APIs
	1. So this is different `kernels`, which are self-defined functions
		1. API functions are faster than self written kernels
	2. *opaque struct type*
		1. no access to underlying code
2. `cuBLAS` (CUDA Basic Linear Algebra Subroutine) library,
	1. Linear Algebra operations, e.g., `matmul`
3. `cuDNN` 
	1. `cuda-course/06_CUDA_APIs/02 CUDNN/README.md`
	2. For DNN, things like matmul, conv, norm, dropout etc.
	3. `cudnn_graph`
		1. Think computation graph. `nodes` are operations and `edges` are tensors
	4. *pre-compiled*, *runtime fusion,* ...
4. Trying to understand `cudnn*Descriptor_t`.![[Pasted image 20250816110839.png|800]]

### PyTorch Extensions
1. Short section; see the example [here](https://github.com/kbjiang/cuda-course/tree/master/09_PyTorch_Extensions)
	1. it shows customized CUDA extension can be *faster* than PyTorch built-in. Nice!
	2. the build leads to error; Github Copilot solved it for me.

