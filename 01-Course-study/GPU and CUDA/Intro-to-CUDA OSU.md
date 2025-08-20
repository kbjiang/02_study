## Intro to CUDA (part 1): High Level Concepts
1. CPU vs GPU
	1. CPU: low latency, large cache
	2. GPU: high throughput, many cores
2. CPU is the host, in charge of programs, only delegate highly parallelable tasks to GPU
	1. CUDA is heterogeneous, takes advantage of both Host and Device
3. CUDA is C with extensions
4. Kernel vs Threads
	1. Kernels execute as a set of parallel threads
	2. Threads are similar to data-parallel tasks
		1. Each thread performs the same operation on a subset of data
		2. Threads execute independently
		3. Threads do not execute at the same rate
			1. different paths taken in if/else statements
			2. different number of iterations in a loop etc
	3. Kernel is just a piece of program, that's why you can say things like "the grid of the 2nd kernel launch will not be scheduled to execute on the device until the 1st kernel has completed its execution."

## Intro to CUDA (part 2): Programming Model
1. CPU is in control of the flow of the program
	1. runs sequentially until a kernel is launched
		1. control was returned to host immediately
	2. main C function does not wait for the kernel to complete unless told explicitly (barrier)
		1. the `asynchronous` nature of CUDA programming

## Intro to CUDA (part 3): Parallelizing a For-Loop
1. `ThreadIdx` to distinguish threads
2. Thread hierarchy (*core design*)

## Intro to CUDA (part 5): Memory Model
1. *Thread-Memory correspondence (core design)*
	1. register/shared/global <> thread/block/grid
	2. register/shared are on chip, therefore much faster than global mem (DRAM).
		1. local memory is 'local' in scope, not location, it's actually off-chip and quite slow; its only for variable not fit in register (?) and should be avoid whenever possible.
2. both register and local memory are 'local'
	1. meaning it's associated with variables declared within the kernel and will be destroyed when the thread is done running
	2. in this example, `temp` is in register and `a` is in global
		```C
		//kernel defintion
		__global__ void kernel( int* a)
		{
			int i = threadIdx.x + blockIdx.x * blockDim.x;
			if( i < 3)
			{
				int temp = a[i+1];
				__syncthreads;
				a[i] = temp;
			}
		}
		```
3. Visualization
	1. ![[Pasted image 20250819220920.png|800]]

## Intro to CUDA (part 6): Synchronization
1. synchronization primitives (*core design*)