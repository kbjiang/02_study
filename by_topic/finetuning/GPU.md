### Huggingface
1. https://huggingface.co/docs/transformers/model_memory_anatomy
	1. 3 main groups of operations grouped by compute-intensity. Tensor contraction, statistical normalizations and element-wise ops
	2. 
2. memory calculation

### DDP
1. https://youtu.be/a6_pY9WwqdQ
	1. concepts of nodes, process/gpu, worlds, global rank, local rank...
	2. recommends DDP over all others
	3. DDP vs DP:
		1. https://huggingface.co/docs/transformers/perf_train_gpu_many#dp-vs-ddp
		2. DP: https://youtu.be/a6_pY9WwqdQ?t=223
1. DDP https://youtu.be/SivkGd6LQoU
	1. concepts of nodes, process/gpu, worlds, global rank, local rank...
	2. So the idea is to define multiple processes with desired data and ops, then link each to a gpu, then spawn 
	3. Initialization
		1. each process get the same architecture
		2. process 0 initializes the weights and then broadcast it to all other gpus
2. https://youtu.be/3XUG7cjte2U
	1. API/COMM https://youtu.be/3XUG7cjte2U?t=307 ![[Pasted image 20230905153550.png]]
### gradient checkpointing
1. https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9

### Pytorch
1. Nice code walkthrough https://youtu.be/-LAtx9Q6DA8?t=487 The series is also good.

### Nvidia
1. https://youtu.be/azLCUayJJoQ?t=1453 Nice summary
MIT 6.S965
1. https://youtube.com/playlist?list=PL80kAHvQbh-ocildRaxjjBy6MR1ZsNCU7&si=7l0UEADknT1jdnmh