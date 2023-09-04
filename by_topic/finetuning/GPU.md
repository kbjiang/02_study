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
	1. each process get the same architecture
	2. process 0 initializes the weights and then broadcast it to all other gpus
### gradient checkpointing
1. https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9