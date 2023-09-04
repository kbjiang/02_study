### Huggingface
1. https://huggingface.co/docs/transformers/model_memory_anatomy
	1. 3 main groups of operations grouped by compute-intensity. Tensor contraction, statistical normalizations and element-wise ops
	2. 
2. memory calculation

### DDP
1. https://youtu.be/a6_pY9WwqdQ
	1. concepts of nodes, GPUs, worlds...
	2. recommends DDP over all others
	3. DDP vs DP:
		1. https://huggingface.co/docs/transformers/perf_train_gpu_many#dp-vs-ddp
		2. DP: https://youtu.be/a6_pY9WwqdQ?t=223

### gradient checkpointing
1. https://medium.com/tensorflow/fitting-larger-networks-into-memory-583e3c758ff9