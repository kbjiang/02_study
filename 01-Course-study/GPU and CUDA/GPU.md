1. [*GPU Glossary*](https://modal.com/gpu-glossary) by Modal
	1. explains GPU concepts clearly and systematically

## Memory and its visualization
tags: #memory #gpu
1. [Transformer Memory Arithmetic: Understanding all the Bytes in nanoGPT](https://erees.dev/transformer-memory/)
	1. Detailed calculation and visualization of GPU memory usage in GPT2
	2. Memory breakdown ^gpt2mem
		1. Steady state memory usage
			1. Tensors persists in memory during training: weights, gradients and states
		2. Peak memory usage
			1. beginning of the backwards pass; in addition to the tensors above, the cached *activations* of the model
			2. output of each layer, logits etc.
	3. NOT really understand the Flame Graph; need to match it with the scripts such as `train.py`
2. Snapshot
	1. https://docs.pytorch.org/docs/stable/torch_cuda_memory.html
	2. https://pytorch.org/blog/understanding-gpu-memory-1/

## Training
1. [The Ultra-Scale Playbook:  Training LLMs on GPU Clusters](https://huggingface.co/spaces/nanotron/ultrascale-playbook?section=benchmarking_thousands_of_configurations) by HuggingFace