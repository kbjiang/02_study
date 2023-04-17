1. looks like chronically we have *Adapator --> LoRA --> PEFT*
2. Nice explanation covers all topics: https://youtu.be/YVU5wAA6Txo
3. it's independent of int8.
4. bitsandbytes

### Papers
1. Adapter:
		1. [Parameter-Efficient Transfer Learning for NLP](https://arxiv.org/pdf/1902.00751.pdf)
			1. memory efficient
			2. can train different tasks sequentially; no forgetting
		
2. LoRA
	1. propsed as an alternative to Adapters and Prefix Tuning (PT).
		1. Adapter is bad at parallelism and introduces inference latency?
		2. PT: hard to optimize; smaller effective sequence length by design
	2. why it works
		1. *When adapting to a specific task, Aghajanyan et al. (2020) shows that the pre-trained language models have a low “instrisic dimension” and can still learn efficiently despite a random projection to a smaller subspace.*
		2. SVD?
	3. LoRA only appies to attention weights, i.e., $W_q, W_k, W_v$, but not MLP weights $W_o$. 
	4. shards?

3. Unified view of PEFT
	1. Fig 1. Very nice summary. ![[Pasted image 20230417105135.png]]
	2. really nice recap of the Transformer Architecture in 2.2.


YiZhu: https://youtu.be/sh79Z8i15PI
Lisa: https://youtu.be/TwE2m6Z991s
	1. human prompting is not enough for steering models like GPT-2/BART. So add/train prefix-tuning at each layer of the Transformer while keep the pretrained parts frozen.
	2. 46:03, better extrapolation when most of the pretrained params are fixed. 
AdapterHub is a cute idea.