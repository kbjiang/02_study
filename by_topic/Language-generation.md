### [How to generate](https://huggingface.co/blog/how-to-generate)
1. In short, auto-regressive language generation is based on the assumption that the probability distribution of a word sequence can be decomposed into the product of conditional next word distributions (the length $T$ is determined on-the-fly when the EOS token is generated):
$$P(w_T|w_{1:T-1}, W_0)=\prod_{t=1}^{T}P(w_t|w_{1:t-1}, W_0), \ w_{1:0}=\emptyset, $$
2. Decoding methods
	1. Greedy search
		1. misses high probability words hidden behind a low probability word
	2. Beam saerch
		1. strongly repetitive;
		2. works well in tasks where length of generation is predictable as in translation/summarization; not the case with dialogue and story generation
		3. human do not follow a distribution of high probability next words.
			![[Pasted image 20230409155528.png]]
	3. Sampling and its tricks
		1. `temperature` to adjust distribution; how is this related to the one in ChatGPT?
		2. *top-k*
			![[Pasted image 20230409171213.png]]
		3. *top-p*
			![[Pasted image 20230409171320.png]]
3. Miscellaneous
	1. `num_return_sequences>1` to get multiple sampled outputs
	2. `repetition_penalty` can prevent repetitions but is very sensitive to models/use cases
	3. `pad_token_id, bos_token_id, eos_token_id` need to be set manually if not in the model by default.