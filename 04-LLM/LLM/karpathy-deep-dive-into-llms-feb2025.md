Date: 2025-02-06
Date of publish: 
Authors: Andrej Karpathy
Tags: #LLM #RL

Talk link: https://youtu.be/7xTGNNLPyMI
Paper link:
Related: 

## Main results
1. Very nice explanation on SFT to RL, with the example of learning chemistry
	1. pretraining: exposition to background knowledge, i.e., texts in the textbook
	2. SFT: imitation of human expert, i.e., examples in the textbook; blind mimicry.
	3. RL: reward only, no intermediates provided, trial and error. i.e., practice problems with check on final answer.
2. Hallucination mitigation
	1. Model learns to say "I don't know". Llama knowledge probing to detect knowledge boundary. https://youtu.be/7xTGNNLPyMI?t=4832&si=bCqiwQHsfnI_SfV_
	2. Use tools
	3. *vague recollection* vs *working memory*
		1. knowledge in the parameters == vague recollection (e.g. of something you read 1 month ago)
		2. knowledge in the tokens of the context window == working memory
3. Models need tokens to think https://youtu.be/7xTGNNLPyMI?t=6416&si=9iWt94PqoNIajcFY
	1. Good example of two *labels*. The left is worse coz all hangs on a *single* token `$3`, while the right has the calculations spread out, each step is easy enough. 
	2. *Basically the model needs tokens/multiple passes of data/intermediate steps to think*. If ask for only the answer, the model will have to squeeze many steps in one pass of the data, which can be challenging.
		1. ![[Pasted image 20250225232641.png|600]]
4. Very nice explanation on why we cannot predefine the learning paths for LLMs (by SFT/imitation) and RL is indispensable
	1. Via RL, LLM overtime figures out the paths best suit itself
5. Good resources
	2. LLM internal visualization https://bbycroft.net/llm
	3. Tokenization visualization https://tiktokenizer.vercel.app/
	4. Huggingface inference ground https://huggingface.co/spaces/huggingface/inference-playground

## What I don't agree

## Questions
