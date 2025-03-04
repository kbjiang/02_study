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
	1. Teach model to say "I don't know". Llama knowledge probing to detect knowledge boundary. https://youtu.be/7xTGNNLPyMI?t=4832&si=bCqiwQHsfnI_SfV_
	2. Use tools
	3. *vague recollection* vs *working memory*
		1. knowledge in the parameters == vague recollection (e.g. of something you read 1 month ago)
		2. knowledge in the tokens of the context window == working memory
3. Models need tokens to think https://youtu.be/7xTGNNLPyMI?t=6416&si=9iWt94PqoNIajcFY
	1. Good example of two *labels*. The left is worse coz all hangs on a *single* token `$3`, while the right has the calculations spread out, each step is easy enough. 
	2. *Basically the model needs tokens/multiple forward passes/intermediate steps to think*. If ask for only the answer, the model will have to squeeze many steps in a single forward pass, which can be challenging.
		1. tokens are like hints, leading the model to the right subspace
		2. ![[Pasted image 20250225232641.png|600]]
	3. A lot of "weakness" LLM has, such as bad at mental arithmetic and spelling, is because they only see tokens, not characters/numbers. E.g., 'strawberry' i  s actually `['st', 'raw', 'berry']`.
4. Very nice explanation on why we cannot predefine the learning paths for LLMs (by SFT/imitation) and RL is indispensable
	1. We human does not know how LLMs learn therefore SFT can be too restricting for things with steps like reasoning.
	2. Via RL, LLM overtime figures out the paths best suit itself
	3. Similarly, in the game of Go. Imitating expert players can only go so far, the model needs reinforcement learning to beat top human players.
5. RLHF
	1. can be run in arbitrary domains! (even the unverifiable ones outside of math/coding such as writing jokes)
	2. because the reward model is a NN, RL always finds a way to 'game' the RM and output nonsense like 'the the the the the'
		1. As a result, you cannot run RLHF for too long before it starts gaming the rewards
		2. *in contrast*, real RL like AlphaGO, with real rewards that are definite and can NOT be gamed, can train forever
6. Good resources
	1. LLM internal visualization https://bbycroft.net/llm
	2. Tokenization visualization https://tiktokenizer.vercel.app/
	3. Huggingface inference ground https://huggingface.co/spaces/huggingface/inference-playground
	4. Together.ai 
	5. Hyperbolic.ai for base models, i.e., before instruction finetuning

## What I don't agree

## Questions
