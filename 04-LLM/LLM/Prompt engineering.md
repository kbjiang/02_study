### Learn to spell (2023-LLM-BootCamp, Lecture 3)
1. "Probabilistic programs" over *statistical pattern matcher*.
2. Prompt is
	1. a
	2. b
	3. c
3. ensembling
	1. one way to get right answer; infinite ways to get wrong answers
	2. inject randomness, such as upper/lower cases, for greater heterogeneity. Wrong answers are more likely to diverge.

1. OpenAI best practices https://help.openai.com/en/articles/6654000-best-practices-for-prompt-engineering-with-the-openai-api

### [Perspectives for first principles prompt engineering](https://huggingface.co/blog/KnutJaegersberg/first-principles-prompt-engineering)
1. There are roughly three (not entirely distinct) frames for prompt engineering:
	- LLMs as internet.zip or interpolative stochastic parrots
	- LLMs as more flexible dynamical systems
	- LLMs as pseudo-cognitive artifacts
2.  LLMs as flexible dynamical systems
	1. Stereotypical association between concepts in pretraining data and token completion
		1. Self-attention selects important tokens in the context that are indicative for completion. This abstraction of meaning guides the completion as an interpretation of the pretraining data. This only works when associated data is 'thick'.
		2. LLMs engage in ‘stereotypical reasoning’ try to interpret the prompt completion situation as ‘one of a kind’ according its interpretation of the pretraining data. It kinda violently stuffs the context with what seems like a typical completion to it, in a kinda retarded way - superhumanly stupid at times, as ‘reasoning’ works as well (or badly) for humanly simple and complex problems a like.
			1. Also related to control theory ([Bhargava et. al,](https://arxiv.org/abs/2310.04444))
			2. This is why exact wording of prompts is important.