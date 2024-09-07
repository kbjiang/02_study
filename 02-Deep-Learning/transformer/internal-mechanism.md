### In-context learning/Meta-learning
1. [Papers I read](obsidian://open?vault=GitHub&file=03_paper_and_talk%2F2024%2F01b-What%20Learning%20Algorithm%20is%20In-Context%20Learning)

### Inner work
1. [Antropic monosemanticity](https://transformer-circuits.pub/2024/scaling-monosemanticity/index.html)
2. Knowledge Circuit
	1. This [video](https://youtu.be/qDgCLeDs4Kg) and references within can be a starting point.
	2. Good analysis using Knowledge Circuit starting 33:00
		1. ROME vs Finetuning, hallucination, ICL etc
	3. It's like understanding the human brain, like which lobe does what. Only here we have attention heads and MLPs.
3. [What does BERT look at](https://doi.org/10.48550/arXiv.1906.04341)
4. Stealing input data. https://youtu.be/WwbukAcMM4k

### MLP
1. [How might LLMs store facts](https://youtu.be/9-Jl0dxWQs8) by 3B1B
	1. MLP makes up 2/3 of the total parameters in TRF
	2. Intuition
		1. Input $\vec{E}$ is output of attn layer and is a superposition of semantic units, such as "Michael Jordan".
		2. The up-scale MLP layer $M_{\text{up}}$ of shape $4m\times m$ with each row as an *inquiry*, e.g. "is the 1st name Michael?", which inner products with $\vec{E}$.
			1. A row is all connections to one neuron (out of $4m$)
			2. Picture in MNIST, $4m$ neurons, each looks for a certain feature from each input of size $m$. If match, neuron lights up
		4. The down-scale MLP layer $M_{\text{down}}$ of shape $m\times 4m$ with each column as a feature*, e.g. "basketball", which gets added to output vector if a match.
		5. Think of the non-linear activation, usually ReLU, as turning floats into Yes/No questions.
2. Superposition
	1. A neuron almost never represents a single/clean concept like "1st name is Michael". It's usually a superposition of concepts.
	2. Number of *nearly* perpendicular (superposed) vectors in a high dim space are exponential of space dimension.
		1. Means MLP can hold way way more nearly orthogonal concepts than there are dimensions.
		2. https://transformer-circuits.pub/2022/toy_model/index.html