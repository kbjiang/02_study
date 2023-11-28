1. NYU-CS2590 [lecture](https://nyu-cs2590.github.io/course-material/spring2023/lecture/lec10/main.pdf)
	1. three approaches to alignment: 
		1. prompting: unprincipled (no idea why works or not), unreliable
		2. supervised finetuning (SFT): sometimes hard to have unique label (e.g., there are multiple ways to write a piece of code for logistic regression.)
		3. RL: extra reward model
	2. RL in detail
		1. Think of tokens as actions:
			- Action space: vocabulary   $a_t = x_t \in \mathcal{V}$
			- State space: history/prefix    $s_t=(x_1, ..., x_{t-1})$
			- Policy: the LM under training    $P_{\theta}(x_t | x_{<t})$
			- Trajectory: a sequence    $x_1,...,x_T$
			- Considered on-policy
		2. ![[Pasted image 20230925201109.png]]
	3. *How is RL different from SFT?*
		1. SFT has unique $y$ for every $x$ while RLHF ranks multiple $y$s. The former kind of forces a single "optimal" solution, while the latter is more tolerant of "negative" solutions.
		2. SFT uses cross entropy as loss function, while RLHF maximizes rewards
2. Another NYU-CS2590 [lecture](https://docs.google.com/presentation/d/13Tylt2SvKvBL2hgILy5CmBtPDv3rXlVrQp01OzAe5Xo/edit#slide=id.p) with [video](https://youtu.be/zjrM-MW-0y0). 
	1. Basically more details on RLHF. 
		1. things like *Bradley-Terry Model* which connects rewards to preferences.
	2. He also thinks of this as another paradigm shift.![[Pasted image 20230925202915.png]]
	3. It's important to include KL term in reward to avoid over optimization against reward model. See Figure 5 of [reference](https://arxiv.org/pdf/2009.01325.pdf)
		1. Note this term is *NOT* the ratio function in PPO algorithm, which is the ratio between the probabilities of taking same action at same state now and previously. See [HF course](https://huggingface.co/learn/deep-rl-course/unit8/clipped-surrogate-objective).
3. DPO instead of PPO
	1. Why not directly learn from preference samples (those RM model was trained on)?
	2. https://youtu.be/vuWbJlBePPA
4. Chip Huyen https://huyenchip.com/2023/05/02/rlhf.html