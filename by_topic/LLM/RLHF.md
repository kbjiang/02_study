1. NYU-CS2590 [lecture](https://nyu-cs2590.github.io/course-material/spring2023/lecture/lec10/main.pdf)
	1. three approaches to alignment: 
		1. prompting: unprincipled (no idea why works or not), unreliable
		2. supervised finetuning (SFT): sometimes hard to have unique label (e.g., there are multiple ways to write a piece of code for logistic regression.)
		3. RL: extra reward model
	2. RL in detail
			![[Pasted image 20230925201048.png]]	![[Pasted image 20230925201109.png]]
	3. *How is RL different from SFT?*
		1. SFT has unique $y$ for every $x$ while RLHF ranks multiple $y$s. The former kind of forces a single "optimal" solution, while the latter is more tolerant of "negative" solutions.
		2. SFT uses cross entropy as loss function, while RLHF maximizes rewards
1. Another NYU-CS2590 [lecture](https://docs.google.com/presentation/d/13Tylt2SvKvBL2hgILy5CmBtPDv3rXlVrQp01OzAe5Xo/edit#slide=id.p) with [video](https://youtu.be/zjrM-MW-0y0). 
	1. Basically more details on RLHF. He also thinks of this as another paradigm shift.![[Pasted image 20230925202915.png]]
	2. It's important to include KL term in loss to avoid over optimization of reward model. See Figure 5 of [reference](https://arxiv.org/pdf/2009.01325.pdf)
