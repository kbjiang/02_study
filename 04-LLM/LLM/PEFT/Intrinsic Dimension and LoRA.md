### Overview
I am reviewing three papers which are developed around the idea of intrinsic dimension (ID). Here is how they are connected. Also refer to the deck with same name I made for a reading group.
	1. [MEASURING THE INTRINSIC DIMENSION OF OBJECTIVE LANDSCAPES](https://arxiv.org/pdf/1804.08838.pdf)
		1. Proposed ID as a tool to understand objective landscape (measure how well a model fits a task)
		2. Proposed subspace training as a PEFT by accident
	2. [INTRINSIC DIMENSIONALITY EXPLAINS THE EFFECTIVENESS OF LANGUAGE MODEL FINE-TUNING](https://arxiv.org/pdf/2012.13255.pdf#page=9&zoom=100,110,132)
		1. Try to explain why *pretraining + finetuning* is so efficient through the lens of ID
	3. [LORA: LOW-RANK ADAPTATION OF LARGE LANGUAGE MODELS](https://arxiv.org/pdf/2106.09685.pdf)
		1. One instance of subspace finetuning

### Paper 1
1. Defined objective landscape
	1. It's determined by dataset and network architecture (with objective function)
	2. usually training can take any/all direction, as opposed to the subspace training
	3. paper seeks to understand the structure of such landscapes
2. To do so, proposed ID
	1. Decomposed full dimension into ID and solution set dimension $D = d_{int} + s$. 
		1. The main mathematical idea here is Degree of Freedom. Imagine the solution set in $\mathbb{R}^3$ is $x + y +z = 1$ and randomly initial point at $x_0, y_0, z_0$. Since the solution set is a hyperplane, we can stay at $x_0, y_0$ but only explore along $z$; we'll eventually hit the solution plane at $z^*=1-x_0-y_0$. 
	2. It's intractable to find $d_{int}$ analytically, therefore *subspace training*.
		1. $\theta^{(D)} = \theta^{(D)}_0 + P \theta^{(d)}, \theta^{(x)}\in \mathbb(R)^x, P\in\mathbb{R}^{D\times d}$. 
		2.  $P \in \mathbb{R}^{D\times d}$ forms an approximately orthonormal basis for a randomly oriented $d$ dimensional subaspace of $\mathbb{R}^D$. In other words, $\theta^{(d)}$ is just how we combine these $d$ vectors.  $\theta^{(d)}$ is initialized to be all zeros, so that the subspace has an origin at $\theta_0^{(D)}$.
		3. *These two facts makes PEFT like LoRA possible:* 
			1. low-rank subspace (therefore less DOF) for cheaper training, and 
			2. shifted origin to take advantage of pretrained $\theta_0^{(D)}$.  
		![[Pasted image 20230423162852.png]]
3. Interesting results/observations
	1. FC of different sizes have similar ID to classify MNIST. Suggests extra dimensions may go directly into solution set; possibly explains why large NNs are easier to train.
	2. LeNet has lower ID with MNIST than FC; however it become much larger after pixels are scrambled. This links to how *indcutive bias* can work for or against us.
	3. It requires exponentially more dims to go from 90% to 100% performance judging by figures in the papers. Probably why subspace training is not good for pretraining or when SOTA is desired.
4. Questions
	1. *What is the role of the frozen $\theta_0^{(D)}$?* It is involved in forward pass and gradient calculation, but how to understand intuitively?
	2. *Do I get a different solution topology from subspace training?* If I randomly project again on a subspace trained model, what do I get?
5. Refs
	1. [original post](https://www.uber.com/blog/intrinsic-dimension/)
	2. [original implementation](https://github.com/uber-research/intrinsic-dimension/blob/master/intrinsic_dim/](https://github.com/uber-research/intrinsic-dimension/blob/master/intrinsic_dim/)


### Paper 2
1. This paper tries to understand *pretraining and fine-tuning* via ID.
	1. replaces $\theta_0^{(D)}$ in paper 1 with pretrained weights.
	2. "One interpretation of the intrinsic parameter vector is that it encodes the task at hand with respect to the original pre-trained representations. Therefore, we can interpret $d$ as the minimal description length of the task within the framework dictated by the pre-trained representations (Hinton & Zemel, 1993). Under this interpretation of intrinsic dimensionality, we hypothesize that pre-training is implicitly lowering the intrinsic dimensionality of the average NLP task, and therefore compress the minimal description length of those same tasks."
2. Interesting results
	1. To prove their hypothesis, sec 5.1 shows ID descreases with increasing pretraining updates. ![[Pasted image 20230424130405.png]]
3. Minimal description length (MDL)
	1. The best point hypothesis H to explain data D is the one minimizes L(H)+L(D│H), where
		1. L(H) is the length of the description of H; and
		2. L(D|H) is the length of the description of the data when encoded with the help of H.
	2. For example, the best polynomial for fitting some data balances these two lengths ![[Pasted image 20230507144743.png]]
4. Questions
	1. Why random projecting pretrained model works? What's the topology look like? Must be some invariance.
5. Refs
	1. [MDL](https://arxiv.org/pdf/math/0406077.pdf)

### Paper 3
1. It's a natural extension from subspace training
	1. $W=W_0 + BA, B \in \mathbb{R}^{d\times r}, A \in \mathbb{R}^{r\times d}$. 
	2. meets the two requirements: $BA$ is low rank by design; $B$ initialized with zero.
	3. Speeding up comes from less gradient calculation. 
		1. No $\partial l/\partial W$ needed. $h=Wx+BAx \rightarrow \partial l /\partial B=\partial l/\partial h \otimes (Ax)$.
		2. Matmul with $W$ still involved in calculating $\partial l/\partial x$ with $x$ being previous layer input.
2. LoRA seems to be the best PEFT (See *Unified view of PEFT* for more detail)
	1. No additional inference latency: $W$ and $AB$ can be combined; contrary to Adapter (sequential)
	2. monotonically approaches full performance as $r$ increases![[Pasted image 20230508081919.png | 500]]
3. Refs
	1. [original implementation](https://github.com/microsoft/LoRA/blob/main/loralib/layers.py)
	2. [Huggingface implementation](https://github.com/huggingface/peft/blob/main/src/peft/tuners/lora.py)
	3. [Helpful review and discussion of the paper](https://openreview.net/forum?id=nZeVKeeFYf9)
	4. [Interesting post](https://kexue.fm/archives/9590)Even though some claims are not correct. 

### Related topics
1. Unified view of PEFT. See my other note.
2. Understanding of representation
	1. [https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/](https://colah.github.io/posts/2014-03-NN-Manifolds-Topology/)
	2. [Visualize NN](https://www.youtube.com/watch?v=UOvPeC8WOt8&ab_channel=vcubingx)
	3. [Deep Learning](https://www.nature.com/articles/nature14539)
	4. [Principles of Riemannian Geometry in Neural Networks](https://proceedings.neurips.cc/paper_files/paper/2017/file/0ebcc77dc72360d0eb8e9504c78d38bd-Paper.pdf)