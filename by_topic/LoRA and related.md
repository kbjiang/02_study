1. propsed as an alternative to Adapters and Prefix Tuning (PT).
	1. Adapter is bad at parallelism and introduces inference latency?
	2. PT: hard to optimize; smaller effective sequence length by design
	3. Both adapter (which is a MLP) and PT (changes the token sequences) are less natural comparing to LoRA. LoRA is a low rank approximation, with greater rank, LoRA gets better; while the others has an optimal value.![[Pasted image 20230420101615.png]]
	4. no additional latency at inference, coz $W_0$, $U$ and $V$ can be combined into one at inference time.
2. How it works
	1. Theoretically it's equivalent to SVD of full weight update matrix $\Delta W$ and keep only first $k$ rank of it
	2. In practice, you just need two matrices with matching size and let them learn $\Delta W$. 
3. why it works
	1. *When adapting to a specific task, Aghajanyan et al. (2020) shows that the pre-trained language models have a low “instrisic dimension” and can still learn efficiently despite a random projection to a smaller subspace.*
4. Me:
	1. me: it's like residual connection? For finetuning, the weight changes are supposed to be minor and a lower rank is sufficient.
	2. shards?
	3. Table 6 shows that even very low ranks of 1 or 2 return decent result. Suggesting significant over parameterization?
5. [In gradient view](https://kexue.fm/archives/9590)
	1. Eq 2 makes sense mathematically; need to visualize the *back-propagate* graph to see how it's done while training.

[Li et al., 2018](https://arxiv.org/pdf/1804.08838.pdf)
1. Main idea/result:
	1. Defined *intrinsic dimension* of *objective landscape*.
	2. since $\theta^{(d)}$ is initialized to be all zeros, the subspacehas an origin at $\theta_0^{(D)}$. ![[Pasted image 20230423162852.png]]
	3. The main mathematical idea here is Degree of Freedom.
2. To train in a random subspace, $\theta_0^{(D)}$ is also randomly initialized in Eqn. 2. This was extended to be fixed (pretrained weights) in LoRA.
	1. It's interesting how having $\theta_0^{(D)}$ helps with subspace training even though it's not trainable. See results in section *Are random subspaces really more parameter-efficient for FC nets?* 
	2. it has to be the forward pass? or extra calculation at backward?
	3. In section *Are convnets always better on MNIST? Measuring dint90 on shuffled data*, it is shown that noisy distribution requires more DOF to model.
3. Features of subspace training
	1. it's architecture and dataset specific
	2. it operates in the entire parameter space. Recall that it's a random projection.
	3. does NOT speed up inference
	4. can be combined with quantization
4. Refs
	1. https://www.uber.com/blog/intrinsic-dimension/


[Aghajanyan et al., 2020](https://arxiv.org/pdf/2012.13255.pdf#page=9&zoom=100,110,132)
1. Li showed a method to measure the degree of redundancy, i.e., intrisic dimension. This paper applies this tool to understand pretraining and fine-tuning.
2. It claims that *the process of pre-training itself implicitly minimizes the intrinsic dimension of later tuning for different NLP tasks.* 
	1. As a result, *standard pre-trained models can learn a large set of NLP tasks with very few parameters*
	2. Finetuning using $\theta_d$: *One interpretation of the intrinsic parameter vector is that it encodes the task at hand with respect to the original pre-trained representations.*
3. Why *the process of pre-training itself implicitly minimizes the intrinsic dimension*
	1. it's shown that pretrained models are redundant in capacity and allows for significant sparsification.
	2. Sec 5.1 shows intrisic dimension descreases with increasing pretraining updates. Note 'full solution', against which $d_{90}$ is measured, at update steps are obtained by finetuning corresponding checkpoint model in the standard way.![[Pasted image 20230424130405.png]]
4. Therefore, *standard pre-trained models can learn a large set of NLP tasks with very few parameters*
5. 